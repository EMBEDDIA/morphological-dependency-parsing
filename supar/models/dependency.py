# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from supar.modules import (MLP, BertEmbedding, Biaffine, BiLSTM, CharLSTM,
                           Triaffine)
from supar.modules.dropout import IndependentDropout, SharedDropout
from supar.utils import Config
from supar.utils.alg import eisner, eisner2o, mst
from supar.utils.transform import CoNLL
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiaffineDependencyModel(nn.Module):
    """
    The implementation of Biaffine Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning (ICLR'17)
          Deep Biaffine Attention for Neural Dependency Parsing
          https://openreview.net/pdf?id=Hk95PK9le/

    Args:
        n_words (int):
            Size of the word vocabulary.
        n_char_feats (int):
            Size of the character vocabulary.
        n_upos_feats (int):
            Size of the UPOS vocabulary.
        n_ufeats (dict):
            Sizes of the universal features vocabularies. Maps a feature name (e.g. Mood) to number of possible values
            (e.g. 12).
        n_rels (int):
            Number of labels in the treebank.
        feats (iterable):
            Specifies which type of additional features to use: 'char' | 'bert' | 'lstm' | 'upos' | 'ufeats'.
            'char': Character-level representations extracted by CharLSTM.
            'bert': BERT representations, other pretrained language models like `XLNet` are also feasible.
            'lstm': LSTM representations, obtained by passing the word embeddings through an additional LSTM.
            'upos': Universal POS tag embeddings.
            'ufeats': Universal feature embeddings.
            Default: None.
        n_embed (int):
            Size of word embeddings. Default: 100.
        n_bert_embed (int):
            Size of BERT embeddings: if 0, use the hidden size from BERT's config.
        n_upos_embed (int):
            Size of the UPOS embeddings.
        n_char_embed (int):
            Size of character embeddings serving as inputs of CharLSTM, required if feat='char'. Default: 50.
        n_lstm_embed (int):
            Size of contextual embeddings, that are produced by passing the word embeddings through an additional LSTM.
            Default: 128.
        n_ufeats_embed (dict):
            Sizes of universal feature embeddings. Maps a feature name to its embedding size.
        bert (str):
            Specify which kind of language model to use, e.g., 'bert-base-cased' and 'xlnet-base-cased'.
            This is required if feat='bert'. The full list can be found in `transformers`.
            Default: `None`.
        n_bert_layers (int):
            Specify how many last layers to use: if 0, use all layers.
            The final outputs would be the learned weighted sum of the hidden states of these layers.
            Default: 0.
        mix_dropout (float):
            Dropout ratio of BERT layers. Required if feat='bert'. Default: .0.
        embed_dropout (float):
            Dropout ratio of input embeddings. Default: .33.
        n_lstm_hidden (int):
            Dimension of LSTM hidden states. Default: 400.
        n_lstm_layers (int):
            Number of LSTM layers. Default: 3.
        lstm_dropout (float): Default: .33.
            Dropout ratio of LSTM.
        n_mlp_arc (int):
            Arc MLP size. Default: 500.
        n_mlp_rel  (int):
            Label MLP size. Default: 100.
        mlp_dropout (float):
            Dropout ratio of MLP layers. Default: .33.
        char_pad_index (int):
            The index of the padding token in the char vocabulary. Default: 0.
        bert_pad_index (int):
            The index of the padding token in the BERT vocabulary. Default: 0.
        upos__pad_index (int):
            The index of the padding token in the UPOS vocabulary. Default: 0.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.
    """
    def __init__(self, n_words,
                 n_char_feats,
                 n_upos_feats,
                 n_ufeats,
                 n_rels,
                 feats=None,
                 n_embed=100,
                 n_bert_embed=0,
                 n_upos_embed=50,
                 n_char_embed=50,
                 n_lstm_embed=128,
                 n_ufeats_embed=None,  # dict: feature_name -> embedding_size
                 bert=None,
                 n_bert_layers=0,
                 mix_dropout=.0,
                 embed_dropout=.33,
                 n_lstm_hidden=400,
                 n_lstm_layers=3,
                 lstm_dropout=.33,
                 n_mlp_arc=500,
                 n_mlp_rel=100,
                 mlp_dropout=.33,
                 char_pad_index=0,
                 bert_pad_index=0,
                 upos_pad_index=0,
                 pad_index=0,
                 unk_index=1,
                 **kwargs):
        super().__init__()

        self.args = Config().update(locals())

        if self.args.feats is None:
            self.args.feats = []

        if self.args.n_ufeats_embed is None:
            self.args.n_ufeats_embed = {}

        assert len(self.args.n_ufeats) == len(self.args.n_ufeats_embed)

        # the embedding layer
        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)

        additional_features_size = 0
        if 'char' in self.args.feats:
            self.char_embed = CharLSTM(n_chars=n_char_feats,
                                       n_embed=n_char_embed,
                                       n_out=n_char_embed,
                                       pad_index=char_pad_index)
            additional_features_size += n_char_embed

        if 'bert' in self.args.feats:
            self.bert_embed = BertEmbedding(model=bert,
                                            n_layers=n_bert_layers,
                                            n_out=n_bert_embed,
                                            pad_index=bert_pad_index,
                                            dropout=mix_dropout,
                                            requires_grad=True)
            # Get the actual embedding size after loading BERT
            self.n_bert_embed = self.bert_embed.n_out
            additional_features_size += self.n_bert_embed

        if 'upos' in self.args.feats:
            self.upos_embed = nn.Embedding(num_embeddings=n_upos_feats,
                                           embedding_dim=n_upos_embed)
            additional_features_size += n_upos_embed

        if 'lstm' in self.args.feats:
            self.lstm_embed = nn.LSTM(input_size=n_embed,
                                      hidden_size=n_lstm_embed,
                                      batch_first=True)
            additional_features_size += n_lstm_embed

        self.ufeats_order = []
        if 'ufeats' in self.args.feats:
            self.ufeats_embed = nn.ModuleDict()
            # Fix the order in which universal feature embeddings will be concatenated with other features
            # (because no amount of documentation will make me trust that dicts are ordered)
            self.ufeats_order = list(self.args.n_ufeats_embed.keys())
            for feature_name, emb_size in self.args.n_ufeats_embed.items():
                self.ufeats_embed[feature_name] = nn.Embedding(num_embeddings=self.args.n_ufeats[feature_name],
                                                               embedding_dim=self.args.n_ufeats_embed[feature_name])
                additional_features_size += emb_size

        self.embed_dropout = IndependentDropout(p=embed_dropout)

        # the lstm layer
        self.lstm = BiLSTM(input_size=n_embed+additional_features_size,
                           hidden_size=n_lstm_hidden,
                           num_layers=n_lstm_layers,
                           dropout=lstm_dropout)
        self.lstm_dropout = SharedDropout(p=lstm_dropout)

        # the MLP layers
        self.mlp_arc_d = MLP(n_in=n_lstm_hidden*2,
                             n_out=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_arc_h = MLP(n_in=n_lstm_hidden*2,
                             n_out=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=n_lstm_hidden*2,
                             n_out=n_mlp_rel,
                             dropout=mlp_dropout)
        self.mlp_rel_h = MLP(n_in=n_lstm_hidden*2,
                             n_out=n_mlp_rel,
                             dropout=mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=n_mlp_rel,
                                 n_out=n_rels,
                                 bias_x=True,
                                 bias_y=True)
        self.criterion = nn.CrossEntropyLoss()
        self.pad_index = pad_index
        self.unk_index = unk_index

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self, words, char_feats=None, bert_feats=None, upos_feats=None, **ufeats):
        """
        Args:
            words (torch.LongTensor) [batch_size, seq_len]:
                The word indices.
            char_feats (torch.LongTensor) [batch_size, seq_len, fix_len]:
                Character indices to be embedded.
            bert_feats (torch.LongTensor) [batch_size, seq_len, fix_len]:
                Subword indices to be embedded.
            upos_feats (torch.LongTensor) [batch_size, seq_len]:
                Universal POS tag indices to be embedded.

        Returns:
            s_arc (torch.Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible arcs.
            s_rel (torch.Tensor): [batch_size, seq_len, seq_len, n_labels]
                The scores of all possible labels on each arc.
        """

        batch_size, seq_len = words.shape
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        additional_features = []
        # get outputs from embedding layers - [batch_size, seq_len, n_embed]
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
        if hasattr(self, 'lstm_embed'):
            hidden_features, _ = self.lstm_embed(word_embed)
            additional_features.append(self.embed_dropout(hidden_features)[0])
        word_embed = self.embed_dropout(word_embed)[0]

        if char_feats is not None:
            additional_features.append(self.embed_dropout(self.char_embed(char_feats))[0])
        if bert_feats is not None:
            additional_features.append(self.embed_dropout(self.bert_embed(bert_feats))[0])
        if upos_feats is not None:
            additional_features.append(self.embed_dropout(self.upos_embed(upos_feats))[0])

        for feature_name in self.ufeats_order:
            curr_embedder = self.ufeats_embed[feature_name]
            embedded_data = curr_embedder(ufeats[feature_name])
            additional_features.append(self.embed_dropout(embedded_data)[0])

        # concatenate the word and feat representations
        if len(additional_features) > 0:
            embed = torch.cat((word_embed, *additional_features), dim=-1)
        else:
            embed = word_embed

        x = pack_padded_sequence(embed, mask.sum(1).cpu(), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        # apply MLPs to the BiLSTM output states
        arc_d = self.mlp_arc_d(x)
        arc_h = self.mlp_arc_h(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        # set the scores that exceed the length of each sentence to -inf
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_arc, s_rel

    def loss(self, s_arc, s_rel, arcs, rels, mask):
        """
        Args:
            s_arc (torch.Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible arcs.
            s_rel (torch.Tensor): [batch_size, seq_len, seq_len, n_labels]
                The scores of all possible labels on each arc.
            arcs (torch.LongTensor): [batch_size, seq_len]
                Tensor of gold-standard arcs.
            rels (torch.LongTensor): [batch_size, seq_len]
                Tensor of gold-standard labels.
            mask (torch.BoolTensor): [batch_size, seq_len, seq_len]
                Mask for covering the unpadded tokens.

        Returns:
            loss (torch.Tensor): scalar
                The training loss.
        """

        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)

        return arc_loss + rel_loss

    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        """
        Args:
            s_arc (torch.Tensor): [batch_size, seq_len, seq_len]
                The scores of all possible arcs.
            s_rel (torch.Tensor): [batch_size, seq_len, seq_len, n_labels]
                The scores of all possible labels on each arc.
            mask (torch.BoolTensor): [batch_size, seq_len, seq_len]
                Mask for covering the unpadded tokens.
            tree (bool):
                If True, ensures to output well-formed trees. Default: False.
            proj (bool):
                If True, ensures to output projective trees. Default: False.

        Returns:
            arc_preds (torch.Tensor): [batch_size, seq_len]
                The predicted arcs.
            rel_preds (torch.Tensor): [batch_size, seq_len]
                The predicted labels.
        """

        lens = mask.sum(1)
        # prevent self-loops
        s_arc.diagonal(0, 1, 2).fill_(float('-inf'))
        arc_preds = s_arc.argmax(-1)
        bad = [not CoNLL.istree(seq[1:i+1], proj)
               for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            alg = eisner if proj else mst
            arc_preds[bad] = alg(s_arc[bad], mask[bad])
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)

        return arc_preds, rel_preds
