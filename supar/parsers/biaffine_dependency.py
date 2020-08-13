# -*- coding: utf-8 -*-

import os

import torch
import torch.nn as nn
from supar.models import BiaffineDependencyModel
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, pad, unk
from supar.utils.field import Field, UFeatsField, SubwordField
from supar.utils.fn import ispunct
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import AttachmentMetric
from supar.utils.transform import CoNLL
from supar.utils.universal import UNIVERSAL_FEATURES

logger = get_logger(__name__)


class BiaffineDependencyParser(Parser):
    """
    The implementation of Biaffine Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning (ICLR'17)
          Deep Biaffine Attention for Neural Dependency Parsing
          https://openreview.net/pdf?id=Hk95PK9le/
    """

    NAME = 'biaffine-dependency'
    MODEL = BiaffineDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.WORD, self.CHAR_FEAT, self.BERT_FEAT = self.transform.FORM  # type: Field
        self.UPOS_FEAT = self.transform.CPOS  # type: Field
        # contains a `Field` (or `None` if not present) for each universal feature
        self.UFEATS_FEAT = self.transform.FEATS  # type: list

        self.ARC, self.REL = self.transform.HEAD, self.transform.DEPREL
        self.puncts = torch.tensor([i
                                    for s, i in self.WORD.vocab.stoi.items()
                                    if ispunct(s)]).to(self.args.device)

    def train(self, train, dev, test, buckets=32, batch_size=5000,
              punct=False, tree=False, proj=False, verbose=True, **kwargs):
        """
        Args:
            train, dev, test (list[list] or str):
                the train/dev/test data, both list of instances and filename are allowed.
            buckets (int):
                Number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                Number of tokens in each batch. Default: 5000.
            punct (bool):
                If False, ignores the punctuations during evaluation. Default: False.
            tree (bool):
                If True, ensures to output well-formed trees. Default: False.
            proj (bool):
                If True, ensures to output projective trees. Default: False.
            verbose (bool):
                If True, increases the output verbosity. Default: True.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self, data, buckets=8, batch_size=5000,
                 punct=False, tree=True, proj=False, verbose=True, **kwargs):
        """
        Args:
            data (str):
                The data to be evaluated.
            buckets (int):
                Number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                Number of tokens in each batch. Default: 5000.
            punct (bool):
                If False, ignores the punctuations during evaluation. Default: False.
            tree (bool):
                If True, ensures to output well-formed trees. Default: False.
            proj (bool):
                If True, ensures to output projective trees. Default: False.
            verbose (bool):
                If True, increases the output verbosity. Default: True.
            kwargs (dict):
                A dict holding the unconsumed arguments.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self, data, pred=None, buckets=8, batch_size=5000,
                prob=False, tree=True, proj=False, verbose=True, **kwargs):
        """
        Args:
            data (list[list] or str):
                The data to be predicted, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: None.
            buckets (int):
                Number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                Number of tokens in each batch. Default: 5000.
            prob (bool):
                If True, outputs the probabilities. Default: False.
            tree (bool):
                If True, ensures to output well-formed trees. Default: False.
            proj (bool):
                If True, ensures to output projective trees. Default: False.
            verbose (bool):
                If True, increases the output verbosity. Default: True.
            kwargs (dict):
                A dict holding the unconsumed arguments.

        Returns:
            A Dataset object that stores the predicted results.
        """

        return super().predict(**Config().update(locals()))

    def _train(self, loader):
        self.model.train()

        bar, metric = progress_bar(loader), AttachmentMetric()

        # words[, char_feats, bert_feats, upos_feats, <...ufeats...>], arcs, rels
        for data in bar:
            words, arcs, rels = data[0], data[-2], data[-1]
            char_feats, bert_feats, upos_feats, ufeats = None, None, None, {}
            n_optional = len(data) - 3  # number of optional features present
            if n_optional > 0:
                popped_features = 0
                if self.CHAR_FEAT is not None:
                    char_feats = data[1 + popped_features]
                    popped_features += 1

                if self.BERT_FEAT is not None:
                    bert_feats = data[1 + popped_features]
                    popped_features += 1

                if self.UPOS_FEAT is not None:
                    upos_feats = data[1 + popped_features]
                    popped_features += 1

                for feature_name, curr_field in zip(UNIVERSAL_FEATURES, self.UFEATS_FEAT):
                    if curr_field is None:
                        continue

                    ufeats[feature_name] = data[1 + popped_features]
                    popped_features += 1

            self.optimizer.zero_grad()

            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.model(words, char_feats=char_feats, bert_feats=bert_feats, upos_feats=upos_feats,
                                      **ufeats)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            self.optimizer.step()
            self.scheduler.step()

            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            metric(arc_preds, rel_preds, arcs, rels, mask)
            bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}")

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, AttachmentMetric()

        for data in loader:
            words, arcs, rels = data[0], data[-2], data[-1]
            char_feats, bert_feats, upos_feats, ufeats = None, None, None, {}
            n_optional = len(data) - 3  # number of optional features present
            if n_optional > 0:
                popped_features = 0
                if self.CHAR_FEAT is not None:
                    char_feats = data[1 + popped_features]
                    popped_features += 1

                if self.BERT_FEAT is not None:
                    bert_feats = data[1 + popped_features]
                    popped_features += 1

                if self.UPOS_FEAT is not None:
                    upos_feats = data[1 + popped_features]
                    popped_features += 1

                for feature_name, curr_field in zip(UNIVERSAL_FEATURES, self.UFEATS_FEAT):
                    if curr_field is None:
                        continue

                    ufeats[feature_name] = data[1 + popped_features]
                    popped_features += 1

            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_arc, s_rel = self.model(words, char_feats=char_feats, bert_feats=bert_feats, upos_feats=upos_feats,
                                      **ufeats)
            loss = self.model.loss(s_arc, s_rel, arcs, rels, mask)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask,
                                                     self.args.tree,
                                                     self.args.proj)
            # ignore all punctuation if not specified
            if not self.args.punct:
                mask &= words.unsqueeze(-1).ne(self.puncts).all(-1)
            total_loss += loss.item()
            metric(arc_preds, rel_preds, arcs, rels, mask)
        total_loss /= len(loader)

        return total_loss, metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        preds = {}
        arcs, rels, probs = [], [], []
        for data in progress_bar(loader):
            words = data[0]
            char_feats, bert_feats, upos_feats, ufeats = None, None, None, {}
            n_optional = len(data) - 3  # number of optional features present
            if n_optional > 0:
                popped_features = 0
                if self.CHAR_FEAT is not None:
                    char_feats = data[1 + popped_features]
                    popped_features += 1

                if self.BERT_FEAT is not None:
                    bert_feats = data[1 + popped_features]
                    popped_features += 1

                if self.UPOS_FEAT is not None:
                    upos_feats = data[1 + popped_features]
                    popped_features += 1

                for feature_name, curr_field in zip(UNIVERSAL_FEATURES, self.UFEATS_FEAT):
                    if curr_field is None:
                        continue

                    ufeats[feature_name] = data[1 + popped_features]
                    popped_features += 1

            mask = words.ne(self.WORD.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(1).tolist()
            s_arc, s_rel = self.model(words, char_feats=char_feats, bert_feats=bert_feats, upos_feats=upos_feats)
            arc_preds, rel_preds = self.model.decode(s_arc, s_rel, mask,
                                                     self.args.tree,
                                                     self.args.proj)
            arcs.extend(arc_preds[mask].split(lens))
            rels.extend(rel_preds[mask].split(lens))
            if self.args.prob:
                arc_probs = s_arc.softmax(-1)
                probs.extend([prob[1:i+1, :i+1].cpu() for i, prob in zip(lens, arc_probs.unbind())])
        arcs = [seq.tolist() for seq in arcs]
        rels = [self.REL.vocab[seq.tolist()] for seq in rels]
        preds = {'arcs': arcs, 'rels': rels}
        if self.args.prob:
            preds['probs'] = probs

        return preds

    @classmethod
    def build(cls, path, min_freq=2, fix_len=20, **kwargs):
        """
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default: 2.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.

        Returns:
            The created parser.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Load pretrained parser if it exists
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        used_features = set()
        form_fields = []  # different ways to represent 'form' tokens, contains a Field or None
        cpos_field = None

        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        form_fields.append(WORD)
        logger.info("Using word embeddings")

        CHAR_FEAT, BERT_FEAT, UPOS_FEAT = None, None, None
        if args.include_char:
            CHAR_FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos, fix_len=args.fix_len)
            logger.info("Using character embeddings")
            used_features.add("char")
            logger.warning("The size of character embeddings is hardcoded to 50.")
        form_fields.append(CHAR_FEAT)

        if args.include_bert:
            logger.info(f"Using BERT embeddings ({args.bert})")
            used_features.add("bert")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.bert)
            BERT_FEAT = SubwordField('bert',
                                     pad=tokenizer.pad_token,
                                     unk=tokenizer.unk_token,
                                     bos=tokenizer.bos_token or tokenizer.cls_token,
                                     fix_len=args.fix_len,
                                     tokenize=tokenizer.tokenize)
            BERT_FEAT.vocab = tokenizer.get_vocab()
        form_fields.append(BERT_FEAT)

        # Note: the LSTM will encode context from non-contextual embeddings so no need to build a custom field
        if args.include_lstm:
            logger.info(f"Using LSTM contextual embeddings (size = {args.lstm_emb_size})")
            used_features.add("lstm")

        if args.include_upos:
            logger.info(f"Using UPOS embeddings (size = {args.upos_emb_size})")
            used_features.add("upos")
            UPOS_FEAT = Field('tags', bos=bos)
        cpos_field = UPOS_FEAT

        ufeats_fields = [None] * len(UNIVERSAL_FEATURES)
        ufeats_sizes = {}
        if args.include_ufeats:
            logger.info(f"Using universal features (size = {args.ufeats_emb_size})")
            used_features.add("ufeats")
            for i, feature_name in enumerate(UNIVERSAL_FEATURES):
                ufeats_sizes[feature_name] = args.ufeats_emb_size

                FEATURE_FIELD = UFeatsField(feature_name, bos=bos, unk=unk, pad=pad)
                ufeats_fields[i] = FEATURE_FIELD

        ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
        REL = Field('rels', bos=bos)

        transform = CoNLL(FORM=form_fields, CPOS=cpos_field, FEATS=ufeats_fields, HEAD=ARC, DEPREL=REL)

        train = Dataset(transform, args.train)
        WORD.build(train, args.min_freq, (Embedding.load(args.embed, args.unk, dim=args.n_embed) if args.embed else None))
        if CHAR_FEAT is not None:
            CHAR_FEAT.build(train)
        if BERT_FEAT is not None:
            BERT_FEAT.build(train)
        if UPOS_FEAT is not None:
            UPOS_FEAT.build(train)
        ufeats_vocab_sizes = {}
        for i, feature_name in enumerate(UNIVERSAL_FEATURES):
            curr_field = ufeats_fields[i]
            if curr_field is not None:
                curr_field.build(train)
                ufeats_vocab_sizes[feature_name] = len(curr_field.vocab)

        REL.build(train)
        args.update({
            'n_words': WORD.vocab.n_init,
            'n_char_feats': len(CHAR_FEAT.vocab) if CHAR_FEAT is not None else 0,
            'n_upos_feats': len(UPOS_FEAT.vocab) if UPOS_FEAT is not None else 0,
            'n_ufeats': ufeats_vocab_sizes,
            'feats': used_features,
            'n_embed': args.n_embed,
            'n_bert_embed': 0,  # Note: 0 means the pretrained hidden size is used
            'n_upos_embed': args.upos_emb_size,
            'n_char_embed': 50,
            'n_lstm_embed': args.lstm_emb_size,
            'n_ufeats_embed': ufeats_sizes,  # map universal feature names to embedding sizes
            'n_rels': len(REL.vocab),
            'bert': args.bert,
            'n_bert_layers': 0,
            'pad_index': WORD.pad_index,
            'unk_index': WORD.unk_index,
            'bos_index': WORD.bos_index,
            'char_pad_index': CHAR_FEAT.pad_index if CHAR_FEAT is not None else None,
            'bert_pad_index': BERT_FEAT.pad_index if BERT_FEAT is not None else None,
            'upos_pad_index': UPOS_FEAT.pad_index if UPOS_FEAT is not None else None
        })
        model = cls.MODEL(**args)
        model.load_pretrained(WORD.embed).to(args.device)
        return cls(args, model, transform)
