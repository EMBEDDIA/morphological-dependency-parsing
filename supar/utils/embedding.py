# -*- coding: utf-8 -*-

import torch


class Embedding(object):

    def __init__(self, tokens, vectors, unk=None):
        self.tokens = tokens
        self.vectors = torch.tensor(vectors)
        self.pretrained = {w: v for w, v in zip(tokens, vectors)}
        self.unk = unk

        # If UNK is not in the embeddings, add it
        if self.vectors.shape[0] > 0 and \
                self.unk is not None and \
                self.unk not in self.pretrained:
            randomly_initialized = torch.rand((self.dim,))
            self.pretrained[self.unk] = randomly_initialized.tolist()
            self.tokens = self.tokens + (self.unk,)
            self.vectors = torch.cat((self.vectors, randomly_initialized.unsqueeze(0)), dim=0)

    def __len__(self):
        return len(self.tokens)

    def __contains__(self, token):
        return token in self.pretrained

    @property
    def dim(self):
        return self.vectors.size(1)

    @property
    def unk_index(self):
        if self.unk is not None:
            return self.tokens.index(self.unk)
        else:
            raise AttributeError

    @classmethod
    def load(cls, path, unk=None, dim=100):
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line for line in f]
        splits = [line.split() for line in lines]
        tokens, vectors = zip(*[(" ".join(s[: -dim]), list(map(float, s[-dim:])))
                                for s in splits])

        return cls(tokens, vectors, unk=unk)
