# -*- coding: utf-8 -*-

import argparse

from supar import BiaffineDependencyParser
from supar.cmds.cmd import parse


def main():
    parser = argparse.ArgumentParser(description='Create Biaffine Dependency Parser.')
    parser.add_argument('--tree', action='store_true',
                        help='whether to ensure well-formedness')
    parser.add_argument('--proj', action='store_true',
                        help='whether to projectivise the data')
    parser.set_defaults(Parser=BiaffineDependencyParser)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--include_char', action='store_true',
                           help='Include character embeddings as features')
    subparser.add_argument('--include_bert', action='store_true',
                           help='Include BERT embeddings as features')
    subparser.add_argument('--include_lstm', action='store_true',
                           help='Include LSTM embeddings as features. Typically used as a less powerful (but maybe '
                                'faster) replacement for BERT embeddings')
    subparser.add_argument('--lstm_emb_size', type=int, default=128,
                           help='Input contextual embedding size')
    subparser.add_argument('--include_upos', action='store_true',
                           help='Include UPOS embeddings as features')
    subparser.add_argument('--upos_emb_size', type=int, default=50)
    subparser.add_argument('--include_ufeats', action='store_true',
                           help='Include universal feature embeddings as features')
    subparser.add_argument('--ufeats_emb_size', type=int, default=30)
    subparser.add_argument('--n_lstm_hidden', type=int, default=400,
                           help='The hidden state size of the three layer LSTM used in parser')
    subparser.add_argument('--build', '-b', action='store_true',
                           help='whether to build the model first')
    subparser.add_argument('--punct', action='store_true',
                           help='whether to include punctuation')
    subparser.add_argument('--max-len', default=None, type=int,
                           help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int,
                           help='max num of buckets to use')
    subparser.add_argument('--train', default='data/ptb/train.conllx',
                           help='path to train file')
    subparser.add_argument('--dev', default='data/ptb/dev.conllx',
                           help='path to dev file')
    subparser.add_argument('--test', default='data/ptb/test.conllx',
                           help='path to test file')
    subparser.add_argument('--embed', default=None,
                           help='path to pretrained embeddings')
    subparser.add_argument('--unk', default='unk',
                           help='unk token in pretrained embeddings')
    subparser.add_argument('--n_embed', default=100, type=int,
                           help='dimension of embeddings')
    subparser.add_argument('--bert', default='bert-base-multilingual-uncased',
                           help='which bert model to use')
    subparser.add_argument('--patience', type=int, default=5,
                           help='Early stopping tolerance (optimizing mean of UAS and LAS)')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--punct', action='store_true',
                           help='whether to include punctuation')
    subparser.add_argument('--buckets', default=8, type=int,
                           help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.conllx',
                           help='path to dataset')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--prob', action='store_true',
                           help='whether to output probs')
    subparser.add_argument('--buckets', default=8, type=int,
                           help='max num of buckets to use')
    subparser.add_argument('--data', default='data/ptb/test.conllx',
                           help='path to dataset')
    subparser.add_argument('--pred', default='pred.conllx',
                           help='path to predicted result')
    parse(parser)


if __name__ == "__main__":
    main()
