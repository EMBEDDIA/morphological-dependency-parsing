# -*- coding: utf-8 -*-

import argparse

from supar import Config
from supar.parsers import (biaffine_parser, crf2o_dependency, crf_constituency,
                           crf_dependency, mst_dependency)

PARSERS = {'biaffine-parser': biaffine_parser,
           'crf2o-dependency': crf2o_dependency,
           'crf-constituency': crf_constituency,
           'crf-dependency': crf_dependency,
           'mst-dependency': mst_dependency}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create syntactic parsing models.'
    )
    parser.add_argument('--conf', '-c', default='config.ini',
                        help='path to config file')
    parser.add_argument('--model', default='biaffine-parser',
                        choices=PARSERS.keys())
    parser.add_argument('--backend', default='nccl',
                        choices=['gloo', 'nccl'])
    args, _ = parser.parse_known_args()
    PARSERS[args.model].run(Config(args.conf).update(vars(args)))
