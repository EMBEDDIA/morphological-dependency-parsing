# SuPar

[![GitHub actions](https://github.com/yzhangcs/parser/workflows/build/badge.svg)](https://github.com/yzhangcs/parser/actions)
[![GitHub stars](https://img.shields.io/github/stars/yzhangcs/parser.svg)](https://github.com/yzhangcs/parser/stargazers)		
[![GitHub forks](https://img.shields.io/github/forks/yzhangcs/parser.svg)](https://github.com/yzhangcs/parser/network/members)
[![LICENSE](https://img.shields.io/github/license/yzhangcs/parser.svg)](https://github.com/yzhangcs/parser/blob/master/LICENSE)

`SuPar` provides a collection of state-of-the-art syntactic parsing models with Biaffine Parser ([Dozat and Manning, 2017](#dozat-2017-biaffine)) as the basic architecture:
* Biaffine Dependency Parser ([Dozat and Manning, 2017](#dozat-2017-biaffine))
* CRFNP Dependency Parser ([Koo et al., 2007](#koo-2007-structured); [Ma and Hovy, 2017](#ma-2017-neural))
* CRF Dependency Parser ([Zhang et al., 2020a](#zhang-2020-efficient))
* CRF2o Dependency Parser ([Zhang et al, 2020a](#zhang-2020-efficient))
* CRF Constituency Parser ([Zhang et al, 2020b](#zhang-2020-fast))

You can load released pretrained models for the above parsers and obtain dependency/constituency parsing trees very conveniently, as detailed in [Usage](#Usage).

The implementations of several popular and well-known algorithms, like MST (ChuLiu/Edmods), Eisner, CKY, MatrixTree, TreeCRF, are also integrated in this package.

Besides POS Tag embeddings used by the vanilla Biaffine Parser as auxiliary inputs to the encoder, optionally, `SuPar` also allows to utilize CharLSTM/BERT layers to produce character/subword-level features.
The BERT module in `SuPar` extracts BERT representations from the pretrained model in [`transformers`](https://github.com/huggingface/transformers). 
It is also compatiable with other language models like XLNet, RoBERTa and ELECTRA, etc.

The CRF models for Dependency/Constituency parsing are our recent works published in ACL 2020 and IJCAI 2020 respectively. 
If you are interested in them, please cite:
```bib
@inproceedings{zhang-etal-2020-efficient,
  title     = {Efficient Second-Order {T}ree{CRF} for Neural Dependency Parsing},
  author    = {Zhang, Yu and Li, Zhenghua and Zhang Min},
  booktitle = {Proceedings of ACL},
  year      = {2020},
  url       = {https://www.aclweb.org/anthology/2020.acl-main.302},
  pages     = {3295--3305}
}

@inproceedings{zhang-etal-2020-fast,
  title     = {Fast and Accurate Neural {CRF} Constituency Parsing},
  author    = {Zhang, Yu and Zhou, Houquan and Li, Zhenghua},
  booktitle = {Proceedings of IJCAI},
  year      = {2020},
  doi       = {10.24963/ijcai.2020/560},
  url       = {https://doi.org/10.24963/ijcai.2020/560},
  pages     = {4046--4053}
}
```

## Contents

* Installation
* Performance
* Usage
* References

## Installation

`SuPar` can be installed via pip
```sh
pip install supar
```
Or installing from source is also permitted:
```sh
git clone https://github.com/yzhangcs/parser && cd parser
python setup.py install
```

As a prerequisite, the following requirements should be satisfied:
* `python`: 3.7
* [`pytorch`](https://github.com/pytorch/pytorch): 1.4
* [`transformers`](https://github.com/huggingface/transformers): 3.0

## Performance

`SuPar` provides several pretrained models for English and Chinese.
The English models are trained on Penn Treebank (PTB) with 39,832 training sentences, while the Chinese models are trained on Penn Chinese Treebank version 7 (CTB7) with 46,572 training sentences.

The performance and parsing speed of these models are listed in the following table.
Notably, punctuation is ignored in all evaluation metrics for PTB, but reserved for CTB7. 

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th align="center">Type</th>
      <th align="center">Name</th>
      <th align="center">Metric</th>
      <th align="center" colspan=2>Performance</th>
      <th align="right">Speed (Sents/s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan=5>PTB</td>
      <td rowspan=4>Dependency</td>
      <td><code>biaffine-dep-en</code></td>
      <td align="center">UAS/LAS</td>
      <td align="center">96.03</td><td align="center">94.37</td>
      <td align="right">1826.77</td>
    </tr>
    <tr>
      <td><code>crfnp-dep-en</code></td>
      <td align="center">UAS/LAS</td>
      <td align="center">96.01</td><td align="center">94.42</td>
      <td align="right">2197.15</td>
    </tr>
    <tr>
      <td><code>crf-dep-en</code></td>
      <td align="center">UAS/LAS</td>
      <td align="center">96.12</td><td align="center">94.50</td>
      <td align="right">652.41</td>
    </tr>
    <tr>
      <td><code>crf2o-dep-en</a></code></td>
      <td align="center">UAS/LAS</td>
      <td align="center">96.14</td><td align="center">94.55</td>
      <td align="right">465.64</td>
    </tr>
    <tr>
      <td>Constituency</td>
      <td><code>crf-con-en</a></code></td>
      <td align="center">F<sub>1</sub></td>
      <td align="center" colspan=2>94.18</td><td align="right">923.74</td>
    </tr>
  </tbody>
  <tbody>
    <tr>
      <td rowspan=5>CTB7</td>
      <td rowspan=4>Dependency</td>
      <td><code>biaffine-dep-zh</code></td>
      <td align="center">UAS/LAS</td>
      <td>88.77</td><td>85.63</td><td align="right">1155.50</td>
    </tr>
    <tr>
      <td><code>crfnp-dep-zh</code></td>
      <td align="center">UAS/LAS</td>
      <td>88.78</td><td>85.64</td><td align="right">1323.75</td>
    </tr>
    <tr>
      <td><code>crf-dep-zh</code></td>
      <td align="center">UAS/LAS</td>
      <td>88.98</td><td>85.84</td><td align="right">354.65</td>
    </tr>
    <tr>
      <td><code>crf-dep-zh</code></td>
      <td align="center">UAS/LAS</td>
      <td>89.35</td><td>86.25</td><td align="right">217.09</td>
    </tr>
    <tr>
      <td>Constituency</td>
      <td><code>crf-con-zh</code></td>
      <td align="center">F<sub>1</sub></td>
      <td align="center" colspan=2>88.67</td>
      <td align="right">639.27</td>
    </tr>
  </tbody>
</table>

## Usage

You can start the training, evaluation and prediction process by using subcommands registered in `parser.cmds`.

```sh
$ python run.py -h
usage: run.py [-h] {evaluate,predict,train} ...

Create the Biaffine Parser model.

optional arguments:
  -h, --help            show this help message and exit

Commands:
  {evaluate,predict,train}
    evaluate            Evaluate the specified parser and dataset.
    predict             Use a trained parser to make predictions.
    train               Train a parser.
```

Before triggering the subcommands, please make sure that the data files must be in CoNLL-X format. 
If some fields are missing, you can use underscores as placeholders.
Below are some examples:

```sh
$ python run.py train -d=0 -p=exp/ptb.char --feat=char  \
      --train=data/ptb/train.conllx  \
      --dev=data/ptb/dev.conllx  \
      --test=data/ptb/test.conllx  \
      --embed=data/glove.6B.100d.txt  \
      --unk=unk

$ python run.py evaluate -d=0 -f=exp/ptb.char --feat=char --tree  \
      --data=data/ptb/test.conllx

$ cat data/naive.conllx 
1       Too     _       _       _       _       _       _       _       _
2       young   _       _       _       _       _       _       _       _
3       too     _       _       _       _       _       _       _       _
4       simple  _       _       _       _       _       _       _       _
5       ,       _       _       _       _       _       _       _       _
6       sometimes       _       _       _       _       _       _       _       _
7       naive   _       _       _       _       _       _       _       _
8       .       _       _       _       _       _       _       _       _

$ python run.py predict -d=0 -p=exp/ptb.char --feat=char --tree  \
      --data=data/naive.conllx  \
      --pred=naive.conllx

# support for outputting the probabilities of predicted arcs, triggered by `--prob`
$ cat naive.conllx
1	Too	_	_	_	_	2	advmod	0.8894	_
2	young	_	_	_	_	0	root	0.9322	_
3	too	_	_	_	_	4	advmod	0.8722	_
4	simple	_	_	_	_	2	dep	0.8948	_
5	,	_	_	_	_	2	punct	0.8664	_
6	sometimes	_	_	_	_	7	advmod	0.8406	_
7	naive	_	_	_	_	2	dep	0.971	_
8	.	_	_	_	_	2	punct	0.9741	_

```

## Hyperparameters

| Param         | Description                                                  |                                 Value                                  |
| :------------ | :----------------------------------------------------------- | :--------------------------------------------------------------------: |
| n_embed       | dimension of embeddings                                      |                                  100                                   |
| n_char_embed  | dimension of char embeddings                                 |                                   50                                   |
| n_bert_layers | number of bert layers to use                                 |                                   4                                    |
| embed_dropout | dropout ratio of embeddings                                  |                                  0.33                                  |
| n_lstm_hidden | dimension of lstm hidden states                              |                                  400                                   |
| n_lstm_layers | number of lstm layers                                        |                                   3                                    |
| lstm_dropout  | dropout ratio of lstm                                        |                                  0.33                                  |
| n_mlp_arc     | arc mlp size                                                 |                                  500                                   |
| n_mlp_rel     | label mlp size                                               |                                  100                                   |
| mlp_dropout   | dropout ratio of mlp                                         |                                  0.33                                  |
| lr            | starting learning rate of training                           |                                  2e-3                                  |
| betas         | hyperparameters of momentum and L2 norm                      |                               (0.9, 0.9)                               |
| epsilon       | stability constant                                           |                                 1e-12                                  |
| annealing     | formula of learning rate annealing                           | <img src="https://latex.codecogs.com/gif.latex?.75^{\frac{t}{5000}}"/> |
| batch_size    | approximate number of tokens per training update             |                                  5000                                  |
| epochs        | max number of epochs                                         |                                 50000                                  |
| patience      | patience for early stop                                      |                                  100                                   |
| min_freq      | minimum frequency of words in the training set not discarded |                                   2                                    |
| fix_len       | fixed length of a word                                       |                                   20                                   |

## References

* <a id="dozat-2017-biaffine"></a> 
Timothy Dozat and Christopher D. Manning. 2017. [Deep Biaffine Attention for Neural Dependency Parsing](https://openreview.net/pdf?id=Hk95PK9le).
* <a id="koo-2007-structured"></a> 
Terry Koo, Amir Globerson, Xavier Carreras and Michael Collins. 2007. [Structured Prediction Models via the Matrix-Tree Theorem](https://www.aclweb.org/anthology/D07-1015/).
* <a id="ma-2017-neural"></a> 
Xuezhe Ma and Eduard Hovy. 2017. [Neural Probabilistic Model for Non-projective MST Parsing](https://www.aclweb.org/anthology/I17-1007/).
* <a id="zhang-2020-efficient"></a> 
Yu Zhang, Zhenghua Li and Min Zhang. 2020.
[Efficient Second-Order TreeCRF for Neural Dependency Parsing](https://www.aclweb.org/anthology/2020.acl-main.302/).
* <a id="zhang-2020-fast"></a> 
Yu Zhang, Houquan Zhou and Zhenghua Li. 2020.
[Fast and Accurate Neural CRF Constituency Parsing](https://www.ijcai.org/Proceedings/2020/560/).
<!-- * [Stack-Pointer Networks for Dependency Parsing](https://www.aclweb.org/anthology/P18-1130.pdf) -->
