# morphological-dependency-parsing

Contains code for running the dependency parsing experiments, described in our paper **Enhancing deep neural networks with morphological information**:
```
@misc{klemen2020enhancing,
      title={Enhancing deep neural networks with morphological information}, 
      author={Matej Klemen and Luka Krsnik and Marko Robnik-Šikonja},
      year={2020},
      eprint={2011.12432},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

This code contains some modifications of the Biaffine Dependency Parser ([Dozat and Manning, 2017](#dozat-2017-biaffine)) 
from [SuPar](https://github.com/yzhangcs/parser). The original repository contains additional 
models for dependency and constituency parsing as well as a richer documentation.

The modifications made here are: 
1. **Decoupled features used as additional input**: the original implementation allowed only one of 
`{character embeddings | UPOS embeddings | BERT embeddings }` to be used at a time in addition to word embeddings.
2. **Option to use Universal Features embeddings as additional input**: Each feature gets embedded with a D-dimensional 
vector, in total creating a vector of length `23 * <ufeats_embedding_size>` (`Typo` is not used).  
3. **Minor component tweaks**: e.g. BERT is tuned together with the parser (previously frozen), BERT parameters are 
tuned with a smaller (fixed) learning rate, all BERT layers are used instead of just the last four. 

## Installation

```shell script
$ git clone https://github.com/matejklemen/morphological-dependency-parsing && cd parser
$ python setup.py install
```

## Usage
For full list of options please check `supar/cmds/{biaffine_dependency.py, cmd.py}`. 
Many of the parameters are self explanatory, so here are just some specifics:
1. `--path` is the path where best checkpoint will be saved to or loaded from.  
2. `--embed` is the path to load the pretrained word embeddings from (if not provided, they will be trained from scratch).
In our case, we use word embeddings, extracted from fastText.
3. `--include_char`, `--include_bert`, `--include_upos`, `--include_ufeats`, `--include_lstm` are flags to determine 
which features are used in addition to word embeddings.  
The character embeddings are fixed to size 50, BERT embeddings are of size corresponding to the used model's hidden size,
while POS, universal feature and LSTM embedding sizes are tunable with `--upos_emb_size` (default: 50), 
`--ufeats_emb_size` (default: 30) and `--lstm_emb_size` (default: 128).
4. `--bert` determines the used BERT (actually any [transformers](https://github.com/huggingface/transformers) 
compatible) model for BERT embeddings.
5. `--patience` is the early stopping tolerance used to stop model training after the mean of UAS and LAS does not 
improve for specified number of rounds.

```shell script
$ python3 -m supar.cmds.biaffine_dependency train \
    --path="en_model_with_bert_upos/model" \
    --tree \
    --device 0 \
    --build  \
    --batch_size=64 \
    --punct \
    --train="UD_English-EWT/en_ewt-ud-train.conllu" \
    --dev="UD_English-EWT/en_ewt-ud-dev.conllu" \
    --test="UD_English-EWT/en_ewt-ud-test.conllu" \
    --embed="" \
    --n_embed=100 \
    --include_upos \
    --upos_emb_size=50 \
    --include_bert \
    --bert="bert-base-multilingual-uncased"
```

## References

* <a id="dozat-2017-biaffine"></a> 
Timothy Dozat and Christopher D. Manning. 2017. [Deep Biaffine Attention for Neural Dependency Parsing](https://openreview.net/pdf?id=Hk95PK9le).
