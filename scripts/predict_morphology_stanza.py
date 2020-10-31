from conllu import parse
import stanza
import os
from copy import deepcopy
from collections import OrderedDict
from tqdm import tqdm


""" A script that reads a file in conllu format, predicts morphological features and writes them to a new file. 
    Before running this, make sure that you have the correct models downloaded:
    `stanza.download(<lang>, processors="tokenize,pos", package=<treebank>)`
"""
if __name__ == "__main__":
    # Note: make sure this is at least 2 levels deep (e.g. ENG/file.txt) so that the new folder name can be inferred (e.g. ENG-noisy)
    src_path = "../data/RUS/ru_gsd-ud-test.conllu.txt"
    # stanza.download("fi", processors="tokenize,pos", package="ftb")
    nlp = stanza.Pipeline(lang='ru', processors='tokenize,pos', package="syntagrus", tokenize_pretokenized=True)

    file_parts = src_path.split(os.sep)
    file_dir = os.sep.join(file_parts[:-1])
    file_name = file_parts[-1]

    target_dir = f"{file_dir}-noisy-stanza"
    os.makedirs(target_dir, exist_ok=True)

    with open(src_path) as f:
        data = parse("".join(f.readlines()))

    num_correct_upos = 0
    num_correct_ufeats = 0
    total_tokens = 0

    # Copy over dependency labels from golden file
    data_copy = deepcopy(data)
    for idx_sent, gold_sent in tqdm(enumerate(data)):
        predicted_sent = nlp([[t["form"] for t in gold_sent]]).sentences[0]
        
        for idx_token, (gold_tok, predicted_tok_obj) in enumerate(zip(gold_sent, predicted_sent.tokens)):
            predicted_tok = predicted_tok_obj.words[0]
            assert gold_tok['form'] == predicted_tok.text
            data_copy[idx_sent][idx_token]["upos"] = predicted_tok.upos
            data_copy[idx_sent][idx_token]["feats"] = predicted_tok.feats

            predicted_feats = predicted_tok.feats
            if predicted_tok.feats is not None:
                predicted_feats = OrderedDict(map(lambda s: s.split("="), predicted_feats.split("|")))

            if gold_tok["feats"] is None and predicted_feats is None:
                num_correct_ufeats += 1
            elif gold_tok["feats"] is not None and predicted_feats is not None:
                gold_ufeats = gold_tok["feats"]
                pred_ufeats = predicted_feats

                common_ufeats = set(gold_ufeats.keys()) & set(pred_ufeats.keys())
                has_same_ufeats = len(common_ufeats) == len(pred_ufeats.keys())

                num_correct_ufeats += int(has_same_ufeats and all([gold_ufeats[feat] == pred_ufeats[feat] for feat in common_ufeats]))

            num_correct_upos += int(gold_tok["upos"] == predicted_tok.upos)
            total_tokens += 1

    print(f"{num_correct_upos}/{total_tokens} correct UPOS tags ({100.0 * num_correct_upos / total_tokens: .2f}%)")
    print(f"{num_correct_ufeats}/{total_tokens} correct UFeats ({100.0 * num_correct_ufeats / total_tokens: .2f}%)")
    
    res = [sent.serialize() for sent in data_copy]
    with open(os.path.join(target_dir, file_name), "w") as f_tgt:
        print(f"Saving new data to '{os.path.join(target_dir, file_name)}'")
        f_tgt.writelines(res)
