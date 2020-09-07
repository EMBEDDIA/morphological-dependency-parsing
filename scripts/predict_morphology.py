from conllu import parse
from ufal.udpipe import Model, Pipeline, ProcessingError
import os


""" A script that reads a file in conllu format, predicts morphological features and writes them to a new file. """
if __name__ == "__main__":
    # Note: make sure this is at least 2 levels deep (e.g. ENG/file.txt) so that the new folder name can be inferred (e.g. ENG-noisy)
    src_path = "../data/ENG/en_ewt-ud-test.conllu.txt"
    # Path to UDPipe model (https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3131)
    model_path = "udpipe_models/english-gum-ud-2.5-191206.udpipe"

    file_parts = src_path.split(os.sep)
    file_dir = os.sep.join(file_parts[:-1])
    file_name = file_parts[-1]

    target_dir = f"{file_dir}-noisy"
    os.makedirs(target_dir, exist_ok=True)

    with open(src_path) as f:
        data = parse("".join(f.readlines()))

    # Clear the associated features which might already be set (e.g. by human annotators)
    sentence_tokens = []
    for curr_sent in data:
        curr_sent_fmt = []
        for i, curr_token in enumerate(curr_sent, start=1):
            curr_sent_fmt.append(f"{i}\t{curr_token['form']}\t_\t_\t_\t_\t_\t_\t_\t_")
        sentence_tokens.append("\n".join(curr_sent_fmt))

    print(f"Source: '{src_path}'")
    print(f"Read {len(sentence_tokens)} sentences.")
    sentence_tokens = "\n\n".join(sentence_tokens)  # type: str

    err = ProcessingError()
    model = Model.load(model_path)
    if not model:
        print(f"Could not load model from '{model_path}'")
        exit(1)

    pipeline = Pipeline(model, "conllu", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
    res = pipeline.process(sentence_tokens)  # returns CoNLLu format
    print(f"Predicted morphological features for {len(parse(res))} sentences.")

    with open(os.path.join(target_dir, file_name), "w") as f_tgt:
        f_tgt.writelines(res)
