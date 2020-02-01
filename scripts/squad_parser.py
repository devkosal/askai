import os, json, pandas as pd, fire, re, logging
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

def squad_parser(directory, tok , data_set: str, squad_version: str = "2.0"):
    """
    convert squad train and dev jsons to dfs. works for both 1.1 and 2.0 datasets
    """
    ver2 = squad_version == "2.0"
    ds_dir = data_set + f"-v{squad_version}.json"
    with open(Path(directory)/ds_dir) as f: file = json.load(f)
    ques, paras, answers, idxs, seq_lens = [],[],[],[],[]
    if ver2: is_impossibles = [] # if ver2, keep track of whether answer is impossible
    for item in tqdm(file["data"]):
        for paragraphs in item["paragraphs"]:
            context = paragraphs["context"]
            tok_context = tok.tokenize(context)
            for qas in paragraphs["qas"]:
                question = qas["question"]
                tok_question = tok.tokenize(question)
                is_impossible = False # flag for determining the anserability of the question. this remain the same if version is 1.1
                if ver2: is_impossible = qas["is_impossible"]
                answers_type = "plausible_answers" if is_impossible else "answers"
                for answer in qas[answers_type]:
                    start_idx = len(tok_question + tok.tokenize(context[:answer["answer_start"]]))
                    end_idx = start_idx + len(tok.tokenize(answer["text"]))
                    ques.append(question)
                    paras.append(context)
                    answers.append((tok_question + tok_context)[start_idx:end_idx])
                    idxs.append([start_idx,end_idx])
                    seq_lens.append(len(tok_context + tok_question))
                    if ver2: is_impossibles.append(is_impossible)
    res = {"question":ques,"paragraph" : paras, "answer":answers, "idxs":idxs, "seq_len":seq_lens}
    if ver2: res["is_impossible"] = is_impossibles
    return pd.DataFrame(res)


def squad_json_to_csv(path_to_jsons_dir, path_to_csv_dir=None, model="albert-base-v2",squad_version="2.0"):
    assert squad_version in ["1.1", "2.0"], f"please enter a valid squad_version: 1.1 or 2.0, not {squad_version}"
    if path_to_csv_dir is None: path_to_csv_dir = path_to_jsons_dir
    tok = AutoTokenizer.from_pretrained(model)
    model_name = re.findall(r"(.+?)-",model)[0]
    if model_name == []: raise ValueError("please enter a valid model name found in pre trained transformers library")
    assert os.path.exists(path_to_csv_dir), "output directory does not exist. please create it"
    logger.info("coverting train dataset")
    train = squad_parser(path_to_jsons_dir, tok, "train",squad_version)
    logger.info("coverting dev dataset")
    val = squad_parser(path_to_jsons_dir, tok, "dev",squad_version)

    # export dfs to csv
    train.to_csv(Path(path_to_csv_dir)/f"train_{squad_version}_{model_name}.csv",index=False)
    val.to_csv(Path(path_to_csv_dir)/f"val_{squad_version}_{model_name}.csv",index=False)
    logger.info("finished exporting csv files")

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fire.Fire(squad_json_to_csv)
