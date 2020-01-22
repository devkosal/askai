
import os, json, pandas as pd, fire, re, logging
from pathlib import Path
from transformers import AutoTokenizer

def squad_parser(directory, tok , data_set: str, squad_version: str = "1.1"):
    """
    convert squad train and dev jsons to dfs
    """
    ds_dir = data_set + f"-v{squad_version}.json"
    with open(Path(directory)/ds_dir) as f: file = json.load(f)
    ques, paras, answers, idxs, seq_len = [],[],[],[],[]
    for item in file["data"]:
        for paragraphs in item["paragraphs"]:
            context = paragraphs["context"]
            tok_context = tok.tokenize(context)
            for qas in paragraphs["qas"]:
                for answer in qas["answers"]:
                    start_idx = len(tok.tokenize(context[:answer["answer_start"]]))
                    end_idx = start_idx + len(tok.tokenize(answer["text"]))
                    ques.append(qas["question"])
                    paras.append(context)
                    answers.append(tok_context[start_idx:end_idx])
                    idxs.append([start_idx,end_idx])
                    seq_len.append(len(tok_context + tok.tokenize(qas["question"])))
    return pd.DataFrame({"question":ques,"paragraph" : paras, "answer":answers, "idxs":idxs, "seq_len":seq_len})


def squad_json_to_csv(path_to_jsons_dir, path_to_csv_dir, model="albert-base-v2",squad_version="1.1"):
    tok = AutoTokenizer.from_pretrained(model)
    model_name = re.findall(r"(.+?)-",model)[0]
    if model_name == []: raise ValueError("please enter a valid model name found in pre trained transformers library")
    assert os.path.exists(path_to_csv_dir), "output directory does not exist. please create it"
    logger.info("coverting train dataset")
    train = squad_parser(path_to_jsons_dir, tok, "train","1.1")
    logger.info("coverting dev dataset")
    val = squad_parser(path_to_jsons_dir, tok, "dev","1.1")

    # export dfs to csv
    train.to_csv(Path(path_to_csv_dir)/f"train_{model_name}.csv",index=False)
    val.to_csv(Path(path_to_csv_dir)/f"val_{model_name}.csv",index=False)
    logger.info("finished expporting csv files")

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    fire.Fire(squad_json_to_csv)
