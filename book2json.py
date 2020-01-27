# textbook to json converter for drqa db and tfidf creator
import os
import textract
from nltk import tokenize
import re
import pandas as pd
import numpy as np
from collections import Counter
import json
import logging
import fire
from pathlib import Path


def splitter(string,final=[]):
    split_string = string.split()
    split_len = len(split_string)
    if split_len < 300:
        final.append(string)
    else:
        split_idx = min(round(len(split_string)/2),300)
        join1,join2 = " ".join(split_string[:split_idx])," ".join(split_string[split_idx:])
        final.append(join1)
        splitter(join2)
    return final


def parser(input_html_file, output_dir=".", max_seq_len=300):
    assert os.path.exists(output_dir), f"output_dir: {output_dir} does not exist"
    output_dir = Path(output_dir)

    book = textract.process(input_html_file,encoding='unicode_escape')
    book = re.sub("\\\+[n,u]?"," ",str(book))

    raw_sents = tokenize.sent_tokenize(book)

    sents, rejects = [],0
    len_spaces = [(len(s),Counter(s)[" "]/len(s)) for s in raw_sents]
    lens, spaces = zip(*len_spaces)
    split_lens = [len(s.split()) for s in raw_sents]; split_lens
    lim = np.quantile(lens,.1)

    for i,sent in enumerate(raw_sents):
        conds = lens[i] > lim and sent[-1] != "?" and spaces[i] < .4
        if conds:
            if split_lens[i] > 300:
                for s in splitter(sent): sents.append(s)
            else:    sents.append(sent)
        else:
            rejects += 1

    logger.info(f"dropped {rejects} out {len(raw_sents)} of sections")

    sents_dict = [{"id":str(i),"text":sent} for i,sent in enumerate(sents)]

    with open(output_dir/'result.jsonl', 'w+') as f:
        for d in sents_dict:
            json.dump(d, f)
            f.write("\n")
    logger.info(f"saved {len(sents_dict)} sentences in result.jsonl in output_dir {output_dir}")

if __name__ == "__main__" :
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fire.Fire(parser)

# test
# parser("books/Introductory Chemistry.html",".",300)
# terminal: book2json.py "books/Introductory Chemistry.html" . 300
# test
# l = splitter("a "*1000); len(l)

# from sklearn.feature_extraction.text import TfidfVectorizer
#
#
# vectorizer_tfidf = TfidfVectorizer(stop_words='english')
#
#
# vectors_tfidf = vectorizer_tfidf.fit_transform(sents)
#
#
# vectorizer_tfidf.get_feature_names()
# df = pd.DataFrame({"id":[i for i,_ in enumerate(sents)],"sent":sents})
