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
from bs4 import BeautifulSoup

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

def sentence_chunker(book, max_seq_len):
    raw_chunks = tokenize.sent_tokenize(book)

    chunks, rejects = [],0
    len_spaces = [(len(s),Counter(s)[" "]/len(s)) for s in raw_chunks]
    lens, spaces = zip(*len_spaces)
    split_lens = [len(s.split()) for s in raw_chunks]; split_lens
    lim = np.quantile(lens,.1)

    for i,sent in enumerate(raw_chunks):
        conds = lens[i] > lim and sent[-1] != "?" and spaces[i] < .4
        if conds:
            if split_lens[i] > max_seq_len:
                for s in splitter(sent): chunks.append(s)
            else:    chunks.append(sent)
        else:
            rejects += 1

    logger.info(f"dropped {rejects} out {len(raw_chunks)} of sections")
    return chunks

def soup_chunker(input_html_file):
    soup = BeautifulSoup(open(input_html_file))
    for p in soup.find_all('p'):
        pass
    tags = ['p',re.compile('^h[1-6]$')]
    texts = [tag.text for tag in soup.find_all(tags)]

    chunks = []
    current = []
    for i,text in enumerate(texts):
        split_len = len(text.split())
        if split_len < 10 or i+1 == len(texts):
            if len(current) > 1: chunks.append(" ".join(current))
            current = []
        else:
            current.append(text)
    return chunks

def parser(input_html_file, output_dir=".", output_level = "section", max_seq_len=400):
    assert os.path.exists(output_dir), f"output_dir: {output_dir} does not exist"
    assert output_level in ["sentence","section"], "output_level must be sentence, paragrpah or section"
    output_dir = Path(output_dir)

    if output_level == "sentence":
        book = textract.process(input_html_file,encoding='unicode_escape')
        book = re.sub("\\\+[n,u]?"," ",str(book))
        chunks = sentence_chunker(book)
    elif output_level == "section":
        chunks = soup_chunker(input_html_file)

    chunks_dict = [{"id":str(i),"text":sent} for i,sent in enumerate(chunks)]

    with open(output_dir/"result.jsonl", 'w+') as f:
        for d in chunks_dict:
            json.dump(d, f)
            f.write("\n")
    logger.info(f"saved {len(chunks_dict)} chunks in result.jsonl in output_dir {output_dir}")

if __name__ == "__main__" :
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fire.Fire(parser)



# test
# parser("tbqa/albert-qa/books/Health Science 100 V3.html","./tbqa/albert-qa")
# terminal: book2json.py "books/Introductory Chemistry.html" . 300
# test
# l = splitter("a "*1000); len(l)

# from sklearn.feature_extraction.text import TfidfVectorizer
#
#
# vectorizer_tfidf = TfidfVectorizer(stop_words='english')
#
#
# vectors_tfidf = vectorizer_tfidf.fit_transform(chunks)
#
#
# vectorizer_tfidf.get_feature_names()
# df = pd.DataFrame({"id":[i for i,_ in enumerate(chunks)],"sent":chunks})
