# html (textbook) to csv or json converter
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
    """
    splits a longer string into multiple smaller strings
    :param string: string which requires splitting
    :param final: intially empty final list of reduced strings
    :return: final list
    """
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
    """
    breaks down a large document by sentences
    :param book: html file
    :param max_seq_len: max allowable sequence length
    :return: list of sentence chunks
    """
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
    """
    uses beautiful soup to divide input text based on <p> tags
    :param input_html_file: input file
    :return: list of section chunks
    """
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

def parser(input_html_file, output_dir=None, output_level = "section", max_seq_len=400, jsonl=False):
    """
    parses the input html file based on user selected options
    :param input_html_file: input file
    :param output_dir: save location for out out file. default is the same as input parent directory
    :param output_level: type of chunking
    :param max_seq_len: max allowed sequence length
    :param jsonl: bool; whether the output should be jsonl. default is csv
    :return: saves a csv file from the input files
    """
    if output_dir is None: output_dir = Path(input_html_file).parent
    assert os.path.exists(output_dir), f"output_dir: {output_dir} does not exist"
    assert output_level in ["sentence","section"], "output_level must be sentence, paragrpah or section"
    output_dir = Path(output_dir)

    if output_level == "sentence":
        book = textract.process(input_html_file,encoding='unicode_escape')
        book = re.sub("\\\+[n,u]?"," ",str(book))
        chunks = sentence_chunker(book)
    elif output_level == "section":
        chunks = soup_chunker(input_html_file)

    if json_out:
        chunks_dict = [{"id":str(i),"text":sent} for i,sent in enumerate(chunks)]
        with open(output_dir/"sections.jsonl", 'w+') as f:
            for d in chunks_dict:
                json.dump(d, f)
                f.write("\n")
        logger.info(f"saved {len(chunks_dict)} chunks in sections.jsonl in output_dir {output_dir}")
    else:
        logger.info("writting csv")
        df = pd.DataFrame({"text":chunks})
        df.index.name = 'id'
        df.to_csv(output_dir/"sections.csv")
        logger.info(f"saved {len(chunks)} chunks in sections.csv in output_dir {output_dir}")

if __name__ == "__main__" :
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fire.Fire(parser)
