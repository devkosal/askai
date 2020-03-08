# utility functions for the web application

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import sqlite3
import re

class Config(dict):
    """config object to store task specific information"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

    def save_to_json(self, output_file):
        json.dump(self.__dict__,open("output_file","w"),indent = 4, sort_keys=True)

# doc retrieval function
def get_doc_by_id(doc_id, cursor):
    """
    returns sqlite db at doc_id
    limited to dbs with a "documents" table with columns 'id' and 'text'
    :param doc_id: desired doc_id
    :param cursor: the sqlite db cursor
    :return: returns the document at id, doc_id
    """
    return cursor.execute(f"select * from documents where id='{doc_id}'").fetchall()


def get_scores(text, vectorizer, X):
    """
    scores relevant sections based on cosine similarity
    :param text: text to compare to all sections
    :param vectorizer: vectorizer object to use to convert text into embedding
    :param X: embeddings of sections
    :return: list of all relevant sections sorted by highest score
    """
    y = vectorizer.transform([text])
    comp = cosine_similarity(X, y, dense_output=False)
    rows, _ = comp.nonzero()
    d = {i:float(comp[i,].todense()) for i in rows}
    return sorted(d.items(), key=lambda x: x[1], reverse=True)


def bold_answer(text, answer):
    """
    finds the answer within text and adds '**' to start and end spans of the answer within text (for bolding it within Markdown)
    :param text: section text in which answer is found
    :param answer: answer text
    :return: text with bolded answer
    """
    p1 = re.compile(f"{answer}",re.IGNORECASE)
    answers = re.findall(p1, text)
    if len(answers) < 1: return text
    answer = answers[0] # selecting the first occurence
    p2 = re.compile(f"(.?){answer}(.?)",re.IGNORECASE)
    return p2.sub(f'\\1**{answer}**\\2', text)


def get_contexts(scored_sections,cursor_or_df,k=5,p=.7):
    """

    :param scored_sections: section sorted by scores (descending)
    :param cursor_or_df: sqlite db or a pandas dataframe
    :param k: the max desired outputs to use
    :param p: the cumulative probability before stopping sections search
    :return: list of top sections
    """
    top_docs = scored_sections[:k]
    top_scores = [i[1] for i in top_docs]
    norm_scores = np.array(top_scores)/sum(top_scores)
    top_ids, total = [],0
    for i,(idx,_) in enumerate(top_docs):
        if total > p: break
        top_ids.append(idx)
        total += norm_scores[i]
    res = [get_doc_by_id(i,cursor_or_df)[0][1] for i in top_ids] if isinstance(cursor_or_df,sqlite3.Cursor) else [cursor_or_df.text.loc[i] for i in top_ids]
    return res
