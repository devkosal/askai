# utility functions for the web application

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from .text import pad_collate_x
from .utils import listify
from itertools import compress
import sqlite3
import re


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


def prep_text(text, question, tok):
    """
    preprocesses text to be inputed into a bert model
    :param text: raw input text
    :param question: raw input question
    :param tok: tokenizer function
    :return: numericalized list of tokens
    """
    tok_text, tok_ques = tok.tokenize(text), tok.tokenize(question)
    truncate_len = 512 - len(tok_ques) - 3*3
    res = ["[CLS]"] + tok_text[:truncate_len] + ["[SEP]"] + tok_ques + ["[SEP]"]
    return torch.tensor(tok.convert_tokens_to_ids(res)).unsqueeze(0)


def get_pred(texts, question, model, tok, pad_idx):
    """
    outputs the best prediction and most relevant text of a single question based on one or more relevant texts
    :param texts: list of relevant texts
    :param question: single question string
    :param model: model used for predictions
    :param tok: tokenizer functions
    :param pad_idx: index of where the pad is placed (found in the model's vocabulary file)
    :return: answer, most relevant section
    """
    bad_match_res = ("could not find a section which matched query","N/A")
    if texts == []: return bad_match_res
    texts = listify(texts)
    # 1. tokenize/encode the input text
    input_ids = pad_collate_x([prep_text(t, question, tok) for t in texts],pad_idx)
    # 2. extract the logits vector for the next possible token
    if torch.cuda.is_available(): input_ids = input_ids.cuda()
    outputs = model(input_ids)
    logits,imp_logits = outputs[:2],outputs[2]
    answerable = ~torch.argmax(imp_logits,dim=1).bool()
    if torch.all(~answerable): return bad_match_res
    texts = list(compress(texts, answerable))
    input_ids = input_ids[answerable]
    # 3. apply argmax to the logits so we have the probabilities of each index
    (start_probs,starts),(end_probs,ends) = [torch.max(out, dim=1) for out in logits]
    start_probs = start_probs.masked_select(answerable)
    starts = starts.masked_select(answerable)
    end_probs = end_probs.masked_select(answerable)
    ends = ends.masked_select(answerable)

    # 4. sort the sums of the starts and ends to determine which answers are the most ideal
    sorted_sums = np.argsort([sp+ep for (sp,ep) in zip(start_probs,end_probs)])[::-1]
    assert len(texts) == len(sorted_sums) == len(start_probs)
    def _proc1(idx,start,end):
        if start > end: return
        elif start == end: end += 1
        pred = tok.convert_ids_to_tokens(input_ids[idx][start:end])
        pred = tok.convert_tokens_to_string(pred)
        return pred.replace("<unk>","")

    # 5. find the best answer
    for i,s in enumerate(sorted_sums):
        ans = _proc1(s,starts[s],ends[s])
        if ans is not None and "<pad>" not in ans and "[SEP]" not in ans:
            section = re.sub("\s+"," ",texts[s])
            section = section.replace("’","")
            return ans, section
    return "Sorry! An answer could not be found but maybe this will help:",texts[s]
