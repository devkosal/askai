import torch
from itertools import compress
import re

def listify(o):
    """convert object --> list"""
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]

def pad_collate_x(samples, pad_idx, pad_first=False):
    """pads and collates only inputs into a single tensor. useful for inference when labels don't exist"""
    max_len = max([len(s[0]) for s in samples])
    res = torch.zeros(len(samples), max_len).long() + pad_idx
    for i,s in enumerate(samples):
        if pad_first: res[i, -len(s[0]):] = torch.LongTensor(s[0])
        else:         res[i, :len(s[0]) ] = torch.LongTensor(s[0])
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
    sorted_sums = torch.argsort(torch.tensor([sp+ep for (sp,ep) in zip(start_probs,end_probs)]),descending=True)#[::-1]
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
