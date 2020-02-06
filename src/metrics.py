import torch
import numpy as np
import collections
def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb).float().mean()

# defining the evaluation metrics based on squad evaluation methods
def acc_qa(input,target,xb):
    """
    Taking the average between the accuracies of predicting the start and ending indices
    """
    return (accuracy(input[0], target[0][:,0]) + accuracy(input[1], target[0][:,1]))/2.0

def acc_pos(input,target,xb):
    """is_impossible accuracy metric for QA MTL"""
    return accuracy(input[2], target[1])

def exact_match(input,target,xb):
    """scores 1 if the predicted answer is exactly the same as actual else it scores 0 """
    def _acc(out, yb): return (torch.argmax(out, dim=1)==yb).float()
    return (_acc(input[0], target[0][:,0]) + _acc(input[1], target[0][:,1]) == 2).float().mean()

def f1_score(input,target,xb):
    """
    F1 score: looks at how many words were correctly predicted and how many words from the actual answer were captured
    Based on the official evaluation script:
    https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """
    inp = [t.clone().detach() for t in input]
    targ = [t.clone().detach() for t in target]
    pred_starts,pred_ends = [torch.argmax(out, dim=1) for out in input[:2]]
    gold_starts,gold_ends = targ[0][:,0], targ[0][:,1]

    def _get_toks(idx,start,end):
        """
        returns the answer tokens from xb based on start and ending indices
        :param idx: index of the item within the minibatch
        :param start: starting index
        :param end: ending index
        :return: list of answer tokens
        """
        if start == end: end += 1
        return xb.clone().detach()[idx][start:end]

    def _score1(pred_toks,gold_toks):
        common = collections.Counter(gold_toks.tolist()) & collections.Counter(pred_toks.tolist()) # finds the common tokens between two pred and gold (actual) tokens
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    pred_toks = [_get_toks(i,start,end) for i,(start,end) in enumerate(zip(pred_starts,pred_ends))]
    gold_toks = [_get_toks(i,start,end) for i,(start,end) in enumerate(zip(gold_starts,gold_ends))]
    score = np.mean([_score1(pred,gold) for pred,gold in zip(pred_toks,gold_toks)])
    return score
