import torch
def accuracy(out, yb): return (torch.argmax(out, dim=1)==yb).float().mean()

# https://github.com/fastai/course-v3/blob/master/nbs/dl2/12a_awd_lstm.ipynb
def accuracy_flat(input, target):
    bs,sl = target.size()
    return accuracy(input.view(bs * sl, -1), target.view(bs * sl))   


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

cc = SmoothingFunction() # not sure what exactly this does
def bleu(input, target):
    inp = torch.argmax(F.softmax(input,dim=-1),dim=-1)
    def _proc_one(cand, ref): # process one
        ref = [ref.tolist()]
        cand = cand.tolist()
        return sentence_bleu(ref, cand, smoothing_function=cc.method1)
    return torch.tensor([_proc_one(inp[i],target[i]) for i in range(len(input))]).float().mean()