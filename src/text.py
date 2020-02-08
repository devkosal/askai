from .dataloaders import *
from .utils import *
from .basics import *
from .callbacks import *
from tqdm import tqdm
import collections
from concurrent.futures import ProcessPoolExecutor


def parallel(func, arr, max_workers=4):
    if max_workers<2: results = list(progress_bar(map(func, enumerate(arr)), total=len(arr)))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(progress_bar(ex.map(func, enumerate(arr)), total=len(arr)))
    if any([o is not None for o in results]): return results


class QATokenizerProcessor(Processor):
    """tokenizer processor for input text"""
    def __init__(self, tok_func, max_sl, start_tok, end_tok, pre_rules=None,post_rules=None):
        self.tok_func,self.max_sl = tok_func,max_sl
        self.pre_rules,self.post_rules=pre_rules,post_rules
        self.start_tok, self.end_tok = start_tok, end_tok

    def proc1(self, x): return [self.start_tok] + self.tok_func(x)[:self.max_sl-2] + [self.end_tok]

    def __call__(self, items): return tqdm([self.proc1(x) for x in items])


class QANumericalizeProcessor(Processor):
    """
    tokens to numeric ids processor
    only works with an existing vocab at the moment and min_freq is not accounted for
    """
    def __init__(self, vocab:dict, unk_tok_idx:int, min_freq=2):
        self.vocab, self.unk_tok_idx, self.min_freq = vocab, unk_tok_idx, min_freq

    def proc1(self, x): return [self.vocab[i] if i in self.vocab else self.unk_tok_idx for i in x]

    def __call__(self, items):
        if getattr(self, 'otoi', None) is None:
            self.otoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.vocab)})
        return tqdm([self.proc1(x) for x in items])


class QALabelProcessor(Processor):
    """creates index and is_impossible labels for QA MTL task"""
    def __init__(self, parse_func = noop, adjustment = 2):
        self.parse_func = parse_func
        self.adjustment = adjustment
        self.vocab=[False, True]
        self.otoi=None

    def cat_proc1(self,item): return self.otoi[item]
    def index_proc1(self, item): return self.parse_func(item) + self.adjustment

    def __call__(self, items):
        if self.otoi is None:
            self.otoi  = {v:k for k,v in enumerate(self.vocab)}
        return [(self.index_proc1(item[0]),self.cat_proc1(item[1])) for item in items]


# samplers
from torch.utils.data import Sampler

class SortSampler(Sampler):
    """samples observations sorted by length to ensure batches contain similarly sized sequences"""
    def __init__(self, data_source, key): self.data_source,self.key = data_source,key

    def __len__(self): return len(self.data_source)

    def __iter__(self):
        return iter(sorted(list(range(len(self.data_source))), key=self.key, reverse=True))


class SortishSampler(Sampler):
    """
    samples observations fuzzily sorted by length to ensure batches contain similarly sized sequence
    useful for training dataloader to avoid modeling bias for a purely sorted dataloader
    """
    def __init__(self, data_source, key, bs):
        self.data_source,self.key,self.bs = data_source,key,bs

    def __len__(self) -> int: return len(self.data_source)

    def __iter__(self):
        idxs = torch.randperm(len(self.data_source))
        megabatches = [idxs[i:i+self.bs*50] for i in range(0, len(idxs), self.bs*50)]
        sorted_idx = torch.cat([tensor(sorted(s, key=self.key, reverse=True)) for s in megabatches])
        batches = [sorted_idx[i:i+self.bs] for i in range(0, len(sorted_idx), self.bs)]
        max_idx = torch.argmax(tensor([self.key(ck[0]) for ck in batches]))  # find the chunk with the largest key,
        batches[0],batches[max_idx] = batches[max_idx],batches[0]            # then make sure it goes first.
        batch_idxs = torch.randperm(len(batches)-2)
        sorted_idx = torch.cat([batches[i+1] for i in batch_idxs]) if len(batches) > 1 else LongTensor([])
        sorted_idx = torch.cat([batches[0], sorted_idx, batches[-1]])
        return iter(sorted_idx)


def pad_collate(samples, pad_idx=1, pad_first=False):
    """pads and collates input and labels into a single tensor"""
    max_len = max([len(s[0]) for s in samples])
    res = torch.zeros(len(samples), max_len).long() + pad_idx
    for i,s in enumerate(samples):
        if pad_first: res[i, -len(s[0]):] = torch.LongTensor(s[0])
        else:         res[i, :len(s[0]) ] = torch.LongTensor(s[0])
    return res, tensor([s[1] for s in samples])


def pad_collate_qa(samples, pad_idx, pad_first=False):
    """pads and collates inputs and labels for QA into a single tensor"""
    max_len = max([len(s[0]) for s in samples])
    res = torch.zeros(len(samples), max_len).long() + pad_idx
    for i,s in enumerate(samples):
        if pad_first: res[i, -len(s[0]):] = torch.LongTensor(s[0])
        else:         res[i, :len(s[0]) ] = torch.LongTensor(s[0])
    qa_idxs = torch.cat([s[1][0].unsqueeze(0) for s in samples])
    imp_labels = torch.tensor([s[1][1] for s in samples])
    return res, (qa_idxs, imp_labels)


def pad_collate_x(samples, pad_idx, pad_first=False):
    """pads and collates only inputs into a single tensor. useful for inference when labels don't exist"""
    max_len = max([len(s[0]) for s in samples])
    res = torch.zeros(len(samples), max_len).long() + pad_idx
    for i,s in enumerate(samples):
        if pad_first: res[i, -len(s[0]):] = torch.LongTensor(s[0])
        else:         res[i, :len(s[0]) ] = torch.LongTensor(s[0])
    return res
