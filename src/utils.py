from typing import *
from collections import OrderedDict
import torch
from functools import partial
from torch import nn
from pathlib import Path
import re
from datetime import datetime
import os

class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass


def listify(o):
    """convert object --> list"""
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]


def setify(o): return o if isinstance(o,set) else set(listify(o))


def uniqueify(x, sort=False):
    """ returns only unique objects out of a list or similar data structure"""
    res = list(OrderedDict.fromkeys(x).keys())
    if sort: res.sort()
    return res


class ListContainer():
    """Comparable to lists with added functionality e.g. indexing with boolean """
    def __init__(self, items): self.items = listify(items)
    def __getitem__(self, idx):
        if isinstance(idx, (int,slice)): return self.items[idx]
        if isinstance(idx, torch.Tensor) and idx.numel()==1: return self.items[idx.item()] # dev added code thanks to 12_text nb errors
        if isinstance(idx[0],bool):
            assert len(idx)==len(self) # bool mask
            return [o for m,o in zip(idx,self.items) if m]
        return [self.items[i] for i in idx]
    def __len__(self): return len(self.items)
    def __iter__(self): return iter(self.items)
    def __setitem__(self, i, o): self.items[i] = o
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self)>10: res = res[:-1]+ '...]'
        return res


def get_batch(dl, *args,**kwargs):
    return next(iter(dl))

def compose(x, funcs, *args, order_key='_order', **kwargs):
    """
    simple functionla programming function to apply a list of functions to x and retunr the result
    :param x: input
    :param funcs: func or list of funcs
    :param order_key: the order key by which to apply function
    :return:
    """
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(listify(funcs), key=key): x = f(x, **kwargs)
    return x

# Mix up
# https://github.com/fastai/course-v3/blob/master/nbs/dl2/10b_mixup_label_smoothing.ipynb
class NoneReduce():
    def __init__(self, loss_func):
        self.loss_func,self.old_red = loss_func,None

    def __enter__(self):
        if hasattr(self.loss_func, 'reduction'):
            self.old_red = getattr(self.loss_func, 'reduction')
            setattr(self.loss_func, 'reduction', 'none')
            return self.loss_func
        else: return partial(self.loss_func, reduction='none')

    def __exit__(self, type, value, traceback):
        if self.old_red is not None: setattr(self.loss_func, 'reduction', self.old_red)


def unsqueeze(input, dims):
    for dim in listify(dims): input = torch.unsqueeze(input, dim)
    return input

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

def lin_comb(v1, v2, beta): return beta*v1 + (1-beta)*v2

# https://github.com/fastai/course-v3/blob/master/nbs/dl2/11_train_imagenette.ipynb
def noop(x): return x

class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)

# second implementation (to include callbacks)
def get_batch(dl, learn):
    learn.xb,learn.yb = next(iter(dl))
    learn.do_begin_fit(0)
    learn('begin_batch')
    learn('after_fit')
    return learn.xb,learn.yb

# https://github.com/fastai/course-v3/blob/master/nbs/dl2/11a_transfer_learning.ipynb
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=1):
        super().__init__()
        self.output_size = sz
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

# transfer learning
# https://github.com/fastai/course-v3/blob/master/nbs/dl2/11a_transfer_learning.ipynb
def adapt_model(learn, data): # adapts model to new dataset
    cut = next(i for i,o in enumerate(learn.model.children())
               if isinstance(o,nn.AdaptiveAvgPool2d))
    m_cut = learn.model[:cut]
    xb,yb = get_batch(data.valid_dl, learn)
    pred = m_cut(xb)
    ni = pred.shape[1]
    m_new = nn.Sequential(
        m_cut, AdaptiveConcatPool2d(), Flatten(),
        nn.Linear(ni*2, data.c_out))
    learn.model = m_new


def set_grad(m, b):
    """
    https://github.com/fastai/course-v3/blob/master/nbs/dl2/11a_transfer_learning.ipynb
    """
    if isinstance(m, (nn.Linear,nn.BatchNorm2d)): return
    if hasattr(m, 'weight'):
        for p in m.parameters(): p.requires_grad_(b)

# Discriminative LRs
# https://github.com/fastai/course-v3/blob/master/nbs/dl2/11a_transfer_learning.ipynb


def bn_splitter(m): # batchnorm splitter
    def _bn_splitter(l, g1, g2):
        if isinstance(l, nn.BatchNorm2d): g2 += l.parameters()
        elif hasattr(l, 'weight'): g1 += l.parameters()
        for ll in l.children(): _bn_splitter(ll, g1, g2)

    g1,g2 = [],[]
    _bn_splitter(m[0], g1, g2)

    g2 += m[1:].parameters()
    return g1,g2


def albert_splitter(m, g1=[],g2=[]):
    """slits albert based on module names and types"""
    l = list(dict(m.named_children()).keys())
    if "qa_outputs" in  l: g2+= m.qa_outputs.parameters()
    if "poss" in l: g2+= m.poss.parameters()
    if isinstance(m,torch.nn.modules.normalization.LayerNorm):
        g1+= m.parameters()
    elif hasattr(m, 'weight'):
        g1+= m.parameters()
    for ll in m.children(): albert_splitter(ll, g1, g2)
    return g1,g2


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


def remove_max_sl(df, max_seq_len):
    """removes inputs which exceed maximum sequence length"""
    init_len = len(df)
    df = df[df.seq_len < max_seq_len-2]
    new_len = len(df)
    print(f"dropping {init_len - new_len} out of {init_len} questions which exceed max sequence length")
    return df


def str2tensor(s):
    """converts string based indices from csv into tensors"""
    indices = re.findall("-?\d+",s)
    return torch.tensor([int(indices[0]), int(indices[1])], dtype=torch.long)


def set_segments(x,sep_idx):
    """identifies token_type_ids for albert to determine which sequence a token belongs to."""
    res = x.new_zeros(x.size())
    for row_idx, row in enumerate(x):
        in_seg_1 = False
        for val_idx,val in enumerate(row):
            if val == sep_idx:
                in_seg_1 = True
            if in_seg_1:
                res[row_idx,val_idx] = 1
    return res


def assert_no_negs(tensor):
    """usefule for asserting labels do not contain any negative values"""
    assert torch.all(torch.eq(tensor, abs(tensor)))


def save_model_qa(learner, output_dir: Path, model_name, version):
    """
    used with SaveModelCallback and functool's partial to set model save details
    :param learner: learner object
    :param output_dir: dir where weights will be stored
    :param model_name:
    :param version:
    :return:
    """
    output_dir = Path(output_dir)
    def _create_dir(dirc):
        if not os.path.exists(dirc): os.mkdir(dirc)
    epoch = learner.epoch
    metric = round(float(learner.qa_avg_stats.valid_stats.avg_stats[1]),2)
    _create_dir(output_dir)
    model_dir = f"{re.sub(r'[ :]+','_',str(datetime.now()))}-{model_name}-acc-{metric}-ep-{epoch}-squad_{version}"
    _create_dir(output_dir/model_dir)
    learner.model.save_pretrained(output_dir/model_dir)
