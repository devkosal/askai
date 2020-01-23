from .dataloaders import *
from .utils import *
from .basics import *
from .callbacks import *

# also look up tok and num prcoesses implementations for opinions data http://localhost:8888/notebooks/devai/RoBERTa%20with%20Devai%20-%20Text%20Summarization.ipynb
from concurrent.futures import ProcessPoolExecutor

def parallel(func, arr, max_workers=4):
    if max_workers<2: results = list(progress_bar(map(func, enumerate(arr)), total=len(arr)))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(progress_bar(ex.map(func, enumerate(arr)), total=len(arr)))
    if any([o is not None for o in results]): return results

class TokenizeProcessor(Processor):
    def __init__(self, tok_func, max_sl, start_tok, end_tok, chunksize=2000, pre_rules=None, post_rules=None, max_workers=4): 
        self.chunksize,self.max_workers = chunksize,max_workers
        self.tok_func,self.max_sl = tok_func,max_sl
        self.pre_rules,self.post_rules=pre_rules,post_rules
        self.start_tok, self.end_tok = start_tok, end_tok

    def proc_one(self, x): return [self.start_tok] + self.tok_func(x)[:self.max_sl] + [self.end_tok]

    def proc_chunk(self, args):
        i,chunk = args
        chunk = [compose(t, self.pre_rules) for t in chunk]
        docs = [self.proc_one(x) for x in chunk]
        docs = [compose(t, self.post_rules) for t in docs]
        return docs

    def __call__(self, items): 
        toks = []
        if isinstance(items[0], Path): items = [reader(i) for i in items]
        chunks = [items[i: i+self.chunksize] for i in (range(0, len(items), self.chunksize))]
        toks = parallel(self.proc_chunk, chunks, max_workers=self.max_workers)
        return sum(toks, [])

    def deprocess(self, toks): return [self.deproc1(tok) for tok in toks]
    def deproc_one(self, tok):    return " ".join(tok)
    
import collections
class NumericalizeProcessor(Processor):
    """
    only works with an existing vocab at the moment and min_freq is not accounted for
    """
    def __init__(self, vocab:dict, unk_tok_idx:int, min_freq=2): 
        self.vocab, self.unk_tok_idx,self.min_freq = vocab, unk_tok_idx, min_freq
    
    def proc_one(self, x): return [self.vocab[i] if i in self.vocab else self.unk_tok_idx for i in x]
    
    def __call__(self, items): 
        if getattr(self, 'otoi', None) is None:
            self.otoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.vocab)})
        return [self.proc_one(x) for x in items]    
    
    def deprocess(self, idxs):
        assert self.vocab is not None
        return [self.deproc_one(idx) for idx in idxs]
    
    def deproc_one(self, idx): return [self.vocab[i] for i in idx]
    
# samplers
from torch.utils.data import Sampler

class SortSampler(Sampler):
    def __init__(self, data_source, key): self.data_source,self.key = data_source,key
    def __len__(self): return len(self.data_source)
    def __iter__(self):
        return iter(sorted(list(range(len(self.data_source))), key=self.key, reverse=True))
    
# added the modified version from devai for text summarization 
# original: https://github.com/fastai/course-v3/blob/master/nbs/dl2/12_text.ipynb
class SortishSampler(Sampler):
    def __init__(self, data_source, key, bs):
        self.data_source,self.key,self.bs = data_source,key,bs
        self.key = key

    def __len__(self) -> int: return len(self.data_source)

    def __iter__(self):
        idxs = torch.randperm(len(self.data_source))
        megabatches = [idxs[i:i+self.bs*50] for i in range(0, len(idxs), self.bs*50)]
        sorted_idx = torch.cat([tensor(sorted(s, key=self.key, reverse=True)) for s in megabatches])
        batches = [sorted_idx[i:i+self.bs] for i in range(0, len(sorted_idx), self.bs)]
        
        #dev
        sorting_vals = self.key(batches[0][0])
        if not hasattr(sorting_vals, '__iter__'):
            max_idx = torch.argmax(tensor([self.key(ck[0]) for ck in batches]))  # find the chunk with the largest key,
        elif len(sorting_vals) == 2:
        # since we dealing with two keys
            sorted_batches = sorted([self.key(ck[0]) for ck in batches], key=lambda t: (t[0],t[1]), reverse=True)
            m = sorted_batches[0] # max key
            max_idx = sorted_batches.index(m) # find index of max key
        else:
            raise ValueError(f"number of sorting key values must be eihter 1 or 2 instead of {len(sorting_vals)}")
        #/dev
        
        batches[0],batches[max_idx] = batches[max_idx],batches[0]            # then make sure it goes first.
        batch_idxs = torch.randperm(len(batches)-2)
        sorted_idx = torch.cat([batches[i+1] for i in batch_idxs]) if len(batches) > 1 else LongTensor([])
        sorted_idx = torch.cat([batches[0], sorted_idx, batches[-1]])
        return iter(sorted_idx)
      
def pad_collate(samples, pad_idx=1, pad_first=False):
    max_len = max([len(s[0]) for s in samples])
    res = torch.zeros(len(samples), max_len).long() + pad_idx
    for i,s in enumerate(samples):
        if pad_first: res[i, -len(s[0]):] = torch.LongTensor(s[0])
        else:         res[i, :len(s[0]) ] = torch.LongTensor(s[0])
    return res, tensor([s[1] for s in samples])    

# for sea2seq transformers (text summ, translation) 
# http://localhost:8888/notebooks/course-nlp/8-translation-transformer.ipynb 
def tfm_collate(samples, pad_idx=1, pad_first=False, backwards=False):
    "Function that collect samples and adds padding. Flips token order if needed"
#     samples = to_data(samples)
    max_len_x,max_len_y = max([len(s[0]) for s in samples]),max([len(s[1]) for s in samples])
    res_x = torch.zeros(len(samples), max_len_x).long() + pad_idx
    res_y = torch.zeros(len(samples), max_len_y).long() + pad_idx
    if backwards: pad_first = not pad_first
    for i,s in enumerate(samples):
        if pad_first: 
            res_x[i,-len(s[0]):],res_y[i,-len(s[1]):] = torch.LongTensor(s[0]),torch.LongTensor(s[1])
        else:         
            res_x[i, :len(s[0])],res_y[i, :len(s[1])] = torch.LongTensor(s[0]),torch.LongTensor(s[1])
    if backwards: res_x,res_y = res_x.flip(1),res_y.flip(1)
    return res_x, 



# Transformer Implementation from Fastai (missing resblocks)
# https://github.com/fastai/course-v3/blob/master/nbs/dl2/translation_transformer.ipynb
from fastai.text import compose as compose_fa,ifnone,feed_forward
class PositionalEncoding(nn.Module):
    "Encode the position with a sinusoid."
    def __init__(self, d):
        super().__init__()
        self.register_buffer('freq', 1 / (10000 ** (torch.arange(0., d, 2.)/d)))
    
    def forward(self, pos):
        inp = torch.ger(pos, self.freq)
        enc = torch.cat([inp.sin(), inp.cos()], dim=-1)
        return enc

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_head=None, p=0., bias=True, scale=True):
        super().__init__()
        d_head = ifnone(d_head, d_model//n_heads)
        self.n_heads,self.d_head,self.scale = n_heads,d_head,scale
        self.q_wgt,self.k_wgt,self.v_wgt = [nn.Linear(
            d_model, n_heads * d_head, bias=bias) for o in range(3)]
        self.out = nn.Linear(n_heads * d_head, d_model, bias=bias)
        self.drop_att,self.drop_res = nn.Dropout(p),nn.Dropout(p)
        self.ln = nn.LayerNorm(d_model)
        
    def forward(self, q, kv, mask=None):
        return self.ln(q + self.drop_res(self.out(self._apply_attention(q, kv, mask=mask))))
    
    def create_attn_mat(self, x, layer, bs):
        return layer(x).view(bs, x.size(1), self.n_heads, self.d_head
                            ).permute(0, 2, 1, 3)
    
    def _apply_attention(self, q, kv, mask=None):
        bs,seq_len = q.size(0),q.size(1)
        wq,wk,wv = map(lambda o: self.create_attn_mat(*o,bs),
                       zip((q,kv,kv),(self.q_wgt,self.k_wgt,self.v_wgt)))
        attn_score = wq @ wk.transpose(2,3)
        if self.scale: attn_score /= math.sqrt(self.d_head)
        if mask is not None: 
            attn_score = attn_score.float().masked_fill(mask, -float('inf')).type_as(attn_score)
        attn_prob = self.drop_att(F.softmax(attn_score, dim=-1))
        attn_vec = attn_prob @ wv
        return attn_vec.permute(0, 2, 1, 3).contiguous().view(bs, seq_len, -1)    
    
def get_output_mask(inp, pad_idx=1):
    return torch.triu(inp.new_ones(inp.size(1),inp.size(1)), diagonal=1)[None,None].bool()   

class EncoderBlock(nn.Module):
    "Encoder block of a Transformer model."
    #Can't use Sequential directly cause more than one input...
    def __init__(self, n_heads, d_model, d_head, d_inner, p=0., bias=True, scale=True, double_drop=True):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.ff  = feed_forward(d_model, d_inner, ff_p=p, double_drop=double_drop)
    
    def forward(self, x, mask=None): return self.ff(self.mha(x, x, mask=mask))

    
class DecoderBlock(nn.Module):
    "Decoder block of a Transformer model."
    #Can't use Sequential directly cause more than one input...
    def __init__(self, n_heads, d_model, d_head, d_inner, p=0., bias=True, scale=True, double_drop=True):
        super().__init__()
        self.mha1 = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.mha2 = MultiHeadAttention(n_heads, d_model, d_head, p=p, bias=bias, scale=scale)
        self.ff   = feed_forward(d_model, d_inner, ff_p=p, double_drop=double_drop)
    
    def forward(self, x, enc, mask_out=None): return self.ff(self.mha2(self.mha1(x, x, mask_out), enc))
    
    
    
# Shift transform callback implementation in order properly hide true y from model
class TFMRBatchTransformCallback(Callback): 
    _order=2
    def __init__(self, tfm): self.tfm = tfm
    def begin_batch(self): self.run.yb_padded = self.tfm(self.yb) 

def shift_tfm(y):
    y = F.pad(y, (1, 0), value=config.pad_idx)
    return y[:,:-1].contiguous()
    
# Text Summarization Learner (since we need to pass in an offset y)
class TextSumLearner(Learner):
    def one_batch(self, i, xb, yb):
        try:
            self.iter = i
            self.xb,self.yb = xb,yb;                        self('begin_batch')
            # we use yb_padded here from the transformation callback
            self.pred = self.model(self.xb,self.yb_padded); self('after_pred')
            self.loss = self.loss_func(self.pred, self.yb); self('after_loss')
            if not self.in_train: return # if not in train, stop function
            self.loss.backward();                           self('after_backward')
            self.opt.step();                                self('after_step')
            self.opt.zero_grad()
        except CancelBatchException:                        self('after_cancel_batch')
        finally:                                            self('after_batch')
    
    
# getting predictions from a batch of logits
# variation of get preds function from my text gen article. Used for predicting on new data
from random import choice
def get_preds(logits_all, p=0.3):
    def _get_pred(logits, p=p):
        probs = F.softmax(logits, dim=-1)
        idxs = torch.argsort(probs, descending=True)
        res, cumsum = [], 0.
        for idx in idxs:
            res.append(idx)
            cumsum += probs[idx]
            if cumsum > p:
                pred_idx = idxs.new_tensor([choice(res)])
                return pred_idx.unsqueeze(-1) # unsqueezing so it can be concatenated together later on
    return torch.cat([_get_pred(i) for i in logits_all])