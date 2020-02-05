import re
from functools import partial
from torch import tensor
import math
import matplotlib.pyplot as plt
import torch
from .utils import *
from torch.distributions.beta import Beta
import time
from fastprogress.fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import format_time
# import apex.fp16_utils as fp16
from pdb import set_trace
import logging

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

class Callback():
    _order=0
    def set_runner(self, run): self.run=run
    def __getattr__(self, k): return getattr(self.run, k)

    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True
        return False

class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.run.n_epochs=0.
        self.run.n_iter=0

    def after_batch(self):
        if not self.in_train: return
        self.run.n_epochs += 1./self.iters # self.iters = len(dl) ie total # of batches. This allows us to keep track of the progress overall as the n.epochs increaes.
        self.run.n_iter   += 1

    def begin_epoch(self):
        self.run.n_epochs=self.epoch # at the begining of an epoch, the the epoch is reset to the value of current epoch (althought the n.epochs is updated above, this just prevents any floating point problems (assumption)
        self.model.train()
        self.run.in_train=True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train=False


class TestCallback(Callback):
    def after_step(self):
        print(self.n_iter)
        if self.n_iter>=10: raise CancelTrainException()


class CudaCallback(Callback):
    def begin_fit(self): self.model.cuda()
    def begin_batch(self): self.run.xb, self.run.yb = self.xb.cuda(), self.yb.cuda()

class AvgStats():
    def __init__(self, metrics, in_train): self.metrics,self.in_train = listify(metrics),in_train

    def reset(self):
        self.tot_loss,self.count = 0.,0
        self.tot_mets = [0.] * len(self.metrics)

    @property
    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets
    @property
    def avg_stats(self):
        return [o/self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count: return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i,m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn

class AvgStatsCallback(Callback):
    _order=1
    def __init__(self, metrics):
        self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)

    def begin_fit(self):
        met_names = ['loss'] + [m.__name__ for m in self.train_stats.metrics]
        names = ['epoch'] + [f'train_{n}' for n in met_names] + [
            f'valid_{n}' for n in met_names] + ['time']
        self.logger(names)

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)

    def after_epoch(self):
        stats = [str(self.epoch)]
        for o in [self.train_stats, self.valid_stats]:
            stats += [f'{v:.6f}' for v in o.avg_stats]
        stats += [format_time(time.time() - self.start_time)]
        self.logger(stats)

# For progress bar: https://github.com/fastai/course-v3/blob/master/nbs/dl2/09c_add_progress_bar.ipynb
class ProgressCallback(Callback):
    _order=-1
    def begin_fit(self):
        self.mbar = master_bar(range(self.epochs))
        self.mbar.on_iter_begin()
        self.run.logger = partial(self.mbar.write, table=True)

    def after_fit(self): self.mbar.on_iter_end()
    def after_batch(self): self.pb.update(self.iter)
    def begin_epoch   (self): self.set_pb()
    def begin_validate(self): self.set_pb()

    def set_pb(self):
        self.pb = progress_bar(self.dl, parent=self.mbar)#, auto_update=False) # commented out 01.22.2020
        self.mbar.update(self.epoch)


# https://github.com/fastai/course-v3/blob/master/nbs/dl2/06_cuda_cnn_hooks_init.ipynb
class BatchTransformXCallback(Callback):
    _order=2
    def __init__(self, tfm): self.tfm = tfm
    def begin_batch(self): self.run.xb = self.tfm(self.xb)

# dev
class BatchTransformXYCallback(Callback):
    _order=2
    def __init__(self, tfm): self.tfm = tfm
    def begin_batch(self): self.run.xb,self.run.yb = self.tfm(self.xb,self.yb)


# https://github.com/fastai/course-v3/blob/master/nbs/dl2/09_optimizers.ipynb
class Recorder(Callback):
    def begin_fit(self): self.lrs,self.losses = [],[]

    def after_batch(self):
        if not self.in_train: return
        self.lrs.append(self.opt.hypers[-1]['lr'])
        self.losses.append(self.loss.detach().cpu())

    def plot_lr  (self): plt.plot(self.lrs)
    def plot_loss(self): plt.plot(self.losses)

    def plot(self, skip_last=0):
        losses = [o.item() for o in self.losses]
        n = len(losses)-skip_last
        plt.xscale('log')
        plt.plot(self.lrs[:n], losses[:n])



class ParamScheduler(Callback):
    _order=1
    def __init__(self, pname, sched_funcs):
        self.pname,self.sched_funcs = pname,listify(sched_funcs)

    def begin_batch(self):
        if not self.in_train: return
        fs = self.sched_funcs
        if len(fs)==1: fs = fs*len(self.opt.param_groups)
        pos = self.n_epochs/self.epochs
        for f,h in zip(fs,self.opt.hypers): h[self.pname] = f(pos)

def annealer(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner

@annealer
def sched_lin(start, end, pos): return start + pos*(end-start)

@annealer
def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2
@annealer
def sched_no(start, end, pos):  return start
@annealer
def sched_exp(start, end, pos): return start * (end/start) ** pos


#This monkey-patch is there to be able to plot tensors
torch.Tensor.ndim = property(lambda x: len(x.shape))

def cos_1cycle_anneal(start, high, end):
    return [sched_cos(start, high), sched_cos(high, end)]

def combine_scheds(pcts, scheds):
    assert sum(pcts) == 1.
    pcts = tensor([0] + listify(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)
    return _inner

# https://github.com/fastai/course-v3/blob/master/nbs/dl2/11_train_imagenette.ipynb
def create_phases(phases):
    phases = listify(phases)
    return phases + [1-sum(phases)]

# https://github.com/fastai/course-v3/blob/master/nbs/dl2/11a_transfer_learning.ipynb
def sched_1cycle(lrs, pct_start=0.3, mom_start=0.95, mom_mid=0.85, mom_end=0.95):
    phases = create_phases(pct_start)
    sched_lr  = [combine_scheds(phases, cos_1cycle_anneal(lr/10., lr, lr/1e5))
                 for lr in lrs]
    sched_mom = combine_scheds(phases, cos_1cycle_anneal(mom_start, mom_mid, mom_end))
    return [ParamScheduler('lr', sched_lr),
            ParamScheduler('mom', sched_mom)]

# LR finder
# https://github.com/fastai/course-v3/blob/master/nbs/dl2/09_optimizers.ipynb
class LR_Find(Callback):
    _order=1
    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter,self.min_lr,self.max_lr = max_iter,min_lr,max_lr
        self.best_loss = 1e9

    def begin_batch(self):
        if not self.in_train: return
        pos = self.n_iter/self.max_iter
        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
        for pg in self.opt.hypers: pg['lr'] = lr

    def after_step(self):
        if self.n_iter>=self.max_iter or self.loss>self.best_loss*10:
            raise CancelTrainException()
        if self.loss < self.best_loss: self.best_loss = self.loss

# https://github.com/fastai/course-v3/blob/master/nbs/dl2/11a_transfer_learning.ipynb
# you can repalce the __call__ function of any callback like here.
class DebugCallback(Callback):
    _order = 999
    def __init__(self, cb_name, f=None): self.cb_name,self.f = cb_name,f
    def __call__(self, cb_name):
        if cb_name==self.cb_name:
            if self.f: self.f(self.run)
            else:      set_trace()

# https://github.com/fastai/course-v3/blob/master/nbs/dl2/12a_awd_lstm.ipynb
class GradientClipping(Callback):
    def __init__(self, clip=None): self.clip = clip
    def after_backward(self):
        if self.clip:  nn.utils.clip_grad_norm_(self.run.model.parameters(), self.clip)

# For QA task:

class SaveModelCallback(Callback):
    def __init__(self,save_model_func,output_dir, *args, **kwargs):
        self.output_dir, self.save_model_func = output_dir,save_model_func
        self.args,self.kwargs = args, kwargs
    def after_epoch(self):
        self.save_model_func(self, self.output_dir, *self.args, **self.kwargs)

class CudaCallbackMTL(Callback):
    def begin_fit(self): self.model.cuda()
    def begin_batch(self): self.run.xb, self.run.yb = \
        self.xb.cuda(), (self.run.yb[0].cuda(), self.run.yb[1].cuda())

# gradient accumulation
class GradientAccumulation(Callback):
    _order=2
    def __init__(self,bs,effective_bs):
        self.bs, self.effective_bs = bs, effective_bs
    def after_loss(self):
        self.loss.div_(self.effective_bs/self.bs)
    def after_backward(self):
        if self.n_iter*self.bs % self.effective_bs != 0: raise CancelBatchException()

class QAAvgStats(AvgStats):
    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i,m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb, run.xb) * bn

class QAAvgStatsCallback(AvgStatsCallback):
    def __init__(self, metrics):
        self.train_stats,self.valid_stats = QAAvgStats(metrics,True),QAAvgStats(metrics,False)

class TrainStatsCallback(Callback):
    _order=5
    def __init__(self, update_freq_pct=.2):
        self.update_freq_pct = update_freq_pct
        logging.basicConfig()
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def begin_epoch(self):
        self.iter = 0
        self.update_freq = int(self.update_freq_pct*len(self.dl))
        assert self.update_freq > 0, f"stats update frequency - {self.update_freq} - is too low."

    def after_batch(self):
        if not self.in_train: return
        self.iter += 1
        if self.n_iter % self.update_freq == 0:
            metric_names = ["loss"] + [m.__name__ for m in self.qa_avg_stats.train_stats.metrics]
            stats = self.qa_avg_stats.train_stats.avg_stats
            named_stats = {n:st for (n,st) in zip(metric_names,stats)}
            self.logger.info(f"epoch {self.epoch} stats for iter {self.iter} out of {self.iters} iters are : {named_stats}")
