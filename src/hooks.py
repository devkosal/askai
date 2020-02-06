from functools import partial
from .utils import *
assert ListContainer != None


def children(m): return list(m.children())

class Hook():
    """base Hook class"""
    def __init__(self, m, f): self.hook = m.register_forward_hook(partial(f, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()
        
from torch.nn import init

class Hooks(ListContainer):
    """Object used to generate stats"""
    def __init__(self, ms, f): super().__init__([Hook(m, f) for m in ms])
    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.remove()
    def __del__(self): self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)
        
    def remove(self):
        for h in self: h.remove()

def append_stats(hook, mod, inp, outp):
    if not hasattr(hook,'stats'): hook.stats = ([],[])
    means,stds = hook.stats
    means.append(outp.data.mean())
    stds .append(outp.data.std())
    

def model_summary(learn, data, find_all=False, print_mod=False):
    """
    generates model summary with shapes using hooks
    https://github.com/fastai/course-v3/blob/master/nbs/dl2/11_train_imagenette.ipynb
    """
    model = learn.model
    xb,yb = get_batch(data.valid_dl, learn)
    mods = find_modules(model, is_lin_layer) if find_all else model.children()
    f = lambda hook,mod,inp,out: print(f"====\n{mod}\n" if print_mod else "", out.shape)
    with Hooks(mods, f) as hooks: learn.model(xb)    
    
    
   
