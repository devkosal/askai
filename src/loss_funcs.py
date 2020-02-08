import torch.nn.functional as F
from torch import nn
from .utils import *


def cross_entropy_qa_mtl(input, target):
    """
    Summing the cross entropy loss from the starting and ending indices and the secondary binary label
    """
    loss = torch.add(F.cross_entropy(input[0], target[0][:,0]) , F.cross_entropy(input[1], target[0][:,1]))
    poss_loss = F.cross_entropy(input[2], target[1])
    return torch.add(loss, poss_loss)


def cross_entropy_qa_mtl_wtd(input, target):
    """
    Summing the cross entropy loss from the starting and ending indices and the secondary binary label
    Plus: answerable labels are given more weight
    """
    mask = (~target[1].bool()).float()
    qa_loss = torch.add(F.cross_entropy(input[0], target[0][:,0], reduction="none"),\
                        F.cross_entropy(input[1], target[0][:,1], reduction="none"))
    qa_loss.mul_(mask+1) # doubles the loss assigned to all answerable labels
    wtd_qa_loss = qa_loss.mean() # reduces the loss by taking the mean

    poss_loss = F.cross_entropy(input[2], target[1])

    return torch.add(wtd_qa_loss, poss_loss)
