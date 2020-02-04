import torch.nn.functional as F
from torch import nn
from .utils import *

def cross_entropy_qa_mtl(input, target):
    """
    Summing the cross entropy loss from the starting and ending indices.
    """
    loss = torch.add(F.cross_entropy(input[0], target[0][:,0]) , F.cross_entropy(input[1], target[0][:,1]))
    poss_loss = F.cross_entropy(input[2], target[1])
    return torch.add(loss, poss_loss)

# def cross_entropy_qa_mtl(input, target):
#     """
#     Summing the cross entropy loss from the starting and ending indices.
#     """
#     assert_no_negs(target[0])
#     assert_no_negs(target[1])

#     mask = (~target[1].bool()).float()
#     qa_loss = torch.add(F.cross_entropy(input[0], target[0][:,0], reduction="none"),\
#                         F.cross_entropy(input[1], target[0][:,1], reduction="none"))/2.0
#     qa_loss.mul_(mask+1)
#     wtd_qa_loss = qa_loss.mean()

#     imp_loss = F.cross_entropy(input[2], target[1])

#     loss = torch.add(wtd_qa_loss, imp_loss)
#     return loss
