import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#Cross entropy for images
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self,x,y, weight=None):
        s = x.data.shape

        x = x.contiguous().view(s[0], s[1], -1).transpose(2,1).contiguous().view(-1, s[1])
        y = y.contiguous().view(s[0], 1, -1).transpose(2,1).contiguous().view(-1)

        return F.cross_entropy(x, y.long(), weight=weight)