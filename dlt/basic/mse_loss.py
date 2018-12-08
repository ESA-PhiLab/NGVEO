import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#MSE loss for images
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self,x,y, weight=None):
        s = x.data.shape

        x = x.contiguous().view(s[0], 1, -1).transpose(2,1).contiguous().view(-1)
        y = y.contiguous().view(s[0], 1, -1).transpose(2,1).contiguous().view(-1)

        x = x[y != -100]
        y = y[y != -100]

        return F.mse_loss(x, y)

