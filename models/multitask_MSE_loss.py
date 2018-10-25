import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#MSE loss for images
from dlt.losses.mse_loss import MSELoss


class Multitask_MSELoss(nn.Module):
    type='regression'

    def __init__(self):
        super(Multitask_MSELoss, self).__init__()
        self.mse_loss_tree_height = MSELoss()
        self.mse_loss_tree_cover = MSELoss()

    def forward(self,x,y, weight=None):
        tree_height_loss= self.mse_loss_tree_height(x[:, 0:1, :, :], y[:, 0:1, :, :])

        #Scale tree_cover to be btween 0 and 5 (closer to treeheight_loss...)
        tree_cover_loss = self.mse_loss_tree_cover(x[:, 1:, :, :] / 20., y[:, 1: , :, :] / 20.)

        return tree_height_loss + tree_cover_loss

