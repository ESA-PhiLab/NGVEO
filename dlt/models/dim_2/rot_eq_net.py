from layers.rot_eq_net.layers_2D import RotConv, VectorMaxPool, VectorBatchNorm, Vector2Magnitude, VectorUpsampling
from torch import nn


class RotEqNet_224(nn.Module):
    def __init__(self,n_classes = 2):
        super(RotEqNet_224, self).__init__()

        self.main = nn.Sequential(
            RotConv(3, 6, [9, 9], 1, 4, n_angles=8, mode=1),
            VectorMaxPool(2,padding=1),
            VectorBatchNorm(6),

            RotConv(6, 16, [9, 9], 1, 4, n_angles=8, mode=2),
            VectorMaxPool(2,padding=1),
            VectorBatchNorm(16),

            RotConv(16, 32, [9, 9], 2, 4, n_angles=8, mode=2),
            VectorMaxPool(2,padding=1),
            VectorBatchNorm(32),

            RotConv(32, 32, [9, 9], 2, 4, n_angles=8, mode=2),
            VectorMaxPool(3,padding=1),
            VectorBatchNorm(32),

            Vector2Magnitude(),
            # This call converts the vector field to a conventional multichannel image/feature image

            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, n_classes, 3, 1, 0),
        )

    def forward(self, x):
        x = self.main(x)
        return x


if __name__ == '__main__':
    from dl_basics.receptive_field import receptiveField
    import torch
    from torch.autograd import Variable

    if True: # First version
        win = receptiveField(224,'Input',start=True)

        win = receptiveField(win, 'RotConv', k=9, s=1, p=4)
        win = receptiveField(win, 'VectorPool', k=2, s=2, p=1)

        win = receptiveField(win, 'RotConv', k=9, s=1, p=4)
        win = receptiveField(win, 'VectorPool', k=2, s=2, p=1)

        win = receptiveField(win, 'RotConv', k=9, s=2, p=4)
        win = receptiveField(win, 'VectorPool', k=2, s=2, p=1)

        win = receptiveField(win, 'RotConv', k=9, s=2, p=2)
        win = receptiveField(win, 'VectorPool', k=3, s=2, p=1)


        win = receptiveField(win, 'Conv1', k=3, s=1, p=0)

        print RotEqNet_224()(Variable(torch.zeros(1,3,224,224)))