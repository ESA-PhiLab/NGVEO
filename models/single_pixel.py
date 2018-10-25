import torch.nn as nn


class SinglePixelNet(nn.Module):
    valid = True
    fow = 128
    type = 'seg'


    def __init__(self, in_channels, n_classes, use_SLAVI=False, use_GARI = False):
        super(SinglePixelNet, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 20, 1),
            nn.BatchNorm2d(20),
            nn.ReLU(),

            nn.Conv2d(20, 40, 1),
            nn.BatchNorm2d(40),
            nn.ReLU(),

            nn.Conv2d(40, 40, 1),
            nn.BatchNorm2d(40),
            nn.ReLU(),

            nn.Conv2d(40, 40, 1),
            nn.BatchNorm2d(40),
            nn.ReLU(),

            nn.Conv2d(40, n_classes, 1),
        )


    def forward(self, x):
        y = self.main(x)
        return y




