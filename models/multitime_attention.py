import torch
from torch import nn



class Multitime_Attention(nn.Module):
    valid = False
    fow = 128
    dim = 2
    label_fow=128
    pad = [0]


    def __init__(self, in_channels, n_classes, net=UNet, mode=1):
        super(Multitime_Attention, self).__init__()
        self.in_channels = in_channels
        if mode == 1:
            self.attention_model = nn.Sequential(
                nn.Conv2d(in_channels, 10, 1, 1),
                nn.ReLU(),
                nn.Conv2d(10, 5, 1, 1),
                nn.ReLU(),
                nn.Conv2d(5, 1, 1, 1),
                nn.ReLU(),
            )
        else:
            self.attention_model = nn.Sequential(
                nn.Conv2d(in_channels, 10, [3,3], 1, padding=1),
                nn.ReLU(),
                nn.Conv2d(10, 5, [3,3], 1, padding=1),
                nn.ReLU(),
                nn.Conv2d(5, 1, [3,3], 1, padding=2),
                nn.ReLU()
            )

        self.net = net(in_channels = in_channels, n_classes=n_classes)
        self.n_classes = n_classes
        self.a = torch.Tensor([1])
        self.b = torch.Tensor([1])


    def forward(self, x):
        #Number of time-series as input
        n_inputs = x.size()[1] // self.in_channels

        #Compute score from attenion model
        score = []
        for i in range(n_inputs):
            cut = x[:,i*self.in_channels:(i+1)*self.in_channels,:,:]
            score.append(self.attention_model(cut))
        score = torch.cat(score,1)
        score = F.softmax(score-torch.max(score,1,keepdim=True)[0],1)

        #Make attention weighted data

        for i in range(n_inputs):
            cut = x[:,i*self.in_channels:(i+1)*self.in_channels,:,:]
            if i == 0:
                composed = cut*score[:,i:i+1,:,:]
            else:
                composed += cut*score[:,i:i+1,:,:]

        y = self.net(composed)

        self.score = score
        self.composed = composed
        return y