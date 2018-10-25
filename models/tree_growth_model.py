import torch
from torch import nn
import numpy as np

from dlt.losses.mse_loss import MSELoss
from dlt.models.dim_2.unet import UNet
from dlt.utils.pytorch import gpu_no_of_var, np_to_var, var_to_np


class UNet_tree_growth(nn.Module):
    valid = True
    fow = [128+64,128+64]
    type = 'seg'

    def __init__(self, in_channels, n_classes, net=UNet):
        super(UNet_tree_growth, self).__init__()
        self.n_classes = 1
        #Model
        self.net = net(in_channels = in_channels, n_classes=1)
        self.bn = nn.BatchNorm2d(1)
        self.criterion = MSELoss()
        #Parameters in growth model
        self.a = nn.Parameter(torch.Tensor([1]))
        self.b = nn.Parameter(torch.Tensor([1]))

    def forward(self, data, time_since_gt):

        self.current_age = self.net(data) #Estimate age now
        age_at_gt = self.current_age - time_since_gt/1000
        log_like_func = lambda x: torch.sign(x)*(torch.abs(x))**(1/3.)

        height = self.a * log_like_func(age_at_gt) + self.b
        print('\nDiff', time_since_gt/1000,'\na',self.a,'\nb',self.b )

        return height

    def parameters(self):
        return [{'params':self.net.parameters()}, {'params':[self.a,self.b],'lr':0.00001}]

    def intersect_train(self, input):
        samples, timedeltas, target = self._parse_input(input)
        height = self(samples, timedeltas)


        height_sorted,sorting = torch.sort(height.view(-1))
        target_sorted = target.view(-1)[sorting]
        n = target_sorted.size()[0]

        ignore_n = int(n*0.1) #If we have cuttings in between, we can allow the shortest heights (recently cut) to be ignored)
        loss = self.criterion(height_sorted[ignore_n:], target_sorted[ignore_n:])
        loss.backward()
        out = {'net_output':var_to_np(height), 'loss':var_to_np(loss)}

        del height, samples, timedeltas, target, sorting, height_sorted, target_sorted, loss
        return out

    def intersect_infer(self, input):
        #Handle input as list of batches
        if len(input)<=4:
            input = [input]

        output = []
        lbl = []
        loss = []
        data = []
        for input_ in input:
            samples, timedeltas, target = self._parse_input(input_ )
            out = self(samples, timedeltas)
            output.append(var_to_np(out))
            data.append(var_to_np(samples))
            lbl.append(var_to_np(target))
            if target is not None:
                l = self.criterion(out, target)
                loss.append(var_to_np(l))

            del out, samples, timedeltas, target,l

        out = np.concatenate(output,0)
        return {'net_output':out,'predicted_class':out,'class_probability':out,'loss':loss}

    def _parse_input(self, input):
        data  = input[0]

        timedeltas  = []
        samples = []
        for i in range(len(data)):
            timedeltas.append(np.array([data[i][1]]))
            samples.append( np.expand_dims(np.moveaxis(data[i][0],-1,0),0))
        samples = np.concatenate(samples,0)
        samples = self._gpu(samples)
        timedeltas = np.concatenate(timedeltas,0)
        timedeltas = np.expand_dims(timedeltas, -1)
        timedeltas = np.expand_dims(timedeltas, -1)
        timedeltas = np.expand_dims(timedeltas, -1)
        timedeltas = self._gpu(timedeltas)

        if len(input)>0:
            target = self._gpu(input[1])
        else:
            target = None

        return samples, timedeltas, target

    def _gpu(self,x):
        return np_to_var(x, gpu_no_of_var(self.a))
