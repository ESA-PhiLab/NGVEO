import torch
from torch import nn
from dlt.models.dim_2.unet import UNet
import numpy as np

from dlt.utils.pytorch import np_to_var, gpu_no_of_var


class Multitime_Average(nn.Module):
    valid = False
    fow = 128
    dim = 2
    label_fow=128
    pad = [0]
    type = 'classification'


    def __init__(self, in_channels, n_classes, net=UNet, mode='avg'):
        super(Multitime_Average, self).__init__()
        self.net = net(in_channels = in_channels, n_classes=n_classes)
        self.mode = mode
        if n_classes == 1:
            self.type = 'regression'

    def intersect_forward(self, input_dict):
        #Read number of time-instances
        n_time_instances = input_dict['n_time_instances']
        try:
            assert(np.all(np.array(n_time_instances) == n_time_instances[0]))
            n_time_instances = n_time_instances[0]
        except:
            raise NotImplementedError('This only works when all samples in batch have the same number of time-instances')


        #If there is just one time_instance we just put it through the forward function
        if n_time_instances==1:
            data = np_to_var(input_dict['data'], gpu_no_of_var(self)).float()
            out =  self.forward(data)
            del data
            return out

        else:

            n_bands = len(input_dict['data_bands'][0])
            n_samples, n_ch, h, w = input_dict['data'].shape


            #Split into multiple batches
            batches = []
            for i in range(n_time_instances):
                batches.append(input_dict['data'][:, (i) * n_bands:(i + 1) * n_bands, :, :])

            #Put on GPU
            batches = [np_to_var(b, gpu_no_of_var(self)).float() for b in batches]

            #Run through network
            outputs = [self(x) for x in batches]

            #Average/median
            combined = [o.view([n_samples, o.size()[1], h, w, 1]) for o in outputs]
            combined = torch.cat(combined, -1)

            cloud_mask = np_to_var(input_dict['cloud'], gpu_no_of_var(self))
            n_cloud_free = n_time_instances - torch.sum(cloud_mask,1,keepdim=True)
            n_cloud_free[n_cloud_free==0] = 1 #Avoid 0-division
            no_cloud_mask = 1-cloud_mask

            if self.mode =='avg':
                for i in range(no_cloud_mask.shape[1]):
                    combined[:,:,:,:,i] =  combined[:,:,:,:,i]*no_cloud_mask[:,i:i+1,:,:]
                combined = torch.sum(combined,-1,keepdim=False)/n_cloud_free
            elif self.mode == 'median':
                combined = torch.median(combined, axis=-1, keepdim=False)

            #Save GPU mem
            del batches, outputs, cloud_mask, no_cloud_mask, n_cloud_free  # Save some GPU memory

            return combined

    def forward(self, x):
        return self.net(x)
