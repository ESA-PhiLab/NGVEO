import sys
import numpy as np
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adam

from dlt.basic.batch import make_batch
from dlt.basic.pytorch_utils import torch_to_np, np_to_torch
from dlt.basic.summary import regression_summary, classification_summary
from dlt.basic.unet import UNet
from sentinel_dataset import Dataset
import os

data_bands = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']


GPU_NO = 0
batch_size = 8
win_size = [256, 256]
n_iteration = 30000
lr = 0.00004

# We showcase to simple examples; cloud detection and atmospheric correction. These tasks can easily be replaced by
# other tasks if required  by replacing the labels. Labels may be provided as GeoTiffs and converted to the np-memmap-
# structure with the data-preparation tools.

#Make output folder
if not os.path.isdir('saved_models'):
    os.mkdir('saved_models')

if len(sys.argv) <2:
    sys.argv.append(2)

if int(sys.argv[1])==1: #Cloud detection (classification problem)
    label_bands = ['cloud']
    criterion = CrossEntropyLoss()
    summary = classification_summary
    output_path = 'saved_models/cloud_detection.pt'

else: #Atmospheric correction (regression problem)
    label_bands = ['B02']
    summary = regression_summary
    output_path = 'saved_models/atmospheric_correction_b02.pt'
    criterion_ = MSELoss()

    # A small rewrite of MSELoss to incorporate ignore-values, indicated by -100
    def criterion(pred,target):
        #Remove samples with target==-100
        pred = pred[target != -100]
        target = target[target != -100]
        return criterion_(pred, target)

#Model and optimizer
model = UNet(num_classes=len(label_bands), in_channels=len(data_bands)).cuda(GPU_NO);
optimizer = Adam(model.parameters(),lr=lr)

# Datasets (T32TMM is reserved for test)
train_dataset = Dataset([os.path.join('data',p) for p in ["T32TNS",]], #["T32TMP","T32TNS","T32TNP","T32TPR","T32TML","T32TMK","T32TNT","T32UPU","T32TPT","T32UQU"]],
                        band_identifiers=data_bands,
                        label_identifiers=label_bands,
                        )

val_dataset = Dataset([os.path.join('data',p) for p in ["T32TNS",]], #["T32UNU","T32TMN"]],
                      band_identifiers=data_bands,
                      label_identifiers=label_bands,
                      )

#Traing steps
for iteration in range(n_iteration+1):
    model.train()

    data, target = make_batch(train_dataset, win_size, batch_size)

    #Put data on gpu
    data = np_to_torch(data).cuda(GPU_NO)
    target = np_to_torch(target).cuda(GPU_NO)
    if criterion == CrossEntropyLoss:
        target = target.long()
    else:
        target = target.float()

    #Run data through network
    pred = model(data)

    #Run training step
    loss = criterion(pred,target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Move data from torch to numpy
    data = torch_to_np(data)
    pred = torch_to_np(pred)
    target = torch_to_np(target)
    loss = torch_to_np(loss)

    #Print results for batch
    summary(iteration, 'Training', data, target, pred, loss)


    #Print results for validation every 100th epoch
    if iteration%100==0:
        model.eval()

        #Loop through 20 batches:
        pred = []
        target = []
        data = []
        for i in range(20):
            d, t = make_batch(val_dataset, win_size, batch_size)
            p = torch_to_np(model(np_to_torch(d).cuda(GPU_NO)))

            pred.append(p)
            target.append(t)
            data.append(d)

        pred = np.concatenate(pred, 0)
        target = np.concatenate(target, 0)
        data = np.concatenate(data, 0)

        summary(iteration, 'Validation', data, target, pred)

    #Save trained model
    if iteration%1000==0:
        torch.save(model.state_dict(), output_path)
        print('Saving model at iteration', iteration)
