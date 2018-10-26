import torch
from models.multitime_avg import Multitime_Average as UNet
import numpy as np
################### main deep learning functions ######################
from torch.autograd import Variable

from demo_forest.datasets import crop

best_models = {
    '2_classes':'Multitime_Average - 2 cls - 02.19.16, October 03, 2018',
    '7_classes':'Multitime_Average - 7 cls - 08.30.42, October 02, 2018',
    'tree_height':'Multitime_Average - tree height - 07.22.35, October 02, 2018/',
    'forest_cover':'Multitime_Average - 2 cls - 05.56.45, October 02, 2018',
    #'multi_task':'Multitime_Average - 2 cls - 05.56.45, October 02, 2018',
}
n_classes = {
    '2_classes':2,
    '7_classes': 7,
    'tree_height': 1,
    'forest_cover':1,
    'multi_task':2
}
class_defs_2 = [[0,2], [5,100]]
class_defs_7 = [[0, 1], [1, 2], [2, 5], [5, 10], [10, 20], [20, 40], [40, 100]]

def load_pretrained_net(setup):
    net_type = setup['net_type']
    if net_type not in best_models.keys():
        print('Error: model_type',net_type,'is not in allowed model_types:',best_models.keys())
        return

    path = 'trained_models/' + net_type + '.pt'
    print('Loading model:', path)

    net = UNet(len(setup['input_bands']), n_classes[net_type])
    net.fow = [512, 512]
    net.eval()
    net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    print(net)
    return net

def apply_net_to_tiles(net, tiles, setup):
    predictions = []
    for i, tile in enumerate(tiles):
        data = tile.get_data(setup['input_bands'],)
        data = [crop(d, setup) for d in data]
        data = [np.expand_dims(d, -1) for d in data]
        data = np.concatenate(data, -1)

        cloud = crop(tile.get_cloud_mask()[0], setup)

        predictions.append(apply_net_to_data(data, cloud, setup['input_bands'], net, net.fow, 1))
    return predictions

def apply_net_to_tiles_AVERAGE(net, tiles, setup):
    data = []
    cloud = []
    #Loop through tiles and concat channels
    for i, tile in enumerate(tiles):
        data_for_tile = tile.get_data(setup['input_bands'])
        data_for_tile = [crop(d, setup) for d in data_for_tile]
        data_for_tile = [np.expand_dims(d, -1) for d in data_for_tile]
        data_for_tile = np.concatenate(data_for_tile, -1)
        data.append(data_for_tile)

        cloud_for_tile = crop(tile.get_cloud_mask()[0], setup)
        cloud.append(cloud_for_tile)

    #Run through network
    data = np.concatenate(data, -1)
    cloud = np.concatenate(cloud, -1)
    return apply_net_to_data(data, cloud, setup['input_bands'], net, net.fow, len(tiles))

def train_net(net, training_tiles):
    #Todo:
    pass


##################### utility deep learning functions #################

def apply_net_to_data(data, cloud, bands, net, in_shape, n_avg):

    #Functions to convert between B x C x H x W format and W x H x C format
    def hwc_to_bchw(x):
        return np.expand_dims(np.moveaxis(x,-1,0),0)

    def bcwh_to_hwc(x):
        return np.moveaxis(x.squeeze(0),0,-1)


    if type(in_shape) == int:
        in_shape = [in_shape,in_shape]


    #Make output array
    predictions = data[:,:,0]*0

    # Loop through patches given center pixel
    start_coords_0 = np.arange(0, predictions.shape[0], in_shape[0])
    start_coords_1 = np.arange(0, predictions.shape[1], in_shape[1])

    for x0 in start_coords_0:
        for x1 in start_coords_1:
            #Cut out a small patch of the data
            data_patch = data[x0:x0 + in_shape[0], x1:x1 + in_shape[1],:]
            cloud_patch = cloud[x0:x0 + in_shape[0], x1:x1 + in_shape[1]]

            # Pad with zeros if we are at the edges
            p0 = in_shape[0] - data_patch.shape[0]
            p1 = in_shape[1] - data_patch.shape[1]

            if p0 > 0:
                data_patch = np.pad(data_patch, [ [0, p0], [0, 0], [0, 0]], 'constant')
                cloud_patch = np.pad(cloud_patch, [[0, p0], [0, 0], [0, 0]], 'constant')

            if p1 > 0:
                data_patch = np.pad(data_patch, [[0, 0], [0, p1], [0, 0]], 'constant')
                cloud_patch = np.pad(cloud_patch, [ [0, 0], [0, p1], [0, 0]], 'constant')

            # Run it through model
            input_dict = {'n_time_instances':[n_avg], 'cloud':hwc_to_bchw(cloud_patch), 'data':hwc_to_bchw(data_patch), 'data_bands': [bands]}
            out = net.intersect_forward(input_dict)
            out = bcwh_to_hwc(torch_to_np(out))

            #Suftmax + argmax for classifications
            if out.shape[2]>1:
                out = np.argmax(softmax(out),axis=2)

            # Insert output in out array
            predictions[x0:x0 + in_shape[0] - p0, x1:x1 + in_shape[1] - p1] = out[0:in_shape[0] - p0, 0:in_shape[1] - p1].squeeze()

    return predictions

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

#Take a pytorch variable and make numpy
def torch_to_np(var):
    try:
        var = var.cpu()
    except:
        None
    var = var.data
    var = var.numpy()


    return var