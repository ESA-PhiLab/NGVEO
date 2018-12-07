import torch
import numpy as np
import os

from dlt.basic.predict_on_large_tile import apply_net_to_large_data
from dlt.basic.unet import UNet
from sentinel_dataset import Dataset

#Configuration
network_path = 'trained_models/forest_cover.pt'
target_is_classes = False #Put to false for regression problems
n_output_channels = 1
model_name = 'forest_cover'

data_bands = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
pred_win_size = [1024, 1024]
window_overlap = [50, 50]

#Load model with weights
net = UNet(n_output_channels, len(data_bands))
weights = torch.load(network_path, map_location=lambda storage, loc: storage)
net.load_state_dict(weights)

#Move model to GPU to enable GPU-computing
net.cuda()

#Put model in evaluation mode
net.eval()

#Load test-tile
dataset = Dataset( 'data/T32TMM/', band_identifiers=data_bands )

# Loop through tiles
for tile in dataset.tiles:
    print('Predicting for tile', tile.tile_id, tile.file_name)

    #Get data
    data = tile.get_data(data_bands)
    data = [np.expand_dims(d,-1) for d in data]
    data = np.concatenate(data,-1)

    #Run through network
    output = apply_net_to_large_data(data, net, pred_win_size, window_overlap, apply_classifier=target_is_classes)

    #Save output as GeoTiff
    tile.export_prediction_to_tif( os.path.join('/local_disk_2/', tile.file_name +'_' + model_name +'.tif'), output)



