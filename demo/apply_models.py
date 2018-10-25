# Imports
import dlt.utils.plotting
dlt.utils.plotting.set_localhost()

from demo.datasets import *
from demo.deep_learning import *
from demo.plotting import *

# Select bands and channels and crop-size
setup = {
    'input_bands':['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12'],
    'label': 'vegetation_height', #  either 'vegetation_height' or 'fractional_forest_cover'
    'xy_coord': [0, 7000], #Upper left corner of excerpt
    'size': [300, 300], #max 10980, 10980
    'net_type':'7_classes' #2_classes, 7_classes, tree_height, forest_cover,
}

rbg_bands =  ['B04', 'B03', 'B02']

# Select tile and crop
tiles = load_tiles(4, 'T37LDK')

# Plot one tile
plot_data_for_tile(tiles[0], rbg_bands, setup)

# Plot multiple tiles
plot_data_for_multiple_times(tiles, rbg_bands, setup)

# Load network
print('Available models:', list(best_models.keys()))
net = load_pretrained_net('7_classes', setup)

# Do computations on GPU
net.cuda()

#Apply network to tiles
predictions                 = apply_net_to_tiles(net, tiles, setup)
predictions_multi_temporal  = apply_net_to_tiles_AVERAGE(net, tiles, setup)

#Plot results for the different tiles
plot_predicitions(tiles, predictions, setup)

#Plot when combining multiple times
plot_predicitions(tiles[1], predictions_multi_temporal, setup)

#Scatter/confusion plot for each tile
scatter_confusion_plot(tiles, predictions, setup)

#Use averaging over multiple times
scatter_confusion_plot(tiles[0], predictions_multi_temporal, setup)
