import matplotlib.pyplot as plt
import numpy as np

import socket
#Get root for the given machien
if socket.gethostname() == 'waldeland':
    root = '/pro/cogsat/'
elif socket.gethostname() == 'kickseed':
    root = '/local_disk_1/sentinel2/'

# Load data
from sentinel_dataset import Dataset


def load_tiles(n=1, mode='test'):
    if mode == 'test':
        tiles = ['T37LCJ']
    elif mode == 'val':
        tiles = ['T37LDJ']
    elif mode == 'train':
        tiles = ['T37LCK', 'T37LDK', ]
    elif mode in ['T37LCK', 'T37LDK', 'T37LDJ','T37LCJ']:
        tiles = [mode]
    else:
        print('Error: mode should be one in  ["test", "train", "val","T37LCK", "T37LDK", "T37LDJ","T37LCJ"], got',mode)
        return

    tiles = Dataset([root + tile for tile in tiles],
                    n_time_instances=n,
                    limit_n_tiles=n
                    )

    return tiles.files

def crop(data, setup):
    xy = setup['xy_coord']
    siz = setup['size']
    return np.array(data[:])[xy[0]:xy[0] + siz[0], xy[1]:xy[1] + siz[1]]
