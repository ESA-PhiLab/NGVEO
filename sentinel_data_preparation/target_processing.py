import os
from sentinel_data_preparation import utils
import numpy as np
import rasterio
import matplotlib.pyplot as plt

class TargetProcessing():
    def __init__(self, params):
        self.params = params
        self.ignore_value = np.nan

    def process_target_data(self, sentinel_file, tile_id=None):

        #Get the tile id
        if tile_id == None:
            tile_id = utils.get_tile_id(sentinel_file, self.params['tile_ids'])

        if tile_id == None:
            return -1

        # Read target data
        target_filename = os.path.join(self.params['target_dir'], tile_id.lower() + "_"
                                       + self.params['target_basename'] + ".tif")
        with rasterio.open(target_filename) as target_file:
            target_data = target_file.read()


            # Create a boolean mask
            mask = np.zeros(target_data[0].shape).astype(np.bool)
            mask[target_data[0] >= 0] = True

            #Set invalid areas equal to ignore_value
            target_data[target_data<0] = self.ignore_value

            # Saving the cloud mask as memory map
            basename = os.path.splitext(os.path.basename(sentinel_file))[0]
            tiles_dir = os.path.join(self.params['outdir'], tile_id, basename)

            if not os.path.exists(tiles_dir):
                os.makedirs(tiles_dir)

            for band_ind, target_name in enumerate(self.params['target_names']):
                utils.save_np_memmap(os.path.join(tiles_dir, target_name), target_data[band_ind], 'float32')

            # Save list of coordinates of labelled pixels
            utils.save_list_of_labelled_pixels(mask, tiles_dir)

            return 0

    def display_image(self, sentinel_file, tile_id=None):

        #Get the tile id
        if tile_id == None:
            tile_id = utils.get_tile_id(sentinel_file, self.params['tile_ids'])

        if tile_id == None:
            return -1

        basename = os.path.splitext(os.path.basename(sentinel_file))[0]
        tiles_dir = os.path.join(self.params['outdir'], tile_id, basename)

        for band_ind, target_name in enumerate(self.params['target_names']):
            target_mmap = os.path.join(tiles_dir, target_name + '.dat')
            target_data = np.memmap(target_mmap, dtype='float32', mode='r', shape=(10980,10980))
            #TODO: Read metadata to get shape
            plt.imshow(target_data)
            plt.show()

