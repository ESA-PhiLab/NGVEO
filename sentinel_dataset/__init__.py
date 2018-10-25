""" This package contains functionallity to read sentinel_dataset files and make datasets for training and evalution """
from sentinel_dataset.dataset import Dataset
from sentinel_dataset.tile import Tile

import numpy as np
def make_visualize_function(data_bands):
    #https://forum.step.esa.int/t/list-of-band-combinations-for-sentinel-2/1156
    RGB_bands = ['b04', 'b03', 'b02']
    RGB_bands = [b.lower() for b in RGB_bands]
    data_bands = [b.lower() for b in data_bands]

    indexes_2_collect = [data_bands.index(b) for b in RGB_bands]

    def collect_rgb_bands(input_dict):
        if type(input_dict) == dict:
            img = input_dict['data']
        else:
            img = input_dict

        if img.shape[0]==1:
            img = img.squeeze(0)
            img = np.moveaxis(img,0,-1)


        if np.max(indexes_2_collect) <= img.shape[2]:
            img = img[:,:,indexes_2_collect]
        else:
            img = img[:,:,0]

        #Normalize
        img*=4
        img[img>1]=1
        
        return img

    return collect_rgb_bands


if __name__ == '__main__':
    """ Example usage """

    #Print usefull info about folder
    #path_to_patches = '/Shared_NGVEO/patches/T37LCK/'
    #path_to_patches = '/home/andersuw/Desktop/COGSAT_DATA/T37LCK/'
    path_to_patches = '/home/auwaldeland/tiles'
    bands = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']

    #Training sentinel_dataset
    train_dataset = Dataset(path_to_patches,
                            preprocessing = None,
                           # filter_on_min_date='2018-04-20',
                           #filter_on_max_date='2018-05-21',
                           band_identifiers=bands,
                           #filter_on_ycoord_range=[0, 9000],
                           label_identifiers=['fractional_forest_cover','vegetation_height','cloud_mask'])

    # Validation sentinel_dataset
    val_dataset = Dataset(path_to_patches,
                          preprocessing=None,
                          #filter_on_min_date='2018-04-20',
                          #filter_on_max_date='2018-05-21',
                          #filter_on_ycoord_range=[10000, 11000],
                          band_identifiers=bands,
                          label_identifiers=['fractional_forest_cover', 'vegetation_height', 'cloud_mask'])

