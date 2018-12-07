from __future__ import division, print_function
import os
import numpy as np
import traceback
import random

#from sentinel_dataset.filter_date import filter_on_date
from sentinel_dataset.tile import Tile
from sentinel_dataset._utils import get_files_and_info
from sentinel_dataset._filter_date import filter_on_date

class MultiSensorDataset():

    def __init__(self, datasets, master_dataset_no = None):
        self.datasets = datasets
        self.master_dataset_no = master_dataset_no

    def __len__(self):
        # Define length of dataset to be the length og the master or the minimum of the datasets
        if self.master_dataset_no is not None:
            return self.datasets[self.master_dataset_no].N
        else:
            return np.min([d.N for d in self.datasets])

    def __getitem__(self, item):
        """ Return a sentinel_dataset and labels for a file """
        return self._getitem(item, non_overlapping=True)

    def get_random_sample(self):
        if self.master_dataset_no is not None:
            N_overlapping = self.datasets[self.master_dataset_no].N_overlapping
        else:
            N_overlapping = np.min([d.N_overlapping for d in self.datasets])
        item = np.random.randint(0, N_overlapping)
        return self._getitem(item, non_overlapping=False)

    def _getitem(self, sample_no, non_overlapping):

        def _find_closest_file(master_tile, slave_dataset):
            """ Function that find tiles that are the closest to the master-tile"""
            # diff_days = []
            diff_days = np.zeros(len(slave_dataset.tiles)) + 1e9
            for i, tile in enumerate(slave_dataset.tiles):
                if tile.tile_id.upper() == master_tile.tile_id.upper():
                    date_master = master_tile.get_meta_data()["datetime"]
                    date_slave = tile.get_meta_data()["datetime"]
                    delta = date_slave - date_master
                    delta /= np.timedelta64(1, 's')
                    diff_days[i] = delta
                # diff_days.append(delta)
            # diff_days = np.array(diff_days)
            selected_file_no = np.argmin(np.abs(diff_days))
            return selected_file_no

        if self.master_dataset_no is None:
            #We draw a random master-dataset
            master_dataset_no = np.random.randint(len(self.datasets))
        else:
            master_dataset_no = self.master_dataset_no

        #Get the master tile
        master_tile_no, master_sample_no = self.datasets[master_dataset_no].convert_sampleno_to_tileno_coordno(sample_no, non_overlapping)
        master_tile = self.datasets[master_dataset_no].tiles[master_tile_no]

        #Use master tile + the other tiles that are closest to master
        tile_numbers = [_find_closest_file(master_tile, ds)
                            if no != self.master_dataset_no else
                         master_tile_no
                         for no,ds in enumerate(self.datasets)
                         ]

        #Collect the tile-objects
        tiles = [self.datasets[ds_no].tiles[tile_no] for ds_no, tile_no in enumerate(tile_numbers)]

        # Get coordinate of center pixel
        if non_overlapping:
            coord = tiles[0].get_non_overlapping_coordinate_no(master_sample_no)
        else:
            coord = tiles[0].get_overlapping_coordinate_no(master_sample_no)

        #Get the data for the first tile
        output = {'data':tiles[0].get_data(self.datasets[0].band_identifiers),
                'data_bands': self.datasets[0].band_identifiers.copy(),
                'target':tiles[0].get_labels(self.datasets[0].label_identifiers),
                'target_bands': self.datasets[0].label_identifiers.copy(),
                'missing_data_mask':tiles[0].get_missing_mask(),  #Todo: make this work for sentinel1. Any comments AB?,
                'name': tiles[0].file_name + ' y:' + str(coord[0]) + ' x:' + str(coord[1]),
                'coord':list(coord),
                'tile_objects':tiles}


        # Get the data for the other tiles
        for i, tile in enumerate(tiles[1:]):
            output["data"] += tile.get_data(self.datasets[i+1].band_identifiers)
            output["data_bands"] +=  self.datasets[i+1].band_identifiers.copy()
            output["missing_data_mask"] += tile.get_missing_mask() #Todo: make this work for sentinel1. Any comments AB?

        return output
