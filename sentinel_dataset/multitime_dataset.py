from __future__ import division, print_function
import numpy as np
import random

from sentinel_dataset.dataset import Dataset

class MultitimeDataset(Dataset):
    def __init__(self, path,
                 label_identifiers=None,
                 band_identifiers=None,
                 filter_on_min_date=None,
                 filter_on_max_date=None,
                 win_size_for_calc_overlapping_patches=[256, 256],
                 n_time_instances = 1,
                 random_time_instances = True
                 ):
        super(MultitimeDataset,self).__init__( path,
                 label_identifiers=None,
                 band_identifiers=None,
                 filter_on_min_date=None,
                 filter_on_max_date=None,
                 win_size_for_calc_overlapping_patches=[256, 256],
                 )

        self.n_time_instances = n_time_instances
        self.random_time_instances = random_time_instances


    def _getitem__(self, sample_no, non_overlapping):
        #Get first tile
        tile_no, coord_no = self.convert_sampleno_to_tileno_coordno(sample_no, non_overlapping)
        tile = self.tiles[tile_no]

        #Get tiles that correspond with the first tile (have the same tile_id)
        other_tiles = [t for t in self.tiles if t.tile_id == tile.tile_id]

        #Pick other tiles
        if not self.random_time_instances:
            random.seed(0)
        other_tiles = random.sample(other_tiles, self.n_time_instances - 1)

        # Get output from the first tile
        output = super(MultitimeDataset, self)._getitem(sample_no, non_overlapping)

        #Merge data and missing_data_mask from the other tiles
        for tile in other_tiles[1:]:
            output["data"] += tile.get_data(self.band_identifiers)
            output["data_bands"] += self.band_identifiers
            output["missing_data_mask"] += tile.get_missing_mask()  # Todo: make this work for sentinel1. Any comments AB?

        return output