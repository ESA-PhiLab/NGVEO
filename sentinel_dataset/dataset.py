from __future__ import division, print_function
import os
import numpy as np
import traceback
import random

#from sentinel_dataset.filter_date import filter_on_date
from sentinel_dataset.tile import Tile
from sentinel_dataset._utils import get_files_and_info
from sentinel_dataset._filter_date import filter_on_date

class Dataset():
    """ Collect DataFiles to create a sentinel_dataset"""
    rgb_bands = ['b04','b03','b02']

    def __init__(self, path,
                 label_identifiers=None,
                 band_identifiers=None,
                 filter_on_min_date=None,
                 filter_on_max_date=None,
                 win_size_for_calc_overlapping_patches=[256, 256],
                 ):
        print('Loading sentinel_dataset from folder:', path)

        self.root = path
        self.band_identifiers = band_identifiers
        self.label_identifiers = label_identifiers
        self.filter_on_min_date = filter_on_min_date
        self.filter_on_max_date = filter_on_max_date

        #Get list of files
        files, roots, dates, file_descriptions = get_files_and_info(path)

        if len(files) == 0:
            raise ValueError('No files imported')

        #Filter files
        keep_files = np.array([True]*len(files))

        if self.filter_on_min_date is not None:
            keep_files *= filter_on_date(dates, self.filter_on_min_date, 'min')

        if self.filter_on_max_date is not None:
            keep_files *= filter_on_date(dates, self.filter_on_max_date, 'max')

        file_paths = files[keep_files]
        roots = roots[keep_files]

        #Load DataFile objects
        self.tiles = []
        for file_path, roots in zip(file_paths, roots):
            self.tiles.append(Tile(roots, win=win_size_for_calc_overlapping_patches))

        #Print to user
        print(' -', len(self.tiles), 'were loaded.')
        if np.sum(keep_files)==0:
            print(' -',np.sum(np.logical_not(keep_files)),'All files where removed due to filter_on_*_date.')
            raise Exception()

        self.count()
        print('')

    def count(self):
        #Measure length of sentinel_dataset. This the sum of the number of labelled pixels (overlapping or non-overlapping) in all tiles
        self.N =  0
        self.N_overlapping = 0
        for f in self.tiles:
            self.N += f.n_non_overlapping()
            self.N_overlapping += f.n_overlapping()
        print(' -', 'Total number of non-overlapping labelled samples:', self.N)
        print(' -', 'Total number of overlapping labelled samples:', self.N_overlapping)

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        """ Return a sentinel_dataset and labels for a file """
        return self._getitem(item, non_overlapping=True)

    def get_random_sample(self):
        item = np.random.randint(0, self.N_overlapping)
        return self._getitem(item, non_overlapping=False)

    def convert_sampleno_to_tileno_coordno(self, sample_no, non_overlapping):
        """ Takes the sample_no and return the corresponding file and coordinate """
        N = 0
        coord_no = 0
        selected_file_no = 0
        for file_no in range(len(self.tiles)):
            file = self.tiles[file_no]

            file_length = file.n_non_overlapping() if non_overlapping else file.n_overlapping()
            coord_no = sample_no - N

            if coord_no < file_length:
                selected_file_no = file_no
                break
            else:
                N += file_length
        return selected_file_no, coord_no

    def _getitem(self, sample_no, non_overlapping):

        tile_no, coord_no = self.convert_sampleno_to_tileno_coordno(sample_no, non_overlapping)

        #Find the given tile id and filename
        tile = self.tiles[tile_no]

        #Get coordinate of center pixel
        if non_overlapping:
            coord = tile.get_non_overlapping_coordinate_no(coord_no)
        else:
            coord = tile.get_overlapping_coordinate_no(coord_no)

        #Return dictionary
        return {'data':tile.get_data(self.band_identifiers),
                'data_bands': self.band_identifiers,
                'target':tile.get_labels(self.label_identifiers),
                'target_bands': self.label_identifiers,
                'missing_data_mask':tile.get_missing_mask(),  #Todo: make this work for sentinel1. Any comments AB?,
                'name': tile.file_name + ' y:' + str(coord[0]) + ' x:' + str(coord[1]),
                'coord':list(coord),
                'tile_object':tile}




