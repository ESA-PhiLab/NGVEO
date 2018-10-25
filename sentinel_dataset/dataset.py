from __future__ import division, print_function
import os
import numpy as np
import traceback
import random

#from sentinel_dataset.filter_date import filter_on_date
from sentinel_dataset.tile import Tile
from sentinel_dataset.utils import get_files_and_info
from sentinel_dataset.filter_date import filter_on_date

class Dataset():
    """ Collect DataFiles to create a sentinel_dataset"""
    rgb_bands = ['b04','b03','b02']

    def __init__(self, path,
                 label_identifiers=None,
                 band_identifiers=None,
                 n_time_instances=1,
                 random_time_instances = False,
                 filter_on_min_date=None,
                 filter_on_max_date=None,
                 filter_on_ycoord_range=None,
                 filter_on_xcoord_range=None,
                 limit_n_tiles=np.inf,
                 win_size_for_calc_overlapping_patches=[256, 256],
                 split = None
                 ):
        print('Loading sentinel_dataset from folder:', path)

        self.root = path
        self.band_identifiers = band_identifiers
        self.label_identifiers = label_identifiers
        self.n_time_instances = n_time_instances
        self.random_time_instances = random_time_instances

        self.filter_on_min_date = filter_on_min_date
        self.filter_on_max_date = filter_on_max_date
        self.filter_on_ycoord_range = filter_on_ycoord_range
        self.filter_on_xcoord_range = filter_on_xcoord_range


        #Get list of files
        files, roots, dates, file_descriptions = get_files_and_info(path)
        n_files_originally = len(files)

        if n_files_originally == 0:
            raise ValueError('No files imported')

        #Filter files
        keep_files = np.array([True]*len(files))

        if self.filter_on_min_date is not None:
            keep_files *= filter_on_date(dates, self.filter_on_min_date, 'min')

        if self.filter_on_max_date is not None:
            keep_files *= filter_on_date(dates, self.filter_on_max_date, 'max')



        file_paths = files[keep_files]

        #Do train/val split
        if split is not None:
            if split >0:
                print('Using', int(len(file_paths)*split), 'first files of',len(file_paths),'files')
                file_paths = file_paths[0:int(len(file_paths)*split)]
            else:
                print('Using', int(-len(file_paths) * split), 'last files of', len(file_paths), 'files')
                file_paths = file_paths[int(len(file_paths) * split):]

        #Load DataFile objects
        self.files = []
        for file_path, root in zip(file_paths, roots):
            self.files.append(Tile(root, win=win_size_for_calc_overlapping_patches))

            #only load maximum limit_n_tiles
            if limit_n_tiles == len(self.files):
                break

        if self.filter_on_ycoord_range is not None:
            raise NotImplementedError()
            # Todo
            # keep_files *= ycoords >= self.filter_on_ycoord_range[0]
            # keep_files *= ycoords <= self.filter_on_ycoord_range[1]

        if self.filter_on_xcoord_range is not None:
            raise NotImplementedError()
            # Todo
            # keep_files *= xcoords >= self.filter_on_xcoord_range[0]
            # keep_files *= xcoords <= self.filter_on_xcoord_range[1]

        #Print to user
        print(' -',len(self.files),'were loaded.' )
        #print(' - Bands in first file:',self.files[0].get_meta_data()['bands'])
        if np.sum(np.logical_not(keep_files)):
            print(' -',np.sum(np.logical_not(keep_files)),'files did not match filters.')
        print('')

        self.count()

    def count(self):
        #Measure length of sentinel_dataset
        self.N =  0
        self.N_overlapping = 0
        for f in self.files:
            self.N += len(f)
            self.N_overlapping += len(f.labelled_pixels)
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

    def _getitem(self, item, non_overlapping):


        #Find file_no and coordinate_no for item
        N = 0
        labelled_coord_no = 0
        selected_file_id = 0
        for file_no in range(len(self.files)):
            file = self.files[file_no]

            file_length = len(file) if non_overlapping else len(file.labelled_pixels)
            labelled_coord_no = item - N

            if labelled_coord_no < file_length:
                selected_file_id = file_no
                break
            else:
                N += file_length

        #Find the given tile id and other files covering the same tile
        tile_id = self.files[selected_file_id].tile_id
        files_for_tile = [f for f in self.files if f.tile_id == tile_id]
        if len(self.files) > self.n_time_instances:
            # If we do this for validation/testing we want the same tiles everytime
            if not self.random_time_instances :
                random.seed(0)
            files_for_tile = random.sample(files_for_tile, self.n_time_instances)
        else:
            files_for_tile = (files_for_tile*self.n_time_instances)[0:self.n_time_instances]

        #Get data from the tiles
        data = []
        lbl = []
        cld = []
        datafile_objects = []

        for f in files_for_tile :
            data += f.get_data(self.band_identifiers)
            lbl += f.get_labels(self.label_identifiers)
            cld += f.get_cloud_mask()
            datafile_objects.append(f)

        #Get coordinate of center pixel
        if non_overlapping:
            coord = self.files[selected_file_id].get_non_overlapping_coordinate_no(labelled_coord_no)
        else:
            coord = self.files[selected_file_id].get_overlapping_coordinate_no(labelled_coord_no)

        return {'data':data,
                'data_bands': self.band_identifiers,
                'target':lbl,
                'target_bands': self.label_identifiers,
                'cloud':cld,
                'name':tile_id+ ' y:' + str(coord[0]) + ' x:' + str(coord[1]),
                'coord':list(coord),
                'datafile_objects':datafile_objects,
                'n_time_instances':len(datafile_objects)}




