from __future__ import division, print_function
import os
import numpy as np
import traceback

from sentinel_dataset.dataset import Dataset

class Dataset_multitime(Dataset):
    """ Collect DataFiles from same tile but with multiple time-stamps to create a sentinel_dataset"""
    rgb_bands = ['b04','b03','b02']

    def __len__(self):
        return 1

    def get_random_sample(self):
        item = np.random.randint(0,len(self.files[0].labelled_pixels))
        return self._getitem(item, non_overlapping=False)

    def _getitem(self, item, non_overlapping) :

        data = []
        labels = []
        cloud = []
        dates = []

        for file_no in range(len(self.files)):

            dta = self.files[file_no].get_data(self.band_identifiers)
            lbl = self.files[file_no].get_labels(self.label_identifiers)
            cld = self.files[file_no].get_cloud_mask()


            data += dta
            labels += lbl
            cloud += [cld]
            dates.append(self.files[file_no].datetime)

        meta = self.files[file_no].get_meta_data()
        meta.update({'datetimes':dates})
        _ , coord = super(Dataset_multitime, self)._getitem(item, non_overlapping)


        return {'data':data,'target':lbl, 'cld':cld, 'meta':meta, 'coord':coord}




