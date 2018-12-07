from __future__ import print_function

import json
import os
import numpy as np

from sentinel_dataset._utils import parse_eodata_folder_name
import rasterio

class Tile(object):
    """ Utilities for reading tile objects"""

    data_prefix = 'data_'
    key_prefix_label = 'lbl_'
    key_meta_data = 'meta_data'
    # key_cloud_mask = 'cloud_mask'


    def __init__(self, path, win=[512,512], key_missing_mask=None):
        """ Constructor """
        self.path = path
        self.name = self.path.strip('/').split('/')[-2]
        self.key_missing_mask = key_missing_mask


        print(' - DataFile: Opening', self.name, self.path)

        #Load meta data
        with np.load( os.path.join(self.path, 'meta_data.npz')) as f:
            try:
                self.meta_data = f['meta_data'][()]
            except:
                self.meta_data = f['metadata'][()]

        #Set the missing mask
        if self.meta_data['sensor'] == "Sentinel-1":
            self.key_missing_mask = "layover_mask"
        else:
            self.key_missing_mask = "cloud_mask"

        # Get shape
        self.shape = self.meta_data['shape']

        #Load list of labelled pixels
        if os.path.isfile(os.path.join(self.path, 'labelled_pixels.npz')):
            with  np.load(os.path.join(self.path, 'labelled_pixels.npz')) as f:
                self.labelled_pixels = np.concatenate( [ np.expand_dims(f['y'],-1), np.expand_dims(f['x'],-1)], -1)
                # self.labelled_pixels is matrix of shape [N,2]
            self.labelled_pixels = self._remove_edge_pixels(win, self.labelled_pixels)

            #Make list of non-overlapping labelled pixels
            if win == [0,0]:
                self.nolp = np.copy(self.labelled_pixels)
            else:
                self.nolp = self._make_list_of_non_overlapping_pixels(win)
        else:
            #When no list with labelled_pixels is provided we assume that all pixels are labelled
            
            #Make list of non-overlapping labelled pixels
            if win == [0,0]:
                self.nolp = None
            else:
                y = np.arange(win[0]//2, self.shape[0]-win[0]//2, win[0])
                x = np.arange(win[1]//2, self.shape[1]-win[1]//2, win[1] )
                y, x = np.meshgrid(x, y, indexing='ij')
                # import pdb; pdb.set_trace()
                # Select pixels with mask==true

                self.nolp = np.concatenate( [np.expand_dims(y.flatten(),1), np.expand_dims(x.flatten(),1)],1)

        #Read mapinfo-json file (this is used in export_predictions_to_tif)
        mapinfo_file = os.path.join(self.path, "mapinfo.json")
        if os.path.isfile(mapinfo_file):
            with open(mapinfo_file) as file:
                self.map_info = json.load(file)
        else:
            self.map_info = None
            print('Warning: mapinfo.json was not found. Writing predictions to tif will not be possible.')

        #Update meta-data
        self.meta_data.update({
            'path': self.path,
        })
        #Add information extracted from folder-name
        self.meta_data.update(
            parse_eodata_folder_name(self.path.strip('/').split('/')[-1]))

        #Add tile id
        self.tile_id = self.meta_data['tile_id']

        #Add filename
        self.file_name = self.meta_data["full_name"]

        #Check that bands in folder matches bands in meta_data
        bands_in_folder = [f.replace(self.data_prefix,'') for f in self._get_memmap_names() if 'data_' in f]
        assert(all([band_in_meta.lower() in bands_in_folder for band_in_meta in self.meta_data['bands']]))


    def _make_list_of_non_overlapping_pixels(self, win):
        "Get list of non-overlapping pixels (nolp) - either load, or make and save"
        file_path = os.path.join(self.path, 'nolp' + str(win))

        #If exist just load it
        if os.path.isfile(file_path + '.npy'):
            print(' -', 'Loading list of non-overlapping pixels')
            return np.load(file_path+ '.npy')

        #Otherwise we make it
        else:
            print('   ', 'Making list of non-overlapping pixels (this may take some time)')
            #Define window
            wl = np.array(win) // 2
            wr = np.array(win) - wl
            tmp_list = np.copy(self.labelled_pixels)

            #Loop through and add coordinates
            nolp = []
            while tmp_list.shape[0] > 0:
                coord = np.copy(tmp_list[0, :])
                nolp.append(coord)
                y, x = coord

                # Remove coordinates that overlap with this pixel
                overlapping_coords = np.greater(tmp_list[:, 0], y - wl[0]) * \
                                     np.less(tmp_list[:, 0], y + wr[0]) * \
                                     np.greater(tmp_list[:, 1], x - wl[1]) * \
                                     np.less(tmp_list[:, 1], x + wr[1])
                overlapping_coords = overlapping_coords.astype('bool')

                tmp_list = tmp_list[np.logical_not(overlapping_coords), :]

                print('    Samples left:', tmp_list.shape[0])

            nolp = np.array(nolp)
            np.save(file_path,nolp)
            return nolp


    def _remove_edge_pixels(self, win, coordinate_list):
        if win ==[0,0]:
            return coordinate_list

        #Get half window size
        wl = np.array(win) // 2
        wr = np.array(win) - wl

        #Check which pixels that are not close to edge of tile
        indexes_within_bounds = np.greater(coordinate_list[:, 0], wl[0]) * \
                             np.less(coordinate_list[:, 0], self.shape[0] - wr[0]) * \
                             np.greater(coordinate_list[:, 1], wl[1]) * \
                             np.less(coordinate_list[:, 1], self.shape[1] - wr[1])
        indexes_within_bounds = indexes_within_bounds.astype('bool')
        coordinate_list = coordinate_list[indexes_within_bounds, :]

        return coordinate_list


    def export_prediction_to_tif(self, out_file_path, prediction):
        """

        :param out_file_path:
        :param prediction: a np-array where second dim indicate n-channels
        :return:
        """

        #Check if map-info is available
        if self.map_info is None:
            class MissingMapInfoException(Exception): pass
            raise MissingMapInfoException()

        #Compute geo-meta data
        geo = self.map_info['transform']
        transf = rasterio.Affine(geo[1], geo[2], geo[0], geo[4], geo[5], geo[3])
        crs = {'init': self.map_info['cs_code']}

        #Write to file
        with rasterio.open(out_file_path, "w", driver="GTiff", compress="lzw", bigtiff="YES",
                           height=prediction.shape[0], width=prediction.shape[1], count=prediction.shape[2],
                           dtype=prediction.dtype,
                           crs=crs, transform=transf) as out_file:

            for band_no in range(0, prediction.shape[2]):
                out_file.write(prediction[:,:,band_no], band_no + 1)
        print('Exported predictions to', out_file_path)

    def get_meta_data(self):
        return self.meta_data


    def get_data(self, band_identifiers=None):
        """ Reads sentinel_dataset from disk and return as a numpy array with dimensions H x W x C where C is number of channels.
            This corresponds to the bands specified in band_identifiers."""
        if band_identifiers == None:
            band_identifiers = self.get_meta_data()['bands']
        if type(band_identifiers)!= list:
            band_identifiers = [band_identifiers]
        return [self._open_memmap(self.data_prefix + b.lower()) for b in band_identifiers]


    def get_labels(self, label_identifiers=None):
        """ Reads labels from disk and return as a numpy array with dimensions H x W x C where C is number of channels.
            This corresponds to the bands specified in band_identifiers."""
        if type(label_identifiers)!= list:
            label_identifiers = [label_identifiers]

        if label_identifiers is None:
            label_identifiers = [l.replace(self.key_prefix_label,'') for l in self._get_memmap_names() if self.key_prefix_label in l]

        label_identifiers = [self.key_prefix_label + bi.lower() for bi in label_identifiers]
        #Hack to make 'cloud_mask' be accsessible as a label
        label_identifiers = [li.replace(self.key_prefix_label+self.key_missing_mask, self.key_missing_mask) for li in label_identifiers]
        return  [self._open_memmap(b.lower()) for b in label_identifiers]


    def get_missing_mask(self):
        """ Reads and returns cloud mask """
        return [np.expand_dims(self._open_memmap(self.key_missing_mask),-1)]


    def _get_memmap_names(self):
        """ Returns list of .dat files in folder. This correspons to the data bands and label types."""
        return [f.replace('.dat','') for f in os.listdir(self.path) if '.dat' in f]


    def _open_memmap(self,filename):
        fp = np.memmap(os.path.join(self.path, filename+'.dat'), dtype='float32', mode='r', shape=tuple(self.shape))
        return fp


    def get_overlapping_coordinate_no(self,no):
        if hasattr(self, 'labelled_pixels'):
            return self.labelled_pixels[no,:]
        else:
            x = no % self.shape[1]
            y = int((no - x) / self.shape[0])
            return np.array([y,x])


    def get_non_overlapping_coordinate_no(self,no):
        return self.nolp[no, :]


    def n_non_overlapping(self):
        return self.nolp.shape[0]

    def n_overlapping(self):
        if hasattr(self,'labelled_pixels'):
            return self.labelled_pixels.shape[0]
        else:
            #Assuming all pixels are labelled
            return np.prod(self.shape)


    def __len__(self):
        #todo: Remove this if not needed. We only keep in case it is used somewhere
        return self.nolp.shape[0]

