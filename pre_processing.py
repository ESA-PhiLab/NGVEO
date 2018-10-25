import numpy as np
import scipy.ndimage
import skimage.measure


""" Example of how to add indexes: """
def GARI(input_dict):
    # http://www.harrisgeospatial.com/docs/BroadbandGreenness.html#Green2
    band_list = [b.lower() for b in input_dict['data_bands']]
    data = input_dict['data']
    NIR = data[:,:, band_list.index('b08')]
    red = data[:, :, band_list.index('b04')]
    green = data[:, :, band_list.index('b03')]
    blue = data[:, :, band_list.index('b02')]
    gamma = 1.7  # Gitelson, Kaufman, and Merzylak (1996, page 296).

    gari =  (NIR - ( green - gamma*(blue - red))) / \
            (NIR + ( green - gamma*(blue - red)))

    gari = np.expand_dims(gari,-1)
    input_dict['data'] =  np.concatenate([data,gari],-1)
    input_dict['data_bands'].append('index_GARI')
    return input_dict

def NDVI(input_dict):
    band_list = [b.lower() for b in input_dict['data_bands']]
    data = input_dict['data']
    b08 = data[:,:, band_list.index('b08')]
    b04 = data[:, :, band_list.index('b04')]

    ndvi =  (b08 - b04) / (b08 + b04)

    ndvi = np.expand_dims(ndvi,-1)
    input_dict['data'] =  np.concatenate([data,ndvi],-1)
    input_dict['data_bands'].append('index_NDVI')
    return input_dict


def SLAVI(input_dict ):
    data = input_dict['data']
    band_list = [b.lower() for b in input_dict['data_bands']]
    b08 = data[:,:, band_list.index('b08')]
    b04 = data[:, :, band_list.index('b04')]
    b12 = data[:, :, band_list.index('b12')]

    slavi =  (b08) / (b04+b12)

    slavi = np.expand_dims(slavi,-1)
    input_dict['data'] = np.concatenate([data, slavi], -1)
    input_dict['data_bands'].append('index_SLAVI')
    return input_dict


def check_for_nans(input_dict):
    """ A utility function to check if input has nans. In such case a warning is printed."""
    if np.any(np.isnan(input_dict['target'])):
        print('NaN found in target for', input_dict['datafile_objects'][0].path )

    if np.any(np.isnan(input_dict['cloud'])):
        print('NaN found in cloud for', input_dict['datafile_objects'][0].path )

    if np.any(np.isnan(input_dict['data'])):
        print('NaN found in data for', input_dict['datafile_objects'][0].path )

    return input_dict



def merge_multitime_labels(input_dict):
    """ Combine labels in the case of multi-temporal data"""
    lbl = input_dict['target']

    #Loop through the different target-channels and sort
    out = []
    for i in range(len(input_dict['target_bands'])):
        out.append(lbl[:,:,i::len(input_dict['target_bands'])])

    #Take mask for each target type
    lbl = [np.max(o,axis=-1,keepdims=True) for o in out]
    lbl = np.concatenate(lbl, axis=2)

    input_dict['target'] = lbl
    return input_dict

def median_filter_lbls(input_dict):
    """ Apply median filter to labels"""
    lbl = input_dict['target']
    lbl[lbl==-100] = np.nan

    lbl = scipy.ndimage.median_filter(lbl,[5,5,1])

    lbl[np.isnan(lbl)]=-100

    input_dict['target'] = lbl
    return input_dict


def mask_cloud_in_label(input_dict):
    """ Insert ignore value where we have clouds"""
    lbl = input_dict['target']
    cld = input_dict['cloud']

    #Do multitime (min operation)
    cld = np.min(cld,axis=-1,keepdims=True)

    #Loop through label-channels
    for i in range(lbl.shape[-1]):
        l = lbl[:,:,i]

        #Change -1 with pytorch ignore val
        l[l==-100] = -100

        #Add ignore vals for clouds
        l[cld[:,:,0]==1] = -100

        lbl[:, :, i] = l

    input_dict['target'] = lbl
    return input_dict

def ignore_missing_labels(input_dict):
    """ Ignore value in input data was set to -1"""
    #Todo: use nan instead of -1 to signal wether there are missing label
    input_dict['target'][input_dict['target']==-1] = -100
    return input_dict


class VegetationHeight_to_Classes():
    def __init__(self, class_defs):
        self.class_defs = class_defs

    def __call__(self, input_dict):
        lbl = input_dict['target']
        new_lbl = (lbl*0 - 100).astype('int64') #Make new array with ignore labels
        for class_no, [from_,to_] in enumerate(self.class_defs):
            new_lbl[ (from_ <= lbl) & (lbl < to_)] = class_no
        input_dict['target'] = new_lbl.astype('int64')
        return input_dict

class PoolImages():
    def __init__(self, downsampling_factor):
        self.n_elements = downsampling_factor

    def __call__(self,input_dict):
        n_elements = self.n_elements
        data = input_dict['data']
        lbl = input_dict['target']
        cld = input_dict['cloud']

        input_dict['data'] = skimage.measure.block_reduce(data, (n_elements,n_elements,1), np.mean)
        input_dict['cloud'] = skimage.measure.block_reduce(cld, (n_elements, n_elements, 1), np.max)
        target_min = skimage.measure.block_reduce(lbl, (n_elements, n_elements,1), np.min)
        target_mean = skimage.measure.block_reduce(lbl, (n_elements, n_elements,1), np.mean)
        target_mean[target_min < 0] = target_min[target_min < 0]
        input_dict['target'] = target_mean

        #"coord" og "name" not updated since these are related to the original image

        return input_dict



class Compose( ):
    """ Combine a list of preprocessing steps"""
    def __init__(self, funcs):
        self.funcs  = funcs

    def __call__(self, input_dict):
        for f in self.funcs:
            input_dict= f(input_dict)
        return input_dict

