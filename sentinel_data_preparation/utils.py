import os
import numpy as np
import fnmatch

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
        if os.path.isdir(os.path.join(a_dir, name))]

def save_np_memmap(filename, data, dtype):
    f = np.memmap(filename +'.dat', mode='w+', dtype=dtype, shape=data.shape)
    f[:] = data.astype(dtype)
    del f

def save_list_of_labelled_pixels(mask, out_path):
# Make image with coordinates
    y = np.linspace(0, mask.shape[0], num=mask.shape[0], dtype='uint16')
    x = np.linspace(0, mask.shape[1], num=mask.shape[1], dtype='uint16')
    y, x = np.meshgrid(x, y, indexing='ij')
    #import pdb; pdb.set_trace()
    # Select pixels with mask==true
    y = y[mask.astype('bool')]
    x = x[mask.astype('bool')]

    #Save
    np.savez(os.path.join(out_path, 'labelled_pixels.npz'), y=y, x=x)


def get_tile_id(input_file, tile_ids):
    for tile_id in tile_ids:
        if fnmatch.fnmatch(input_file.upper(), "*" + tile_id.upper() + "*"):
            return tile_id

def find_file(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result