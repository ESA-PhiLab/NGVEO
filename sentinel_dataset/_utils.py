from __future__ import division, print_function

import io
import os
import sys

import numpy as np
import os.path

def get_files_and_info(paths):
    """
    Get filenames, root and meta data stored in filename
    :param path: Path to parent directory
    :param use_content_file: If True, then we will look for a file named content.txt instead of running a costly os.walk. If no such file exists, it wil be generated for the next time.
    :return:
    """
    if type(paths) != list:
        paths = [paths]

    roots = []
    folders = []

    for path in paths:

        if path[-1]!= '/':
            path += '/';


        for subdir, dirs, file_list in os.walk(path):
            if 'meta_data.npz' in file_list:
                if subdir[-1] != '/':
                    subdir += '/'
                roots.append(os.path.join(subdir))
                folders.append(subdir.split('/')[-2])


    #Parse filenames
    file_description = []
    dates = []

    for root, folder in zip(roots,folders):
        parsed_folder_name = parse_eodata_folder_name(folder)

        file_description.append(parsed_folder_name)
        dates.append(parsed_folder_name['datetime'])


    return np.array(folders), np.array(roots), np.array(dates),  np.array(file_description)

def parse_eodata_folder_name(name):
    #Example of name:
    # S2A_MSIL1C_20170526T074241_N0205_R049_T37LCJ_20170526T074901

    # Remove extension
    name = name.split('.')[0]

    #Add full name
    out = {'full_name':name}

    #Split name into different parts
    name_parts = name.split('_')

    #Figure out which sentinel (1, 2, or 3)
    out['sentinel_type'] = int(name_parts[0][1])

    if out['sentinel_type'] == 2:
        """
        Sentinel 2: (after  6th of December, 2016)

        MMM_MSIL1C_YYYYMMDDHHMMSS_Nxxyy_ROOO_Txxxxx_<Product Discriminator>.SAFE

        The products contain two dates.

        The first date (YYYYMMDDHHMMSS) is the datatake sensing time.
        The second date is the "<Product Discriminator>" field, which is 15 characters in length, and is used to distinguish between different end user products from the same datatake. Depending on the instance, the time in this field can be earlier or slightly later than the datatake sensing time.

        The other components of the filename are:

        MMM: is the mission ID(S2A/S2B)
        MSIL1C: denotes the Level-1C product level
        YYYYMMDDHHMMSS: the datatake sensing start time
        Nxxyy: the Processing Baseline number (e.g. N0204)
        ROOO: Relative Orbit number (R001 - R143)
        Txxxxx: Tile Number field
        SAFE: Product Format (Standard Archive Format for Europe)"""

        out['product_discriminator'] = _sentinel_datetime_2_np_datetime(name_parts[6])

        #We only support the new format
        #TODO: add support for older sentinel 1 name formats
        if not out['product_discriminator']>np.datetime64('2016-12-06T00:00:00'):
            raise NotImplementedError('parse_eodata_folder_name() does not support sentinel-2 data earlier than 6th of December 2016')

        out['misson_id'] = name_parts[0]
        out['product_level'] = name_parts[1][3:]
        out['datetime'] = _sentinel_datetime_2_np_datetime(name_parts[2])
        out['processing_baseline_number'] = int(name_parts[3][1:])
        out['relative_orbit_number'] = int(name_parts[4][1:])
        out['tile_id'] = name_parts[5]

    elif out['sentinel_type'] == 1:
        """ https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/naming-conventions """
        out['misson_id'] = name_parts[0]

        out['mode'] = name_parts[1]

        out['product_type'] = name_parts[2][0:3]
        out['resolution_class'] = name_parts[2][-1]

        out['processing_level'] = int(name_parts[3][0])
        out['product_class'] = name_parts[3][1]
        out['polarization'] = name_parts[3][2:]

        out['datetime'] = _sentinel_datetime_2_np_datetime(name_parts[4])
        out['start_date'] = _sentinel_datetime_2_np_datetime(name_parts[4])
        out['end_date'] = _sentinel_datetime_2_np_datetime(name_parts[5])

        out['absolute_orbit_number'] = int(name_parts[6][1:])
        out['mission_data_take_id'] = name_parts[7]
        out['product_unique_id'] = name_parts[8]

    elif out['sentinel_type'] == 3:
        # TODO: add support for sentinel 3 name formats
        raise NotImplementedError('parse_eodata_folder_name() does not support sentinel-3 yet')

    return out

def _sentinel_datetime_2_np_datetime(sentinel_datetime_string):
    date, time = sentinel_datetime_string.split('T')
    year = date[0:4]
    month = date[4:6]
    day = date[6:8]
    hour = time[0:2]
    min = time[2:4]
    sec = time[4:6]

    np_datetime_str = year + '-' + month + '-' + day + 'T' + hour + ':' + min + ':' + sec
    return np.datetime64(np_datetime_str)



