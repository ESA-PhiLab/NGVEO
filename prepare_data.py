import json
import os
import shutil
from sentinel_dataset._utils import parse_eodata_folder_name
from sentinel_data_preparation.data_preparation import DataPreparation

#Input data are level-1
config = {
    "sensor": "sentinel-2",
    "eocloud_path_list": "example_eodata_l1.txt",
    "rad_cor": "toa",
    "resolution": "10m",
    "bands": ["b02", "b03", "b04", "b08",  # 10m
              "b05", "b06", "b07", "b8a", "b11", "b12",  # 20m
              "b01", "b09","b10"],  # 60m
    "process_clouds": "yes",
    "process_target_data": "no",
    "outdir": 'data/',
    'tile_ids':["T29SQB","T29SQC","T30STJ","T29TPE","T30SUJ","T32TMP","T32TNS","T32TNP","T32TPR","T32TML","T32TMK","T32TNT","T32UPU","T32TPT","T32UQU","T32UNU","T32TMN","T32TMM"]
}

s2_dp = DataPreparation(config)
s2_dp.run_all()


#For atmospheric coreection we wil use Level-2 data as labels
config = {
    "sensor": "sentinel-2",
    "eocloud_path_list": "example_eodata_l2.txt",
    "rad_cor": "toa",
    "resolution": "10m",
    "bands": ["b02", "b03", "b04", "b08",  # 10m
              "b05", "b06", "b07", "b8a", "b11", "b12",  # 20m
              "b01", "b09"],  # 60m
    "process_clouds": "no",
    "process_target_data": "no",
    "outdir": 'data_atm_corr/',
    'tile_ids':["T32TMP","T32TNS","T32TNP","T32TPR","T32TML","T32TMK","T32TNT","T32UPU","T32TPT","T32UQU","T32UNU","T32TMN","T32TMM"]
}

s2_dp = DataPreparation(config)
s2_dp.run_all()


#Move amtospheric corrected data into the data-folder to act as labels
def use_atm_corr_data_as_labels(atm_corr_dir, out_dir, delete_atm_corr_dir = False):

    for dir, dirnames, filenames in os.walk(atm_corr_dir):
        for filename in filenames:
            if 'data_b' in filename:
                src = os.path.join(dir, filename)

                atm_corr_properties = parse_eodata_folder_name(dir.split('/')[-1])
                date_atm_corr = str(atm_corr_properties['datetime']).replace('-','').replace(':','')

                #find corresponding output-folder
                for outdir, _, _ in os.walk(out_dir):
                    if date_atm_corr in outdir:
                        out_properties = parse_eodata_folder_name(outdir.split('/')[-1])
                        if atm_corr_properties['tile_id']==out_properties['tile_id']:
                            dest = os.path.join(outdir, filename.replace(atm_corr_dir, '')).replace('/data_b', '/lbl_b')  # Rename data as labels
                            print('Moving', src, 'to', dest)
                            shutil.move(src,dest)

        if delete_atm_corr_dir:
            shutil.rmtree(atm_corr_dir)

use_atm_corr_data_as_labels('data_atm_corr/','data/')


##

