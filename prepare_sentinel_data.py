from sentinel_data_preparation.data_preparation import DataPreparation
import json
import os

def run_sentinel2():
    config = {
        "sensor" : "sentinel-2",
        # "eocloud_path_list" : "sentinel2_eocloud_filenames.txt",
        "eocloud_path_list" : "sentinel2_files_debug.txt",
        "rad_cor" : "toa",
        "tile_ids" : ["t37lcj", "t37lck"],
        "resolution" : "10m",
        "bands": ["b02", "b03", "b04", "b08",             #10m
                "b05", "b06", "b07", "b8a", "b11", "b12", #20m
                "b01", "b09", "b10"],                     #60m
        #
        "process_clouds" : "yes",
        "process_target_data" : "yes",
        #
        "target_dir" : "liwale_data",
        "target_basename" : "vegt_forestmask_forest",
        "target_names" : ['lbl_vegetation_height', 'lbl_fractional_forest_cover'],
        #
        "outdir" : os.path.join('.', 'tiles_all_bands')
    }
    with open('sentinel2_config.json', 'w') as outfile:
        json.dump(config, outfile)

    sentinel2_config = "sentinel2_config.json"
    s2_dp = DataPreparation(sentinel2_config)
    s2_dp.run_all()


def run_sentinel1():
    config = {
        "sensor" : "sentinel-1",
        "eocloud_path_list" : "sentinel1_files_debug.txt",
        "tile_ids" : ["t37lcj", "t37lck"],  #Sentinel-2 tiles, for multi-sensor processing
        "bands": ["VV", "VH"],
        #
        "dem_dir" : "/home/absalberg/nvgeo/geocode_sar/dem",
        "dem_basename" : "dem_10m_utm37s",
        #
        "tmp_dir" : "/home/absalberg/nvgeo/geocode_sar/tmp_geocoded_sar_imgs/tmp",
        "aux_dir" : "/home/absalberg/nvgeo/geocode_sar/",
        "geocode_config_file" : "/home/absalberg/nvgeo/geocode_sar/geocode.cfg",
        "source_dir" : "/home/absalberg/nvgeo/geocode_sar/",
        #
        "include_layover_mask" : "yes",
        "process_target_data" : "yes",
        #
        "target_dir" : "liwale_data",
        "target_basename" : "vegt_forestmask_forest",
        "target_names" : ['lbl_vegetation_height', 'lbl_fractional_forest_cover'],
        #
        "tmp_outdir" : "/home/absalberg/nvgeo/geocode_sar/tmp_geocoded_sar_imgs",
        "outdir" : os.path.join('.', 'tiles_all_bands')
    }

    with open('sentinel1_config.json', 'w') as outfile:
        json.dump(config, outfile)

    sentinel1_config = "sentinel1_config.json"
    s1_dp = DataPreparation(sentinel1_config)
    s1_dp.run_all()


if __name__ == "__main__":
    run_sentinel1()
    # run_sentinel2()
