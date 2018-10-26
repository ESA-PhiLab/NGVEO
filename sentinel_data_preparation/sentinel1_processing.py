import os
import shutil
from sentinel_data_preparation import utils
import rasterio
import numpy as np

class SensorProcessing():
    def __init__(self, params):
        self.params = params
        self.ignore_value = -1
        self.layover_mask = None

    def process_data(self, input_dir, tile_id):

        #Clean tmp output dir
        tmp_output_dir = self.params['tmp_outdir']
        shutil.rmtree(tmp_output_dir, ignore_errors=True)
        if not os.path.exists(tmp_output_dir):
            os.makedirs(tmp_output_dir)


        # for tile_id in self.params['tile_ids']:
        dem_file = os.path.join(self.params['dem_dir'], tile_id.lower() + "_" + self.params['dem_basename'] + ".tif")

        print("NR_GeocodeSAR " + "SENTINEL-1 " + input_dir + " " + self.params['tmp_outdir'] +
              " " + self.params['tmp_dir'] + " " + self.params['aux_dir'] + " -dem " + dem_file + " -config "
                + self.params['geocode_config_file'])
        retcode = os.system(self.params['source_dir'] + "/" + "NR_GeocodeSAR " +"SENTINEL-1 " + input_dir + " "
                + self.params['tmp_outdir'] + " " + self.params['tmp_dir'] + " " + self.params['aux_dir']
                            + " -dem " + dem_file + " -config " + self.params['geocode_config_file'])

        # Read the geocoded Sentinel-1 data
        sigma0_filename = utils.find_file("*SIGMA0-dB.tif", tmp_output_dir)
        if len(sigma0_filename) > 0:  # files exists
            with rasterio.open(sigma0_filename[0]) as sar_file:
                data = sar_file.read()
                transform = sar_file.transform
            sensing_time = os.path.basename(sigma0_filename[0])[4:12]

            # Read the geocoded Sentinel-1 data
            layover_filename = utils.find_file("*LAYOVERSHADOW.tif", tmp_output_dir)[0]
            with rasterio.open(layover_filename) as layover_file:
                self.layover_mask = layover_file.read()

            metadata = {'UTM-coordinate': transform * (0, 0),
                         'path_to_tile_in_eodata': input_dir,
                         'resolution': 10,
                         'shape': data[0].shape,
                         'no_data_value': -1,
                         'date': sensing_time,
                         'sensor': 'Sentinel-1',
                         'bands': ['VV', 'VH']
                         }

            data_vv = data[0]
            data_vh = data[1]

            # Saving
            basename = os.path.splitext(os.path.basename(input_dir))[0]
            tiles_dir = os.path.join(self.params['outdir'], tile_id, basename)

            if not os.path.exists(tiles_dir):
                os.makedirs(tiles_dir)

            # Save meta data
            np.savez(os.path.join(tiles_dir, 'metadata.npz'), meta_data=metadata)

            # Save memory_maps
            utils.save_np_memmap(os.path.join(tiles_dir, 'data_vv'), data_vv, 'float32')
            utils.save_np_memmap(os.path.join(tiles_dir, 'data_vh'), data_vh, 'float32')

            utils.save_np_memmap(os.path.join(tiles_dir, 'layover_mask'), self.layover_mask, 'float32')

            return 0
        else:
            return -1

    def get_layover_shadow_mask(self):
        return self.layover_mask[0]