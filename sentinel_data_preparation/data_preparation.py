import json
from sentinel_data_preparation.target_processing import TargetProcessing
import sentinel_data_preparation.sentinel2_processing
import sentinel_data_preparation.sentinel1_processing

class DataPreparation():
    def __init__(self, config_file):
        self.params = self.read_config_file(config_file)
        self.sensor = self.params['sensor']
        self.eocloud_path_list = self.params['eocloud_path_list']

    def read_config_file(self, config_file):
        with open(config_file) as file:
            params = json.load(file)
        # from pprint import  pprint
        # pprint(params)
        return params

    def run_all(self):
        if self.sensor == "sentinel-2":
            SensorProcessing = sentinel_data_preparation.sentinel2_processing.SensorProcessing
        if self.sensor == "sentinel-1":
            SensorProcessing = sentinel_data_preparation.sentinel1_processing.SensorProcessing

        sen_pro = SensorProcessing(self.params)

        #Process all files in the EOCloud list
        for filename in open(self.eocloud_path_list, 'r'):
            input_file = filename[0:-1]
            mask = None

            if self.sensor == "sentinel-2":
                tile_id = sen_pro.process_data(input_file)

                if self.params.get('process_clouds', "no") == "yes":
                    mask = sen_pro.process_clouds(input_file)

                if self.params.get('process_target_data', "no") == "yes":
                    target_pro = TargetProcessing(self.params)
                    target_pro.process_target_data(input_file, tile_id, mask)


            if self.sensor == "sentinel-1":
                for tile_id in self.params['tile_ids']:
                    ret_val = sen_pro.process_data(input_file, tile_id)

                    if self.params.get('include_layover_mask', "no") == "yes" and ret_val==0:
                        mask = sen_pro.get_layover_shadow_mask()

                    if self.params.get('process_target_data', "no") == "yes" and ret_val==0:
                        target_pro = TargetProcessing(self.params)
                        target_pro.process_target_data(input_file, tile_id, mask)
                    import pdb; pdb.set_trace()



