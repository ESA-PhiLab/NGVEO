import json
from sentinel_data_preparation.target_processing import TargetProcessing
import sentinel_data_preparation.sentinel2_processing
import sentinel_data_preparation.sentinel1_processing

class DataPreparation():
    def __init__(self, config_file):
        if type(config_file) == dict:
            self.params = config_file
        else:
            self.params = self.read_config_file(config_file)
        self.sensor = self.params['sensor']
        self.eocloud_path_list = self.params['eocloud_path_list']

        if self.sensor == "sentinel-2":
            SensorProcessing = sentinel_data_preparation.sentinel2_processing.SensorProcessing
        if self.sensor == "sentinel-1":
            SensorProcessing = sentinel_data_preparation.sentinel1_processing.SensorProcessing

        #import pdb; pdb.set_trace()
        
        self.sen_pro = SensorProcessing(self.params)

        self.target_pro = TargetProcessing(self.params)

    def read_config_file(self, config_file):
        with open(config_file) as file:
            params = json.load(file)
        # from pprint import  pprint
        # pprint(params)
        return params

    def run_target_processing(self, tile_id=None):
        target_pro = TargetProcessing(self.params)

        #Process all files in the EOCloud list
        for filename in open(self.eocloud_path_list, 'r'):
            input_file = filename[0:-1]
            target_pro.process_target_data(input_file, tile_id)


    def display_targets(self, tile_id=None):
        # Process all files in the EOCloud list
        for filename in open(self.eocloud_path_list, 'r'):
            input_file = filename[0:-1]
            self.target_pro.display_image(input_file, tile_id)


    def display_images(self):
        for filename in open(self.eocloud_path_list, 'r'):
            input_file = filename[0:-1]

            if self.sensor == "sentinel-1":
                for tile_id in self.params['tile_ids']:
                    self.sen_pro.display_image(input_file, tile_id)

            if self.sensor == "sentinel-2":
                self.sen_pro.display_image(input_file) #TODO: Implement display_image funcion for Sentinel-2


    def run_all(self):
        #Process all files in the EOCloud list
        for filename in open(self.eocloud_path_list, 'r'):
            input_file = filename.strip('\n')

            if self.sensor == "sentinel-2":
                self.sen_pro.process_data(input_file)

                if self.params.get('process_clouds', "no") == "yes":
                    self.sen_pro.process_clouds(input_file)

                if self.params.get('process_target_data', "no") == "yes":
                    self.target_pro.process_target_data(input_file)


            if self.sensor == "sentinel-1":
                for tile_id in self.params['tile_ids']:
                    ret_val = self.sen_pro.process_data(input_file, tile_id)
                    if ret_val == 0: # 0 errors
                        if self.params.get('include_layover_mask', "no") == "yes" and ret_val==0:
                            self.sen_pro.get_layover_shadow_mask()

                        if self.params.get('process_target_data', "no") == "yes":
                            self.target_pro.process_target_data(input_file, tile_id)

