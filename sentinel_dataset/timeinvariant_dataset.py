from sentinel_dataset.dataset import Dataset
import random

class TimeinvDataset(Dataset):
    n_time_instances = 4

    def _getitem(self, item, non_overlapping):

        #Find file_no and coordinate_no for item
        N = 0
        labelled_coord_no = 0
        for file_no in range(len(self.files)):
            file = self.files[file_no]

            file_length = len(file) if non_overlapping else len(file.labelled_pixels)
            labelled_coord_no = item - N

            if labelled_coord_no < file_length:
                break
            else:
                N += file_length

        tile_id = self.files[file_no].tile
        files_for_tile = [f for f in self.files if f.tile == tile_id]
        if len(self.files)>self.n_time_instances:
            files_for_tile = random.sample(files_for_tile, self.n_time_instances)
        else:
            files_for_tile = (files_for_tile*self.n_time_instances)[0:self.n_time_instances]

        data = []
        for f in files_for_tile :

            #Get data
            d = f.get_data(self.band_identifiers)
            data+=d
            lbl = f.get_labels(self.label_identifiers)
            cld = self.files[file_no].get_cloud_mask()
            meta = self.files[file_no].get_meta_data()

        #Get coordinate
        if non_overlapping:
            coord = self.files[file_no].get_non_overlapping_coordinate_no(labelled_coord_no)
        else:
            coord = self.files[file_no].get_overlapping_coordinate_no(labelled_coord_no)
        #meta['name'] += str(coord)
        #Todo: fix this

        return [data, lbl, cld, meta], coord

