import numpy as np
import fnmatch
import os
from sentinel_data_preparation import utils
from xml.etree import ElementTree
import rasterio
from rasterio.warp import reproject, Resampling
from s2cloudless import S2PixelCloudDetector


class SensorProcessing():
    def __init__(self, params):
        self.params = params
        self.ignore_value = -1

    def read_sentinel2_metadata(self, xml_file, geotiff=None, resolution=10):
        metadata = ElementTree.parse(xml_file)
        root = metadata.getroot()
        for child in root:
            print(child.tag, child.attrib)
            iter_ = metadata.getiterator()
            for elem in iter_:
                if elem.tag == "SENSING_TIME":
                    sensing_time = elem.text
                if elem.tag == "HORIZONTAL_CS_NAME":
                    cs_name = elem.text
                    geog_cs = cs_name[0:5]
                    utm_zone = cs_name[cs_name.find("zone") + 5::]
                    zone_text = cs_name[cs_name.find("zone") + 5::]
                    if zone_text.find("N"):
                        zone = "N"
                        zone_number = int(zone_text[0:(zone_text.find("N"))])
                    else:
                        zone = "S"
                        zone_number = int(zone_text[0:(zone_text.find("S"))])
                if elem.tag == "HORIZONTAL_CS_CODE":
                    cs_code = elem.text
                if elem.tag == "Size":
                    if int(elem.attrib['resolution']) == resolution:
                        siz = elem.getchildren()
                        for size_elements in siz:
                            if size_elements.tag == "NROWS":
                                nrows = int(size_elements.text)
                            if size_elements.tag == "NCOLS":
                                ncols = int(size_elements.text)
                if elem.tag == "Geoposition":
                    if int(elem.attrib['resolution']) == resolution:
                        geopos = elem.getchildren()
                        for geopos_elements in geopos:
                            if geopos_elements.tag == "ULX":
                                ulx = float(geopos_elements.text)
                            if geopos_elements.tag == "ULY":
                                uly = float(geopos_elements.text)
                            if geopos_elements.tag == "XDIM":
                                xdim = float(geopos_elements.text)
                            if geopos_elements.tag == "YDIM":
                                ydim = float(geopos_elements.text)
                        geo_transform = [ulx, xdim, 0, uly, 0, ydim]
        print("https://stackoverflow.com/questions/39059122/how-to-ensure-data-is-written-to-geotiff-using-gdal")

        metadata = {'transform': geo_transform, 'cs': geog_cs,
                    'zone': zone, 'zone_number': zone_number,
                    'nrows': nrows, 'ncols': ncols,
                    'sensing_time': sensing_time}
        return metadata

    # Read Sentinel-2 data
    def read_sentinel2_data(self, input_file, params, save_img=False):
        input_dir = os.path.join(input_file, "GRANULE")
        sub_directories = utils.get_immediate_subdirectories(input_dir)
        image_dir = os.path.join(input_dir, sub_directories[0], "IMG_DATA")

        input_bands = params['bands']
        # input_bands = ['B02', 'B03', 'B04', 'B08',  # 10m
        #                'B05', 'B06', 'B07', 'B8A', 'B11', 'B12',  # 20m
        #                'B01', 'B09', 'B10']  # 60m

        num_bands = len(input_bands)
        scale_factor = 10000.0 #read from metadata?

        for img_dir, dirnames, filenames in os.walk(image_dir):
            for band_ind, band_product in enumerate(input_bands):
                band_product = band_product.upper()
                if fnmatch.filter(filenames, "*" + band_product + ".jp2"):
                    img_filename = os.path.join(image_dir, fnmatch.filter(filenames, "*" + band_product + ".jp2")[0])
                    with rasterio.open(img_filename) as ds:
                        img = ds.read()
                        if band_ind == 0:  # First band need to be 10m
                            tmparr = np.empty_like(img)
                            ds10 = ds
                            aff10 = ds.transform
                            img_stack = np.zeros((img.shape[1], img.shape[2], num_bands))
                            img_stack[:, :, band_ind] = img.squeeze() / scale_factor
                        elif input_bands[band_ind] == "B03" or input_bands[band_ind] == "B04" or input_bands[
                            band_ind] == "B08":  # 10m
                            img_stack[:, :, band_ind] = img.squeeze() / scale_factor
                        else:
                            reproject(img, tmparr,
                                      src_transform=ds.transform,
                                      dst_transform=aff10,
                                      src_crs=ds.crs,
                                      dst_crs=ds.crs,
                                      resampling=Resampling.cubic_spline)
                            img_stack[:, :, band_ind] = tmparr.squeeze() / scale_factor

        if save_img:
            with rasterio.open("image_stack.tif", "w", driver="GTiff", compress="lzw", bigtiff="YES",
                               height=img_stack.shape[0], width=img_stack.shape[1], count=num_bands,
                               dtype=img_stack.dtype,
                               crs=ds.crs, transform=aff10) as out_file:

                for band_no in range(0, num_bands):
                    out_file.write(img_stack[:, :, band_no], band_no + 1)

        return (img_stack, ds10)

    # Check if all Sentinel-2 bands are available
    def is_all_bands(self, input_file):
        input_dir = os.path.join(input_file, "GRANULE");
        sub_directories = utils.get_immediate_subdirectories(input_dir)
        image_dir = os.path.join(input_dir, sub_directories[0], "IMG_DATA")

        input_bands = ['B01', 'B02', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
        band_present = 1

        for img_dir, dirnames, filenames in os.walk(image_dir):
            for band_ind, band_product in enumerate(input_bands):
                if fnmatch.filter(filenames, "*" + band_product + ".jp2"):
                    band_present *= 1
                else:
                    band_present *= 0
        return band_present

    def cloud_detection(self, input_file, save_masks=False):
        input_dir = os.path.join(input_file, "GRANULE")
        sub_directories = utils.get_immediate_subdirectories(input_dir)
        image_dir = os.path.join(input_dir, sub_directories[0], "IMG_DATA")

        input_bands = ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']  # Band order is strict
        # num_bands = len(input_bands)
        scale_factor = 10000.0 #Read from metadata ?
        for img_dir, dirnames, filenames in os.walk(image_dir):
            for band_ind, band_product in enumerate(input_bands):
                band_product = band_product.upper()
                if fnmatch.filter(filenames, "*" + band_product + ".jp2"):
                    img_filename = os.path.join(image_dir, fnmatch.filter(filenames, "*" + band_product + ".jp2")[0])
                    with rasterio.open(img_filename) as ds:
                        img = ds.read()
                        if band_ind == 0:  # First band need to be 60m
                            tmparr = np.empty_like(img)
                            aff60 = ds.transform
                            img_stack = np.zeros((img.shape[0], img.shape[1], img.shape[2], len(input_bands)))
                            img_stack[:, :, :, band_ind] = img / scale_factor
                        elif input_bands[band_ind] == "B09" or input_bands[band_ind] == "B10":  # 60m
                            img_stack[:, :, :, band_ind] = img / scale_factor
                        else:
                            reproject(img, tmparr,
                                      src_transform=ds.transform,
                                      dst_transform=aff60,
                                      src_crs=ds.crs,
                                      dst_crs=ds.crs,
                                      resampling=Resampling.bilinear)
                            img_stack[:, :, :, band_ind] = tmparr / scale_factor

                        if input_bands[band_ind] == "B02":  # 10m
                            aff10 = ds.transform
                            nrows10 = img.shape[1]
                            ncols10 = img.shape[2]
                            ds10 = ds

            cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2)
            cloud_probs = cloud_detector.get_cloud_probability_maps(img_stack)
            cloud_mask = cloud_detector.get_cloud_masks(img_stack).astype(rasterio.uint8)

            cloud_probs_10 = np.zeros((1, nrows10, ncols10))
            reproject(cloud_probs, cloud_probs_10,
                      src_transform=aff60,
                      dst_transform=aff10,
                      src_crs=ds.crs,
                      dst_crs=ds.crs,
                      resampling=Resampling.cubic_spline)

            cloud_mask_10 = np.zeros((1, nrows10, ncols10))
            reproject(cloud_mask, cloud_mask_10,
                      src_transform=aff60,
                      dst_transform=aff10,
                      src_crs=ds.crs,
                      dst_crs=ds.crs,
                      resampling=Resampling.nearest)

            if save_masks:
                with rasterio.open("cloud_probs.tif", "w", driver="GTiff", compress="Lzw",
                                   height=nrows10, width=ncols10, count=1, dtype=cloud_probs_10.dtype,
                                   crs=ds.crs, transform=aff10) as out_file:
                    out_file.write(cloud_probs_10)
                with rasterio.open("cloud_mask.tif", "w", driver="GTiff", compress="Lzw",
                                   height=nrows10, width=ncols10, count=1, dtype=cloud_mask_10.dtype,
                                   crs=ds.crs, transform=aff10) as out_file:
                    out_file.write(cloud_mask_10)

        return (cloud_probs_10, cloud_mask_10, ds10)


    def process_data(self, input_file):

        #Get the tile id
        tile_id = utils.get_tile_id(input_file, self.params['tile_ids'])
        if tile_id == None:
            print("No Sentinel-2 data with acquired tile ids in the EoCloud path list.")
            return 1

        # Read the S2 xml file to get the sensing time
        input_dir = os.path.join(input_file, "GRANULE");
        sub_directories = utils.get_immediate_subdirectories(input_dir)
        for in_dir, dirnames, filenames in os.walk(os.path.join(input_dir, sub_directories[0])):
            if fnmatch.filter(filenames, "MTD*xml"):
                metadata_filename = os.path.join(input_dir, sub_directories[0],
                                                 fnmatch.filter(filenames, "MTD*xml")[0])
                metadata_xml = self.read_sentinel2_metadata(metadata_filename, resolution=10)

        # Check if all bands is available
        success = self.is_all_bands(input_file)
        if not success:
            print("Not all bands present in file: " + input_file)
            return 1

        # Read the Sentinel-2 data
        data, ds_img = self.read_sentinel2_data(input_file, self.params) #save_img=False)

        # Saving the data as memory maps
        metadata = {'UTM-coordinate': ds_img.transform * (0, 0),
                     'path_to_tile_in_eodata': input_file,
                     # 'normalization_values': {'b1': 10, 'b1a': 111},
                     'resolution': self.params['resolution'],
                     'shape': data[:,:,0].shape,
                     'no_data_value': -1,
                     'date': metadata_xml['sensing_time'],
                     'sensor': 'Sentinel-2',
                     'bands': self.params['bands']
                     }

        basename = os.path.splitext(os.path.basename(input_file))[0]
        tiles_dir = os.path.join(self.params['outdir'], tile_id, basename)

        if not os.path.exists(tiles_dir):
            os.makedirs(tiles_dir)

        # Save metadata
        np.savez(os.path.join(tiles_dir, 'metadata.npz'), metadata=metadata)

        # Save memory maps
        for i, bname in enumerate(self.params['bands']):
            band_filename = 'data_' + bname.lower()
            utils.save_np_memmap(os.path.join(tiles_dir, band_filename), data[:, :, i], 'float32')

        return tile_id


    def process_clouds(self, input_file):
        #Get the tile id
        tile_id = utils.get_tile_id(input_file, self.params['tile_ids'])

        # Perform cloud detection using
        # https://github.com/sentinel-hub/sentinel2-cloud-detector/blob/master/examples/sentinel2-cloud-detector-example.ipynb
        cloud_prob, cloud_mask, ds_cloud = self.cloud_detection(input_file) #, save_masks=False)

        #Saveing the cloud maks as memory map
        basename = os.path.splitext(os.path.basename(input_file))[0]
        tiles_dir = os.path.join(self.params['outdir'], tile_id, basename)

        if not os.path.exists(tiles_dir):
            os.makedirs(tiles_dir)

        utils.save_np_memmap(os.path.join(tiles_dir, 'cloud_mask'), cloud_mask[0], 'float32')

        return cloud_mask[0]











