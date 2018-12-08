## NGVEO - New Generation Value chain for Earth Observation

Python code for applying convolutional neural networks (CNN) to Earth Observation (EO) data from Sentinel 1 and 2 using python and PyTorch.

The code provides a simple framework for (1) converting EO data to an efficient format for deep learning, (2) training CNNs, and (3) applying trained networks to new data. We provide two simple examples, one regression problem (atmospheric correction) and one classification problem (cloud detection). The code can easily be modified for other solving other problems by adding your own training data in the prepare_data.py-code. In addition we pre-trained models for tree-height estimation and forest-cover estimation can be dowloaded at LINK HERE

### Setup:
- Make sure GDAL is installed. 
- Download code and setup python:
```console
git clone https://github.com/ESA-PhiLab/NGVEO.git
cd NGVEO
virtualenv -p python3 env
source env/bin/activate
pip3 install -r REQUIREMENTS.txt
``` 
- If you do not have access to ESAs eodata-drive from your computer you need to manually download the SAFE-files/folders listed in example_eodata_l*.txt (http://finder.eocloud.eu).  Also, rename the paths in these files to point to the location where the SAFE-files are downloaded.

### Main files:
- **prepare_data.py** - Move data from ESAs eodata-storage to a local storage with an efficient np.memmap-format suitable for training CNNs. Both training-data and test-data should be prepared. 
- **train.py** - Train the CNN.
- **predict.py** - Apply the trained network to new data and store as results as GEOtiff. 

### Credit:
- UNet implementation by Jackson Huang: https://github.com/jaxony/unet-pytorch 
- SAR geocoding library by the Norwegian Computing Center

### Authors:
Anders U. Waldeland* (anders@nr.no), Arnt-Børre Salberg*, Allessandro Marin** and Øyvind Due Trier* 

\* Norwegian Computing Center (https://www.nr.no/)

\** Phi Lab, European Space Agency (http://blogs.esa.int/philab/)

 
