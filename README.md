## NGVEO - New Generation Value chain for Earth Observation
Code for applying convolutional neural networks (CNN) to Earth Observation data from Sentinel 1 and 2 using python and PyTorch.

### Setup:
- Clone repository.
- Install the requred python packages: pip3 install -r REQUIREMENTS.txt
- Download example data from GDRIVE_LINK_GOES_HERE, and extract content into data-catalog.
- It is recommended to have a computer with a GPU (minimum 8GB memory) with CUDA and CUDNN installed.

### Typical workflow:
- Define a training dataset and make a list of tiles to include (https://finder.eocloud.eu/)
- prepare_data.py - Move data from ESAs eodata-storage to a local storage with an efficient np.memmap-format suitable for training CNNs. Both training-data and test-data should be prepared. 
- train.py - code for training a CNN
- predict.py - code for applying a trained network to new data and store as GEO-tiff 
- demo.ipynb - notebook for visualizing results using a trained model



### Credit:
- UNet implementation by Jackson Huang: https://github.com/jaxony/unet-pytorch
- SAR geocoding library by the Norwegian Computing Center

### Authors:
Anders U. Waldeland* (anders@nr.no), Arnt-Børre Salberg*, Allessandro Marin** and Øyvind Due Trier* 

\* Norwegian Computing Center (https://www.nr.no/)

\** Phi Lab, European Space Agency (http://blogs.esa.int/philab/)

 
