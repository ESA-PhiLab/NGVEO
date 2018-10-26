## NGVEO - New Generation Value chain for Earth Observation
<<<<<<< HEAD
This repository contains code for applying deep learning to Earth Observation data, with focus on Sentinel data.

The code base is being actively developed until January 2019 and can not be expected to be stable until then.
=======

This repository contains code for applying convolutional neural networks (CNN) to Earth Observation data. The code supports earth observation data from ESAs Sentinel satelites.

Warning: The code base is being actively developed until January 2019 and can not be expected to be stable until then.

#### Setup:
- Download or clone repository
- Install requirements: pip3 install -r REQUIREMENTS.txt
- For the forest-demo, download example data and move the folder "data" into demo_forest.
- It is recomended to have a computer with a GPU (minimun 8GB memory) with CUDA and CUDNN installed.

#### Main files:
- preprocessing.py - Move data from ESAs eodata-storage to an efficient np.memmap-format suitable for training CNNs.
- train.py - code for training a CNN
- inference.py - code for applying a trained CNN to new EO data
- evalutaion.py - code for evaluating the performance of the CNN

#### Other files worth taking a look at:
- sentinel_dataset/tile.py      - class for reading each EO-tile with the np.memmap-format
- sentinel_dataset/dataset.py   - class for combining tile-objects into one dataset for training, validation, or testing

#### Forest-demo:
Use the demo_forest.ipynb-notebook for applying the tree-height and tree-cover networks to sentnel data.



>>>>>>> master
