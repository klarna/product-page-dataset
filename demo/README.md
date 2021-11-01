[comment]: <> (>📋  A template README.md for code accompanying a Machine Learning paper)

# Graph Neural Networks for DOM Tree ElementPrediction

This repository is the official implementation of [Graph Neural Networks for DOM Tree ElementPrediction](?). 

[comment]: <> (>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials)

## Requirements

To install requirements:


- Install python 3.7

- Run:
```setup
make env-create
make download-requirements
source .venv/bin/activate
```

[comment]: <> (>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...)

## Training

To train the model(s) in the paper, run the following command while pointing to the respective gin config file. 
Example that runs GCN-Mean model on 10,000 data points:
```train
python -m tlc.train --dataset path_to_dataset --n-data 10000 --gin-config-path gin_configs/GCN-Mean.gin
```
The hyper parameters can be modified by changing them in the gin files located in the `gin_configs` folder.

Due to the two stage setup of the FreeDOM model we have provided `train_freedom.sh` script for convenience.

Results of training and evaluation will be saved to the `runs` folder.


