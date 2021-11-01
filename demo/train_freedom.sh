#!/bin/bash

python -m tlc.train --gin-config-path gin_configs/freedom_config.gin
python -m tlc.models.freedom.augment_dataset --n-train-data 10000 --n-test-data 10000 --model_path runs/freedom_stage1/model.pt --tree-type FreeDOMDataTree --tree-batch-size 1 --gpu
python -m tlc.train --gin-config-path gin_configs/freedom_stage2_config.gin