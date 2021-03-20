#!/bin/bash

for model in resnet32
do

    python -u train_sgd.py --randomseed 1 --arch=$model --epochs=201  --save-dir=save_$model |& tee -a log_$model

    python -u train_pbfgs.py --epochs 20 --alpha 0.0 --params_start 0 --params_end 101  --batch-size 1024  --n_components 40 --arch=$model  --save-dir=save_$model |& tee -a log_$model 

done
