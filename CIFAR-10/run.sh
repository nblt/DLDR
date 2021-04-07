#!/bin/bash
# total time comparison on one GPU

for model in resnet20
do

    CUDA_VISIBLE_DEVICES=1 python -u train_sgd.py --randomseed 1 --arch=$model --epochs=50  --save-dir=save_$model |& tee -a log_$model
    
    CUDA_VISIBLE_DEVICES=0 python -u train_pbfgs_cuda.py --epochs 20 --alpha 0.0 --params_end 51  --batch-size 1024  --n_components 40 --resume=./save_resnet20/checkpoint_refine_0.th --arch=$model  --save-dir=save_$model |& tee -a log_$model 

done
