#!/bin/bash

# ImageNet experiments
c=0
datasets=ImageNet
for model in resnet18
do
	CUDA_VISIBLE_DEVICES=0,1 python3 main.py -a $model --epochs 90 --dist_url 'tcp://127.0.0.1:8888' --dist_backend 'nccl' --multiprocessing_distributed --world_size 1 --rank 0 /home/pami/Datasets/ILSVRC2012

	# CUDA_VISIBLE_DEVICES=0,1,2 python -u train_pbfgs_imagenet.py --epochs 4 --print-freq 1000 --datasets $datasets --corrupt $c --alpha 0 --params_start 0 --params_end 241  --batch-size 256  --n_components 120 --arch=$model  --save-dir=save_$model |& tee -a log_$model 
done

