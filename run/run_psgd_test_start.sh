#!/bin/bash
c=0
datasets=CIFAR10
model=resnet110
pretrain='output/sgd-CIFAR10-resnet110-128-0.1-20211231-162234'

for ((i=0;i<=20;i=i+2));
do
    CUDA_VISIBLE_DEVICES=2 python -u train_psgd.py --epochs=40 --arch=$model --datasets=$datasets --batch_size=128 --lr=1 --corrupt=$c --params_start=$(expr $i + 0) --params_end=$(expr $i + 80) --n_components=40 --pretrain_dir=$pretrain --save_dir=output/${datasets}/psgd_test_start
done