#!/bin/bash
c=0
datasets=CIFAR10
model=resnet110
pretrain='output/sgd-CIFAR10-resnet110-128-0.1-20211231-162234'

#for ((i=0;i<=20;i=i+2));
for i in 20 40 50;
do
    CUDA_VISIBLE_DEVICES=3 python -u train_psgd_dp_test_start.py --epochs=40 --arch=$model --datasets=$datasets --batch_size=128 --lr=1 --corrupt=$c --params_start=0 --params_end=$(expr $i + $i) --n_components=$(expr $i) --pretrain_dir=$pretrain --save_dir=output/${datasets}/psgd_test_start_3
done

# CUDA_VISIBLE_DEVICES=3 python -u train_psgd_test_start.py --epochs=40 --arch=$model --datasets=$datasets --batch_size=128 --lr=1 --corrupt=$c --params_start=0 --params_end=100 --n_components=50 --pretrain_dir=$pretrain --save_dir=output/${datasets}/psgd_test_start_1