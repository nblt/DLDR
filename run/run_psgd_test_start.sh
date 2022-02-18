#!/bin/bash
c=0
datasets=CIFAR10
model=resnet20
pretrain='output/sgd-CIFAR10-resnet20-128-0.1-20220102-161735'

for ((i=40;i<=80;i=i+2));
# for i in 20 40 50;
do
    CUDA_VISIBLE_DEVICES=2 python -u train_psgd.py --epochs=40 --arch=$model --datasets=$datasets --batch_size=128 --lr=1 --corrupt=$c --params_start=0 --params_end=$(expr $i) --n_components=40 --pretrain_dir=$pretrain --save_dir=output/${datasets}/psgd_test_start_5
done

# CUDA_VISIBLE_DEVICES=3 python -u train_psgd_test_start.py --epochs=40 --arch=$model --datasets=$datasets --batch_size=128 --lr=1 --corrupt=$c --params_start=0 --params_end=100 --n_components=50 --pretrain_dir=$pretrain --save_dir=output/${datasets}/psgd_test_start_1