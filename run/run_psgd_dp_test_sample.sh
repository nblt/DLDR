datasets=CIFAR10
model=resnet20
pretrain='output/CIFAR10/sgd_ddp_test_sample/sgd_ddp-CIFAR10-resnet20-128-0.1-20220110-141224'

for i in 20 40;
do 
    for mode in epoch step mean smooth;
    do
        CUDA_VISIBLE_DEVICES=2 python -u train_psgd_dp.py --arch=$model --datasets=$datasets --sample_mode=$mode --params_start=0 --params_end=$(expr $i + $i) --n_components=$(expr $i) --pretrain_dir=$pretrain --save_dir=output/${datasets}/psgd_dp_test_sample
    done 
done 