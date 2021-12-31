c=0
datasets=CIFAR10
model=resnet20
pretrain='output/sgd-CIFAR10-resnet20-128-0.1-20211231-161416'
CUDA_VISIBLE_DEVICES=3 python -u train_psgd.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81  --batch-size 128  --n_components 40 --arch=$model --pretrain-dir=$pretrain --save-dir=output --log_wandb --project=DLDR