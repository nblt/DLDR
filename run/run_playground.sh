c=0
datasets=CIFAR10
model=resnet110
pretrain='output/sgd-CIFAR10-resnet110-128-0.1-20211231-162234'
CUDA_VISIBLE_DEVICES=0 python -u playground.py --epochs=40 --datasets=$datasets --lr=1 --corrupt=$c --params_start=0 --params_end=81  --batch_size=128  --n_components=1 --arch=$model --pretrain_dir=$pretrain --save_dir=output --log_wandb --project=DLDR