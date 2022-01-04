c=0
datasets=CIFAR10
model=resnet20
pretrain='output/sgd-CIFAR10-resnet20-128-0.1-20211231-161416'
CUDA_VISIBLE_DEVICES=2 python -u train_pbfgs.py --epochs=20 --datasets=$datasets --corrupt=$c --params_start=0 --params_end=81 --batch-size=1024 --n_components=40 --arch=$model --pretrain_dir=$pretrain --save_dir=output/${datasets}/pbfgs --log_wandb --project=DLDR