c=0
datasets=CIFAR10
model=resnet20
pretrain='output/sgd_ddp-CIFAR10-resnet20-128-0.1-20220103-234309'
CUDA_VISIBLE_DEVICES=2 python -u train_psgd_dp.py \
    --epochs=40 \
    --datasets=$datasets \
    --lr=1 \
    --corrupt=$c \
    --params_start=0 \
    --params_end=81 \
    --batch_size=128 \
    --n_components=40 \
    --arch=$model \
    --pretrain_dir=$pretrain \
    --save_dir=output \
    # --log_wandb \
    # --project=DLDR