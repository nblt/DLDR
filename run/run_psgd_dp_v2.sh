c=0
datasets=CIFAR100
model=resnet101
pretrain='output/CIFAR100/sgd/sgd-CIFAR100-resnet101-128-0.1-20220104-171853'
CUDA_VISIBLE_DEVICES=2 python -u train_psgd_dp_v2.py \
    --epochs=40 \
    --arch=$model \
    --datasets=$datasets \
    --batch_size=128 \
    --lr=1 \
    --corrupt=$c \
    --params_start=0 \
    --params_end=40 \
    --n_components=20 \
    --pretrain_dir=$pretrain \
    --save_dir=output/${datasets}/psgd_dp_v2 \
    # --log_wandb \
    # --project=DLDR