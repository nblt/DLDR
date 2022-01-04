c=0
datasets=CIFAR100
model=resnet101

CUDA_VISIBLE_DEVICES=3 python -u train_sgd.py \
    --datasets=$datasets \
    --lr=0.1 \
    --corrupt=$c \
    --arch=$model \
    --epochs=200  \
    --save_dir=output/${datasets}/sgd \
    --log_wandb \
    --project=DLDR