c=0
datasets=CIFAR10
model=resnet20

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 train_sgd_ddp.py \
    --datasets=$datasets \
    --lr=0.001 \
    --min_lr=0.00001 \
    --warmup_epochs 10 \
    --batch_size 128 \
    --corrupt=$c \
    --arch=$model \
    --epochs=150  \
    --save_dir=output \
    --log_wandb \
    --project=DLDR