c=0
datasets=CIFAR10
model=resnet20

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 train_sgd_ddp.py \
    --datasets=$datasets \
    --warmup_epochs=0 \
    --opt='momentum' \
    --lr=0.1 \
    --sched='onecycle' \
    --batch_size=128 \
    --corrupt=$c \
    --arch=$model \
    --epochs=150  \
    --save_dir=output/${datasets}/sgd_ddp_test \
    --log_wandb \
    --project=DLDR