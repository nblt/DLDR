c=0
datasets=ImageNet
model=resnet18

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 train_sgd_ddp.py \
    --datasets=$datasets \
    --warmup_epochs=5 \
    --opt='momentum' \
    --lr=0.1 \
    --sched='step' \
    --batch_size=128 \
    --corrupt=$c \
    --arch=$model \
    --epochs=150  \
    --step_sample_freq=500 \
    --save_dir=output/${datasets}/sgd_ddp \
    --log_wandb \
    --project=DLDR