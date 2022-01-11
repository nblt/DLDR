datasets1=CIFAR10
datasets2=CIFAR100
model1=resnet20
model2=resnet110
model3=resnet34

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 train_sgd_ddp.py \
    --datasets=$datasets1 \
    --warmup_epochs=0 \
    --opt='momentum' \
    --lr=0.1 \
    --sched='step' \
    --batch_size=128 \
    --arch=$model1 \
    --epochs=150  \
    --save_dir=output/${datasets1}/sgd_ddp_test_sample \
    --log_wandb \
    --project=DLDR

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 train_sgd_ddp.py \
    --datasets=$datasets1 \
    --warmup_epochs=0 \
    --opt='momentum' \
    --lr=0.1 \
    --sched='step' \
    --batch_size=128 \
    --arch=$model2 \
    --epochs=150  \
    --save_dir=output/${datasets1}/sgd_ddp_test_sample \
    --log_wandb \
    --project=DLDR

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 train_sgd_ddp.py \
    --datasets=$datasets2 \
    --warmup_epochs=0 \
    --opt='momentum' \
    --lr=0.1 \
    --sched='step' \
    --batch_size=128 \
    --arch=$model3 \
    --epochs=150  \
    --save_dir=output/${datasets2}/sgd_ddp_test_sample \
    --log_wandb \
    --project=DLDR