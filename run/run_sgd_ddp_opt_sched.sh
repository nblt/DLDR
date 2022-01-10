datasets=CIFAR10
save_dir=output/${datasets}/ddp_opt_sched

for opt in 'adam' 'adamw';
do
    for sched in 'step' 'cosine';
    do
        # CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 train_sgd_ddp_opt_sched.py --warmup_epochs=0 --opt=$opt --lr=0.01 --sched=$sched --save_dir=$save_dir --log_wandb --project=DLDR
    
        CUDA_VISIBLE_DEVICES=2 python -u train_psgd_dp.py --epochs=40 --arch='resnet20' --lr=1 --params_start=0 --params_end=80 --n_components=40 --pretrain_dir=$save_dir/sgd_ddp_opt_sched-CIFAR10-resnet20-128-0.01-$opt-$sched --save_dir=$save_dir --log_wandb --project=DLDR

        CUDA_VISIBLE_DEVICES=2 python -u train_psgd_dp.py --epochs=40 --arch='resnet20' --lr=1 --params_start=0 --params_end=40 --n_components=20 --pretrain_dir=$save_dir/sgd_ddp_opt_sched-CIFAR10-resnet20-128-0.01-$opt-$sched --save_dir=$save_dir --log_wandb --project=DLDR

        # CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 train_sgd_ddp_opt_sched.py --warmup_epochs=5 --opt=$opt --lr=0.01 --sched=$sched --save_dir=$save_dir --log_wandb --project=DLDR
    
        CUDA_VISIBLE_DEVICES=2 python -u train_psgd_dp.py --epochs=40 --arch='resnet20' --lr=1 --params_start=0 --params_end=80 --n_components=40 --pretrain_dir=$save_dir/sgd_ddp_opt_sched-CIFAR10-resnet20-128-0.01-$opt-$sched-warmup --save_dir=$save_dir --log_wandb --project=DLDR

        CUDA_VISIBLE_DEVICES=2 python -u train_psgd_dp.py --epochs=40 --arch='resnet20' --lr=1 --params_start=0 --params_end=40 --n_components=20 --pretrain_dir=$save_dir/sgd_ddp_opt_sched-CIFAR10-resnet20-128-0.01-$opt-$sched-warmup --save_dir=$save_dir --log_wandb --project=DLDR
    done
done 

for sched in 'onecycle';
do
    for opt in 'adam' 'adamw';
    do
        # CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 train_sgd_ddp_opt_sched.py --warmup_epochs=0 --opt=$opt --lr=0.01 --sched=$sched --save_dir=$save_dir --log_wandb --project=DLDR
    
        CUDA_VISIBLE_DEVICES=2 python -u train_psgd_dp.py --epochs=40 --arch='resnet20' --lr=1 --params_start=0 --params_end=80 --n_components=40 --pretrain_dir=$save_dir/sgd_ddp_opt_sched-CIFAR10-resnet20-128-0.01-$opt-$sched --save_dir=$save_dir --log_wandb --project=DLDR

        CUDA_VISIBLE_DEVICES=2 python -u train_psgd_dp.py --epochs=40 --arch='resnet20' --lr=1 --params_start=0 --params_end=40 --n_components=20 --pretrain_dir=$save_dir/sgd_ddp_opt_sched-CIFAR10-resnet20-128-0.01-$opt-$sched --save_dir=$save_dir --log_wandb --project=DLDR

    done
done 
