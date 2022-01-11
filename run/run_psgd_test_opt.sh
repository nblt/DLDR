
datasets1=CIFAR10
datasets2=CIFAR100
model1=resnet20
model2=resnet110
model3=resnet34

pretrain1='output/sgd-CIFAR10-resnet20-128-0.1-20220102-161735'
pretrain2='output/sgd-CIFAR10-resnet110-128-0.1-20211231-162234'
pretrain3='output/CIFAR100/sgd/sgd-CIFAR100-resnet34-128-0.1-20220104-171647'

for i in 20 40;
do 
    for model in $model1 $model2;
    do 
        if [ "$model" = "resnet20" ]; then
            pretrain=$pretrain1
        fi
        if [ "$model" = "resnet110" ]; then
            pretrain=$pretrain2
        fi 
        CUDA_VISIBLE_DEVICES=3 python -u train_psgd_dp.py --arch=$model --datasets=$datasets1 --warmup_epochs=0 --opt=momentum --lr=1 --sched=step --decay_epochs=30 --params_start=0 --params_end=$(expr $i + $i) --n_components=$(expr $i) --pretrain_dir=$pretrain --save_dir=output/${datasets1}/psgd_test_opt --log_wandb --project=DLDR
        CUDA_VISIBLE_DEVICES=3 python -u train_psgd_dp.py --arch=$model --datasets=$datasets1 --warmup_epochs=3 --opt=momentum --lr=1 --sched=step --decay_epochs=20 --params_start=0 --params_end=$(expr $i + $i) --n_components=$(expr $i) --pretrain_dir=$pretrain --save_dir=output/${datasets1}/psgd_test_opt --log_wandb --project=DLDR
        CUDA_VISIBLE_DEVICES=3 python -u train_psgd_dp.py --arch=$model --datasets=$datasets1 --warmup_epochs=0 --opt=adam --lr=0.1 --sched=onecycle --params_start=0 --params_end=$(expr $i + $i) --n_components=$(expr $i) --pretrain_dir=$pretrain --save_dir=output/${datasets1}/psgd_test_opt --log_wandb --project=DLDR
        CUDA_VISIBLE_DEVICES=3 python -u train_psgd_dp.py --arch=$model --datasets=$datasets1 --warmup_epochs=0 --opt=adamw --lr=0.1 --sched=onecycle --params_start=0 --params_end=$(expr $i + $i) --n_components=$(expr $i) --pretrain_dir=$pretrain --save_dir=output/${datasets1}/psgd_test_opt --log_wandb --project=DLDR
    done 
    CUDA_VISIBLE_DEVICES=3 python -u train_psgd_dp.py --arch=$model3 --datasets=$datasets2 --warmup_epochs=0 --opt=momentum --lr=1 --sched=step --decay_epochs=30 --params_start=0 --params_end=$(expr $i + $i) --n_components=$(expr $i) --pretrain_dir=$pretrain3 --save_dir=output/${datasets2}/psgd_test_opt --log_wandb --project=DLDR
    CUDA_VISIBLE_DEVICES=3 python -u train_psgd_dp.py --arch=$model3 --datasets=$datasets2 --warmup_epochs=3 --opt=momentum --lr=1 --sched=step --decay_epochs=20 --params_start=0 --params_end=$(expr $i + $i) --n_components=$(expr $i) --pretrain_dir=$pretrain3 --save_dir=output/${datasets2}/psgd_test_opt --log_wandb --project=DLDR
    CUDA_VISIBLE_DEVICES=3 python -u train_psgd_dp.py --arch=$model3 --datasets=$datasets2 --warmup_epochs=0 --opt=adam --lr=0.1 --sched=onecycle --params_start=0 --params_end=$(expr $i + $i) --n_components=$(expr $i) --pretrain_dir=$pretrain3 --save_dir=output/${datasets2}/psgd_test_opt --log_wandb --project=DLDR
    CUDA_VISIBLE_DEVICES=3 python -u train_psgd_dp.py --arch=$model3 --datasets=$datasets2 --warmup_epochs=0 --opt=adamw --lr=0.1 --sched=onecycle --params_start=0 --params_end=$(expr $i + $i) --n_components=$(expr $i) --pretrain_dir=$pretrain3 --save_dir=output/${datasets2}/psgd_test_opt --log_wandb --project=DLDR

done