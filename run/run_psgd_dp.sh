datasets=CIFAR10
model=resnet20
pretrain='output/sgd-CIFAR10-resnet20-128-0.1-20220102-161735'
CUDA_VISIBLE_DEVICES=2 python -u train_psgd_dp.py \
    --epochs=40 \
    --arch=$model \
    --datasets=$datasets \
    --batch_size=128 \
    --lr=1 \
    --sample_mode=epoch \
    --params_start=0 \
    --params_end=80 \
    --n_components=40 \
    --pretrain_dir=$pretrain \
    --save_dir=output/${datasets}/psgd_dp_test \
    # --log_wandb \
    # --project=DLDR