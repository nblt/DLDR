c=0
datasets=ImageNet
model=resnet18
pretrain='/home/DiskB/hzhcode/DLDR/save_resnet18'
CUDA_VISIBLE_DEVICES=1,2,3 python -u train_psgd.py \
    --epochs=40 \
    --arch=$model \
    --datasets=$datasets \
    --batch_size=128 \
    --lr=1 \
    --corrupt=$c \
    --params_start=0 \
    --params_end=241 \
    --n_components=120 \
    --pretrain_dir=$pretrain \
    --save_dir=output/${datasets}/imagenet_psgd_test \
    # --log_wandb \
    # --project=DLDR