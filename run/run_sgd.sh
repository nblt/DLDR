c=0
datasets=CIFAR10
model=resnet110

CUDA_VISIBLE_DEVICES=0 python -u train_sgd.py --datasets $datasets --lr 0.1 --corrupt $c --arch=$model --epochs=150  --save-dir=output --log_wandb --project=DLDR