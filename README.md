# DLDR

This repository is the official implementation of **Low Dimensional Landscape Hypothesis is True: DNNs can be Trained in Tiny Subspaces**.

## Requirements

The environment requires:

+ python 3.6
+ pytorch 1.4.0
+ CUDA Version 10.0.130

## Usage

We provide training examples in `run.sh`(take CIFAR experiments for example):

```bash
# CIFAR experiments
# Label noise levels
c=0.2
datasets=CIFAR10
for model in resnet20
do
    CUDA_VISIBLE_DEVICES=0 python -u train_sgd.py --datasets $datasets --lr 0.1 --corrupt $c --arch=$model --epochs=150  --save-dir=save_labelnoise$c\_$model |& tee -a log_$model
    
    CUDA_VISIBLE_DEVICES=0 python -u train_pbfgs.py --epochs 20 --datasets $datasets --corrupt $c --params_start 0 --params_end 81  --batch-size 1024   --n_components 40 --arch=$model  --save-dir=save_labelnoise$c\_$model |& tee -a log_$model 

    CUDA_VISIBLE_DEVICES=0 python -u train_psgd.py --epochs 40 --datasets $datasets --lr 1 --corrupt $c --params_start 0 --params_end 81  --batch-size 128  --n_components 40 --arch=$model  --save-dir=save_labelnoise$c\_$model |& tee -a log_$model 
done
```

1. enter directory

   ```bash
   $ cd DLDR
   ```
   
2. run the demo

   ```bash
   $ ./run.sh
   
   ```

Specifically, the demo consists of two steps:

The first step is performing DLDR sampling using regular optimizer (e.g., SGD):

```bash
$ CUDA_VISIBLE_DEVICES=0 python -u train_sgd.py --datasets $datasets --lr 0.1 --corrupt $c --arch=$model --epochs=200  --save-dir=save_labelnoise$c\_$model |& tee -a log_$model
```

You can set `randomseed` and total training epochs `epochs` by yourself. By default, we will sample the model parameters after every epoch training.

The second step is training the *independent* variables obtained from DLDR from the same initialization of the model:

```bash
$ CUDA_VISIBLE_DEVICES=0 python -u train_pbfgs.py --epochs 20 --datasets $datasets --corrupt $c --params_start 0 --params_end 81  --batch-size 1024   --n_components 40 --arch=$model  --save-dir=save_labelnoise$c\_$model |& tee -a log_$model 
```

where `epochs`  denotes the total training epochs for P-BFGS algorithm, `params_start` denotes where our DLDR sampling begins and `params_end` denotes where DLDR sampling stops. `n_components` is the number of variables we use.

We can set the `datasets` among CIFAR10 and CIFAR100. The label noise level `c` is a real number in the range `[0, 1]`.

## Citation
>@article{li2022low, \
  title={Low Dimensional Trajectory Hypothesis is True: DNNs can be Trained in Tiny Subspaces}, \
  author={Li, Tao and Tan, Lei and Huang, Zhehao and Tao, Qinghua and Liu, Yipeng and Huang, Xiaolin}, \
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, \
  year={2022}, \
  publisher={IEEE} \
}