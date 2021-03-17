# DLDR

Dynamic Linear Dimension Reduction (DLDR) for Optimization

## Requirements

This is my experiment environment:

+ python 3.6
+ pytorch 1.4.0
+ CUDA Version 10.0.130

## Usage

We take CIFAR-100 dataset for example.

1. enter directory

   ```bash
   $ cd DLDR
   $ cd CIFAR-100
   ```

2. run the demo

   ```bash
   $ ./run.sh
   ```

   Specifically, the demo consists of two steps. 

   The first step is pre-training the model using regular optimizer (e.g., SGD) and perform DLDR sampling:

   ```bash
   $ python -u train_sgd.py --randomseed 1 --arch=$model --epochs=201  --save-dir=save_$model |& tee -a log_$model
   ```

   You can set `randomseed` and total training epochs `epochs` by yourself. By default, we will sample for total 100 epochs before and after every epochs training.

   The second step is training the variables obtained from DLDR from the SAME initialization in last step:

   ```bash
   $ python -u train_v9_mutigpu.py --epochs 20 --alpha 0.0 --params_start 0 --params_end 101  --batch-size 1024  --n_components 40 --arch=$model  --save-dir=save_$model |& tee -a log_$model 
   ```

   where `epochs`  denotes the total training epochs for quasi-Newton algorithm, `alpha` denotes the learning rate for the rest part of the gradient after projection,  `params_start` denotes where our training begins and `params_end` denotes where DLDR sampling stops. `n_components` is the number of variables we use.

   We can easily change our model `$model` in `models.py`.

