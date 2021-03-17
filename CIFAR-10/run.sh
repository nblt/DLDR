
#!/bin/bash

for model in resnet20
do

    python -u train_sgd.py --randomseed 1 --arch=$model --epochs=201  --save-dir=save_$model |& tee -a log_$model

    python -u train_v9_mutigpu.py --epochs 20 --alpha 0.0 --params_end 51  --batch-size 1024  --n_components 40 --arch=$model  --save-dir=save_$model |& tee -a log_$model 
 
	# python -u train_v9_acc_grad.py --epochs 20 --alpha 0.0 --params_end 51  --batch-size 128  --n_components 40 --resume=./save_$model/checkpoint_refine_0.th --arch=$model  --save-dir=save_$model |& tee -a log_$model 
    # python -u train_v9_acc_grad.py --epochs 20 --alpha 0.01 --params_end 51  --batch-size 128  --n_components 40 --resume=./save_$model/checkpoint_refine_0.th --arch=$model  --save-dir=save_$model |& tee -a log_$model 
    # python -u train_v9_acc_grad.py --epochs 20 --alpha 0.0 --params_end 101  --batch-size 128  --n_components 40 --resume=./save_$model/checkpoint_refine_0.th --arch=$model  --save-dir=save_$model |& tee -a log_$model 
    # python -u train_v9_acc_grad.py --epochs 20 --alpha 0.01 --params_end 101  --batch-size 128  --n_components 40 --resume=./save_$model/checkpoint_refine_0.th --arch=$model  --save-dir=save_$model |& tee -a log_$model 
done
