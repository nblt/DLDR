import os
import logging 
import numpy as np
import torch 
from tqdm import tqdm
from sklearn.decomposition import PCA

from .model import get_model_param_vec

def get_W(args, model, mode="", beta=0.8, step_times=10000):
    """
    mode = "" or "epoch" -> epoch end 
    mode = "step" -> step end 
    mode = "mean" -> mean 
    mode = "smooth" -> exp smooth 
    """
    W = []
    logging.info(f'params: from {args.params_start} to {args.params_end}')
    for epoch in tqdm(range(args.params_start, args.params_end)):
        ############################################################################
        # if i % 2 != 0: continue
        if mode == "" or mode == "epoch":   
            model.load_state_dict(torch.load(os.path.join(args.pretrain_dir, f'{epoch}.pt')))
            W.append(get_model_param_vec(model))

        elif mode == "step":
            model.load_state_dict(torch.load(os.path.join(args.pretrain_dir, f'{epoch}.pt')))
            W.append(get_model_param_vec(model))
            for step in range(step_times):
                load_path = os.path.join(args.pretrain_dir, f"{epoch}-{step}.pt")
                if os.path.exists(load_path):
                    try:
                        model.load_state_dict(torch.load(load_path))
                        W.append(get_model_param_vec(model))
                    except:
                        continue

        elif mode == "mean":
            cnt = 1
            model.load_state_dict(torch.load(os.path.join(args.pretrain_dir, f'{epoch}.pt')))
            tmp_W = get_model_param_vec(model)
            for step in range(step_times):
                load_path = os.path.join(args.pretrain_dir, f"{epoch}-{step}.pt")
                if os.path.exists(load_path):
                    try:
                        model.load_state_dict(torch.load(load_path))
                        tmp_W += get_model_param_vec(model)
                        cnt += 1
                    except:
                        continue
            tmp_W = tmp_W/cnt
            W.append(tmp_W)

        elif mode == "smooth":
            model.load_state_dict(torch.load(os.path.join(args.pretrain_dir, f'{epoch}.pt')))
            tmp_W = get_model_param_vec(model)
            for step in range(step_times):
                load_path = os.path.join(args.pretrain_dir, f"{epoch}-{step}.pt")
                if os.path.exists(load_path):
                    try:
                        model.load_state_dict(torch.load(load_path))
                        tmp_W = beta*tmp_W + (1-beta) * get_model_param_vec(model)
                    except:
                        continue
            W.append(tmp_W)

        else:
            raise RuntimeError(f"Unknown mode {mode} to get W for PCA") 
        
    W = np.array(W)
    logging.info(f'W: {W.shape}')
    return W

def get_P(args, W, output_dir=None):
    # Obtain base variables through PCA
    pca = PCA(n_components=args.n_components)
    pca.fit_transform(W)
    P = np.array(pca.components_)
    if output_dir is not None:
        np.save(os.path.join(output_dir, f"P_{args.params_start}_{args.params_end}_{args.n_components}.npy"), P)
    logging.info(f'ratio: {pca.explained_variance_ratio_}')
    logging.info(f'P: {P.shape}')
    return P, pca
