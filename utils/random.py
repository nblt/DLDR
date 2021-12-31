import random
import numpy as np 
import torch

def set_random_seed(seed=233): 
    print ('Random Seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False