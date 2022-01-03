import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class pretrain_model(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

class reparam_model(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model 

    def forward(self, x):
        return self.model(x)