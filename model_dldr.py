import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class pretrain_model(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

class low_dim_linear_model(nn.Module):

    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        self.param = nn.Parameter(torch.zeros((1, n_components)), requires_grad=True)

    def forward(self, P):
        return torch.mm(self.param, P)
class reparam_model_v1(nn.Module):

    def __init__(self, model, param0, n_components, P = None):
        super().__init__()
        self.model = model 
        self.param0 = param0
        self.n_components = n_components
        self.P = P
        self.low_dim_linear_model = low_dim_linear_model(n_components)

    def update_model_param(self, param_vec):
        idx = 0
        for name, param in self.model.named_parameters():
            arr_shape = param.data.shape
            size = 1
            for i in range(len(list(arr_shape))):
                size *= arr_shape[i]
            assign_param = param_vec[idx:idx+size]
            assign_param = assign_param.reshape(arr_shape)
            param.data = assign_param
            idx += size

    def forward(self, x):
        assert self.P is not None, "there is no transformation"
        low_param = self.low_dim_linear_model(self.P)[0]
        # print(low_param, self.param0)
        model_param = self.param0 + low_param
        self.update_model_param(model_param)

        return self.model(x)

    def get_param(self):
        return self.low_dim_linear_model.parameters()

    def update_low_dim_grad(self):
        grad = utils.get_model_grad_vec(self.model)
        gk = torch.mm(self.P, grad.reshape(-1,1))
        for name, weight in self.low_dim_linear_model.named_parameters():
            weight.grad = gk.transpose(0, 1)


class fn(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(3, 3),
            nn.Linear(3, 3),
            nn.Linear(3, 1)
        )
    def forward(self, x):
        return self.layer(x)

if __name__ == "__main__":

    model = fn()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model))
    print('number of params: {}'.format(n_parameters))

    input = torch.rand(3)
    output = model(input)
    output.backward()
    print("first")
    for name, weight in model.named_parameters():
        # print("weight:", weight) # 打印权重，看是否在变化
        if weight.requires_grad:
            print(f">>> {name} weight and grad:", weight, weight.grad) # 打印梯度，看是否丢失
    print("zero grad")
    model.zero_grad()

    P = torch.rand((3, n_parameters))
    print("P.shape", P.shape)
    reparam_model = reparam_model_v1(model, 3, P)

    output = reparam_model(input)
    output.backward()
    reparam_model.update_low_dim_grad()
    for name, weight in reparam_model.named_parameters():
        # print("weight:", weight) # 打印权重，看是否在变化
        if weight.requires_grad:
            print(f">>> {name} weight and grad:", weight, weight.grad) # 打印梯度，看是否丢失