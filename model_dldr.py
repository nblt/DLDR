import math
from numpy import require
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
        with torch.no_grad():
            return torch.mm(self.param, P)

class reparam_model_v1(nn.Module):

    def __init__(self, model, param0, n_components, P = None):
        super().__init__()
        self.model = model 
        self.param0 = nn.Parameter(param0, requires_grad=False)
        self.n_components = n_components
        self.P = nn.Parameter(P, requires_grad=False)
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

    def prepare_forward(self):
        with torch.no_grad():
            assert self.P is not None, "there is no transformation"
            low_param = self.low_dim_linear_model(self.P)[0]
            # print(f"low param:{low_param[:20]}, param0:{self.param0[:20]}")
            # self.param0 = self.param0.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model_param = self.param0 + low_param
            self.update_model_param(model_param)

    def forward(self, x):
        return self.model(x)

    def get_param(self):
        return self.low_dim_linear_model.parameters()

    def update_low_dim_grad(self):
        with torch.no_grad():
            grad = utils.get_model_grad_vec(self.model)
            # print(f"grad: {grad[:20]}")
            gk = torch.mm(self.P, grad.reshape(-1,1))
            # print(f"gk: {gk.transpose(0, 1)[:20]}")
            for name, weight in self.low_dim_linear_model.named_parameters():
                weight.grad = gk.transpose(0, 1)

class low_dim_linear_model_v2(nn.Module):

    def __init__(self, n_components: list):
        super().__init__()
        self.n_group = len(n_components)
        self.n_components = n_components
        self.param = nn.Parameter(torch.zeros((1, sum(n_components))), requires_grad=True)

    def forward(self, P: list):
        assert len(P) == self.n_group, f"Wrong match group len of P:{len(P)} and n_group:{self.n_group}"
        idx = 0
        output = []
        for i, n in enumerate(self.n_components):
            output.append(torch.mm(self.param[:, idx:idx+n], P[i])[0])
            idx += n 
        return output

class reparam_model_v2(nn.Module):

    def __init__(self, model: nn.Module, 
                 param0, 
                 n_components: list or int, 
                 group_params: list = None, 
                 other_params: list = None, 
                 P = None):
        super().__init__()
        if isinstance(n_components, list):
            assert len(group_params) == len(n_components) == len(P), f"Wrong match group lengths of group_params:{len(group_params)}, n_components:{len(n_components)}, P:{len(P)}"
            self.n_group = len(n_components)
            self.group_params = group_params
            self.other_params = other_params
            self.param0 = param0
            self.n_components = n_components
            self.P = P
        
        elif isinstance(n_components, int):
            self.n_group = 1
            self.group_params = [[p for p in model.parameters()]]
            self.other_params = []
            self.param0 = [param0]
            self.n_components = [n_components]
            self.P = [P]

        self.model = model 
        self.low_dim_linear_model = low_dim_linear_model_v2(self.n_components)

    def update_model_param(self, model_param):
        for n in range(self.n_group):
            idx = 0
            for param in self.group_params[n]:
                arr_shape = param.data.shape
                size = 1
                for i in range(len(list(arr_shape))):
                    size *= arr_shape[i]
                assign_param = model_param[n][idx:idx+size]
                assign_param = assign_param.reshape(arr_shape)
                param.data = assign_param
                idx += size

    def forward(self, x):
        assert self.P is not None, "there is no transformation"
        low_param = self.low_dim_linear_model(self.P)
        # print("low_param", low_param)
        # print("param0", self.param0)
        model_param = [self.param0[n] + low_param[n] for n in range(self.n_group)]
        self.update_model_param(model_param)

        return self.model(x)

    def get_param(self):
        return self.low_dim_linear_model.parameters()

    def get_model_grad_vec(self):
        group_grad_vec = []
        for n in range(self.n_group):
            grad_vec = []
            for param in self.group_params[n]:
                grad_vec.append(param.grad.detach().reshape(-1))
            group_grad_vec.append(torch.cat(grad_vec, 0))
        return group_grad_vec

    def update_low_dim_grad(self):
        group_grad_vec = self.get_model_grad_vec()
        grad_vec = []
        for n in range(self.n_group):
            gk = torch.mm(self.P[n], group_grad_vec[n].reshape(-1,1))
            grad_vec.append(gk)
        for weight in self.low_dim_linear_model.parameters():
            weight.grad = torch.cat(grad_vec, 0).transpose(0, 1)

# for test
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

class reparam_model_v3(nn.Module):
    
    def __init__(self, model, n_components):
        super().__init__()
        self.model = model 
        # self.param0 = nn.Parameter(param0, requires_grad=False)
        self.n_components = n_components
        # self.P = nn.Parameter(P, requires_grad=False)
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

    def prepare_forward(self, P, param0):
        with torch.no_grad():
            low_param = self.low_dim_linear_model(P)[0]
            # print(f"low param:{low_param[:20]}, param0:{self.param0[:20]}")
            # self.param0 = self.param0.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model_param = param0 + low_param
            self.update_model_param(model_param)

    def forward(self, x):
        return self.model(x)

    def get_param(self):
        return self.low_dim_linear_model.parameters()

    def update_low_dim_grad(self, P):
        with torch.no_grad():
            grad = utils.get_model_grad_vec(self.model)
            # print(f"grad: {grad[:20]}")
            gk = torch.mm(P, grad.reshape(-1,1))
            # print(f"gk: {gk.transpose(0, 1)[:20]}")
            for name, weight in self.low_dim_linear_model.named_parameters():
                weight.grad = gk.transpose(0, 1)

if __name__ == "__main__":

    model = fn()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model))
    print('number of params: {}'.format(n_parameters))
    
    # all_params = model.parameters()
    # weight_params = [[], []]
    # other_params = []

    # for pname, p in model.named_parameters():
    #     if '0' in pname:
    #         weight_params[0].append(p)
    #     elif '1' in pname:
    #         weight_params[1].append(p)
    #     else:
    #         other_params.append(p)

    # P = []
    # n_components = [2, 2]
    # for i, g in enumerate(weight_params):
    #     num = sum(p.numel() for p in g)
    #     P.append(torch.rand((n_components[i], num)))

    # print("P", P)

    # param0 = []
    # for i, g in enumerate(weight_params):
    #     vec = []
    #     for param in g:
    #         vec.append(param.detach().cpu().reshape(-1))
    #     param0.append(torch.concat(vec, 0))
    
    param0 = torch.from_numpy(utils.get_model_param_vec(model))
    print("param0", param0)

    # reparam_model = reparam_model_v2(model, param0, n_components, weight_params, other_params, P)
    reparam_model = reparam_model_v2(model=model, param0=param0, n_components=3, P=torch.rand((3, n_parameters)))
    input = torch.rand(3)
    output = reparam_model(input)
    output.backward()
    reparam_model.update_low_dim_grad()
    for name, weight in reparam_model.named_parameters():
        # print("weight:", weight) # 打印权重，看是否在变化
        if weight.requires_grad:
            print(f">>> {name} weight and grad:", weight, weight.grad) # 打印梯度，看是否丢失