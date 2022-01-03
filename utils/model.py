import os
import shutil
import numpy as np
import torch
from torch._six import inf
from pathlib import Path

from .dist import save_on_master

def get_model_param_vec(model):
    """
    Return model parameters as a vector
    """
    vec = []
    for name, param in model.named_parameters():
        vec.append(param.detach().cpu().numpy().reshape(-1))
    return np.concatenate(vec, 0)


def get_model_grad_vec(model):
    """
    Return the model grad as a vector
    """
    vec = []
    for name, param in model.named_parameters():
        vec.append(param.grad.detach().reshape(-1))
    return torch.cat(vec, 0)


def update_grad(model, grad_vec):
    idx = 0
    for name, param in model.named_parameters():
        arr_shape = param.grad.shape
        size = 1
        for i in range(len(list(arr_shape))):
            size *= arr_shape[i]
        param.grad.data = grad_vec[idx:idx+size].reshape(arr_shape)
        idx += size


def update_param(model, param_vec):
    idx = 0
    for name, param in model.named_parameters():
        arr_shape = param.data.shape
        size = 1
        for i in range(len(list(arr_shape))):
            size *= arr_shape[i]
        param.data = param_vec[idx:idx+size].reshape(arr_shape)
        idx += size

def update_param_ddp(model, param_vec):
    idx = 0
    for name, param in model.named_parameters():
        arr_shape = param.shape
        size = 1
        for i in range(len(list(arr_shape))):
            size *= arr_shape[i]
        assign_param = param_vec[idx:idx+size]
        assign_param = assign_param.reshape(arr_shape)
        param = assign_param
        idx += size

def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        output_dir, _ = os.path.split(filename)
        shutil.copyfile(filename, os.path.join(output_dir, 'model_best.pth.tar'))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_model(args, epoch, output_dir, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }

        save_on_master(to_save, checkpoint_path)

def sample_model(epoch, output_dir, model_without_ddp):
    output_dir = Path(output_dir)
    epoch_name = str(epoch)
    checkpoint_path = output_dir / ('%s.pt' % epoch_name)
    save_on_master(model_without_ddp.state_dict(), checkpoint_path)

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        # for name, parms in parameters.named_parameters():	
        #     print('->name:', name, '->parms_norm:', torch.norm(parms), '->grad_requirs:',parms.requires_grad, \
        #     ' ->grad_norm:',torch.norm(parms.grad) if parms.grad is not None else 0)
        # parameters = parameters.parameters()
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
