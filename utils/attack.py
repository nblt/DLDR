import torch
import torch.nn as nn

from .model import get_model_param_vec, update_param

def epoch_adversarial(loader, model, attack, param_purturbation=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    p0 = get_model_param_vec(model)
    p0 = torch.from_numpy(p0).cuda()
    model.eval()
    for X,y in loader:
        X,y = X.cuda(), y.cuda()
        
        delta = attack(model, X, y, **kwargs)

        '''
        if param_purturbation is not None:
            update_param(model, p0 + param_purturbation)
        yp = model(X+delta)
        if param_purturbation is not None:
            update_param(model, p0 - param_purturbation)
            yp += model(X+delta)
        '''
        yp = None
        if param_purturbation is not None:
            for t in [-1,  1]:
                update_param(model, p0 + param_purturbation * t)
                if yp is None:
                    yp = model(X+delta)
                else:
                    yp += model(X+delta)
                update_param(model, p0)
        else:
            yp = model(X+delta)

        loss = nn.CrossEntropyLoss()(yp,y)
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return 1. - total_err / len(loader.dataset), total_loss / len(loader.dataset)

def fgsm(model, X, y, epsilon=8./255):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def pgd_linf(model, X, y, epsilon=8./255, alpha=0.01, num_iter=7, randomize=True):
    """ Construct PGD adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()