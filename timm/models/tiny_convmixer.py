import math
import numpy as np
from functools import partial
from torchstat import stat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn.parameter import Parameter
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .registry import register_model
from .deconv import deConv2d

_cfg = {
    'url': '',
    'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
    'crop_pct': .96, 'interpolation': 'bicubic',
    'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head'
}


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x - self.fn(x)

class tiny_ConvMixer2(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, n_classes=1000, 
                weight_init='kaiming_uniform_5',
                alpha_init='uniform_0_1',
                norm_type='batch',
                add_dim=0,
                ):
        super(tiny_ConvMixer2, self).__init__()

        print('tiny_convmixer2')
        self.stem = deConv2d(int(math.log(dim, 2))+add_dim if add_dim >= 0 else 1, 3, dim, kernel_size=patch_size, stride=patch_size, bias=False, 
                        weight_init=weight_init, alpha_init=alpha_init)
        self.act_layer = nn.GELU()

        if norm_type == 'batch':
            self.stem_bn = nn.BatchNorm2d(dim)
        elif norm_type == 'layer':
            self.stem_bn = nn.LayerNorm((32//patch_size))
        self.convmix_layers = nn.Sequential(
            *[
                nn.Sequential(
                    Residual(nn.Sequential(
                        deConv2d(int(math.log(dim, 2))+add_dim if add_dim >= 0 else 1, dim, dim, kernel_size, groups=dim, padding="same", bias=False, 
                                weight_init=weight_init, alpha_init=alpha_init),
                        nn.GELU(),
                        nn.BatchNorm2d(dim) if norm_type == 'batch' else nn.LayerNorm((32//patch_size))
                    )),
                    deConv2d(int(math.log(dim, 2))+add_dim if add_dim >= 0 else 1, dim, dim, kernel_size=1, bias=False, 
                            weight_init=weight_init, alpha_init=alpha_init),
                    nn.GELU(),
                    nn.BatchNorm2d(dim) if norm_type == 'batch' else nn.LayerNorm((32//patch_size))
                )for i in range(depth)
            ])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(dim, n_classes)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.act_layer(x)
        x = self.stem_bn(x)
        x = self.convmix_layers(x)
        x = self.classifier(x)
        return x

class One_ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, n_classes=1000, 
                weight_init='kaiming_uniform_5',
                alpha_init='uniform_0_1', bn=True):
        super(One_ConvMixer, self).__init__()
        self.stem = deConv2d(1, 3, dim, kernel_size=patch_size, stride=patch_size, bias=False, 
                        weight_init=weight_init, alpha_init=alpha_init)
        self.act_layer = nn.GELU()

        self.stem_bn = nn.BatchNorm2d(dim) if bn else nn.Identity()
        self.convmix_layers = nn.Sequential(
            *[
                nn.Sequential(
                    Residual(nn.Sequential(
                        deConv2d(1, dim, dim, kernel_size, groups=dim, padding="same", bias=False, 
                                weight_init=weight_init, alpha_init=alpha_init),
                        nn.GELU(),
                        nn.BatchNorm2d(dim) if bn else nn.Identity()
                    )),
                    deConv2d(1, dim, dim, kernel_size=1, bias=False, 
                            weight_init=weight_init, alpha_init=alpha_init),
                    nn.GELU(),
                    nn.BatchNorm2d(dim) if bn else nn.Identity()
                )for i in range(depth)
            ])
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(dim, n_classes)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.act_layer(x)
        x = self.stem_bn(x)
        x = self.convmix_layers(x)
        x = self.classifier(x)
        return x


@register_model
def tiny_convmixer_32_4(pretrained=False, **kwargs):
    model = tiny_ConvMixer2(32, 4, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def one_convmixer_32_4(pretrained=False, **kwargs):
    model = One_ConvMixer(32, 4, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def one_convmixer_32_4_nobn(pretrained=False, **kwargs):
    model = One_ConvMixer(32, 4, kernel_size=8, patch_size=1, n_classes=10, bn=False)
    model.default_cfg = _cfg
    return model

@register_model
def one_convmixer_256_8_nobn(pretrained=False, **kwargs):
    model = One_ConvMixer(256, 8, kernel_size=8, patch_size=1, n_classes=10, bn=False)
    model.default_cfg = _cfg
    return model

@register_model
def tiny_convmixer_32_4_orthogonal_lognormal(pretrained=False, **kwargs):
    model = tiny_ConvMixer2(32, 4, kernel_size=8, patch_size=1, n_classes=10, weight_init='orthogonal', alpha_init='lognormal')
    model.default_cfg = _cfg
    return model

@register_model
def tiny_convmixer_32_4_sparse_09_lognormal(pretrained=False, **kwargs):
    model = tiny_ConvMixer2(32, 4, kernel_size=8, patch_size=1, n_classes=10, weight_init='sparse_09', alpha_init='lognormal')
    model.default_cfg = _cfg
    return model

@register_model
def tiny_convmixer_256_8(pretrained=False, **kwargs):
    model = tiny_ConvMixer2(256, 8, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def tiny_convmixer_256_8(pretrained=False, **kwargs):
    model = tiny_ConvMixer2(256, 4, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def tiny_convmixer_1024_2(pretrained=False, **kwargs):
    model = tiny_ConvMixer2(1024, 2, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def tiny_convmixer_256_8_orthogonal_lognormal(pretrained=False, **kwargs):
    model = tiny_ConvMixer2(256, 8, kernel_size=8, patch_size=1, n_classes=10, weight_init='orthogonal', alpha_init='lognormal')
    model.default_cfg = _cfg
    return model

@register_model
def tiny_convmixer_256_3_orthogonal_lognormal(pretrained=False, **kwargs):
    model = tiny_ConvMixer2(256, 8, kernel_size=3, patch_size=1, n_classes=10, weight_init='orthogonal', alpha_init='lognormal')
    model.default_cfg = _cfg
    return model

@register_model
def tiny_convmixer_256_8_2_orthogonal_lognormal(pretrained=False, **kwargs):
    model = tiny_ConvMixer2(256, 8, kernel_size=8, patch_size=2, n_classes=10, weight_init='orthogonal', alpha_init='lognormal')
    model.default_cfg = _cfg
    return model

@register_model
def tiny_convmixer_256_8_orthogonal_lognormal_layer(pretrained=False, **kwargs):
    model = tiny_ConvMixer2(256, 8, kernel_size=8, patch_size=1, n_classes=10, 
                            weight_init='orthogonal', alpha_init='lognormal', norm_type='layer')
    model.default_cfg = _cfg
    return model

@register_model
def tiny_convmixer_256_3_orthogonal_lognormal_10(pretrained=False, **kwargs):
    model = tiny_ConvMixer2(256, 8, kernel_size=3, patch_size=1, n_classes=10, 
                            weight_init='orthogonal', alpha_init='lognormal', add_dim=10)
    model.default_cfg = _cfg
    return model