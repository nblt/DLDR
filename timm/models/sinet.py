import sys
sys.path.append('../../../')
import torch
import torch.nn as nn
from torchstat import stat
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model


_cfg = {
    'url': '',
    'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
    'crop_pct': .96, 'interpolation': 'bicubic',
    'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head'
}

class FC(nn.Module):

    def __init__(self, hidden_features, hidden_layers, n_classes=10):
        super(FC, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(3*32*32, hidden_features),
            nn.ReLU(inplace=True),
            *[nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(inplace=True))
            for _ in range(hidden_layers)],
            nn.Linear(hidden_features, n_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

@register_model
def fc_256_3(pretrained=False, **kwargs):
    model = FC(hidden_features=256, hidden_layers=3, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def fc_128_3(pretrained=False, **kwargs):
    model = FC(hidden_features=128, hidden_layers=3, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def fc_64_3(pretrained=False, **kwargs):
    model = FC(hidden_features=64, hidden_layers=3, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def fc_32_3(pretrained=False, **kwargs):
    model = FC(hidden_features=32, hidden_layers=3, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def fc_16_3(pretrained=False, **kwargs):
    model = FC(hidden_features=16, hidden_layers=3, n_classes=10)
    model.default_cfg = _cfg
    return model