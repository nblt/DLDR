import sys
sys.path.append('../../../')
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

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer_MobileV2(dim, depth, t=6, kernel_size=9, patch_size=7, n_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[LinearBottleNeck(dim, dim, kernel_size=kernel_size, t=t) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )

class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, t=6):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, kernel_size=kernel_size, padding="same", groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.in_channels == self.out_channels:
            residual += x

        return residual

@register_model
def convmixer_mobilev2_64_4_2(pretrained=False, **kwargs):
    model = ConvMixer_MobileV2(64, 4, t=2, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_mobilev2_32_4_2(pretrained=False, **kwargs):
    model = ConvMixer_MobileV2(32, 4, t=2, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

if __name__ == "__main__":
    model = ConvMixer_MobileV2(128, 8, 2, 8, 1, n_classes=10)
    stat(model, (3, 32, 32))