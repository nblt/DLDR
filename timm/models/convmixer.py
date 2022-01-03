import sys
sys.path.append('../../../')
import convmixer
import torch.nn as nn
from convmixer import (
    ConvMixer, EmbMixer, ConvMixer2, ConvMixer_nopooling, BothPath_v2, BothPath_v5, ConvMixer_SE, GAP_ConvMixer, 
    PoolMixer, TransConvMixer, TransConvMixer_WindowRes
)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model


_cfg = {
    'url': '',
    'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
    'crop_pct': .96, 'interpolation': 'bicubic',
    'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head'
}


@register_model
def convmixer_1536_20(pretrained=False, **kwargs):
    model = ConvMixer(1536, 20, kernel_size=9, patch_size=7, n_classes=1000)
    model.default_cfg = _cfg
    return model


@register_model
def convmixer_768_32(pretrained=False, **kwargs):
    model = ConvMixer(768, 32, kernel_size=7, patch_size=7, n_classes=1000)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_256_16(pretrained=False, **kwargs):
    model = ConvMixer(256, 16, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

# 671,754
@register_model
def convmixer_256_8(pretrained=False, **kwargs):
    model = ConvMixer(256, 8, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

# 1,724,426
@register_model
def convmixer_256_8_SE1(pretrained=False, **kwargs):
    model = ConvMixer_SE(256, 8, kernel_size=8, patch_size=1, n_classes=10, reduction=1)
    model.default_cfg = _cfg
    return model

# 690,218
@register_model
def convmixer_256_8_SE64(pretrained=False, **kwargs):
    model = ConvMixer_SE(256, 8, kernel_size=8, patch_size=1, n_classes=10, reduction=64)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_128_4(pretrained=False, **kwargs):
    model = ConvMixer(128, 4, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_128_4_4(pretrained=False, **kwargs):
    model = ConvMixer(128, 4, kernel_size=8, patch_size=4, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_128_4_8(pretrained=False, **kwargs):
    model = ConvMixer(128, 4, kernel_size=8, patch_size=8, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def transconvmixer_128_4(pretrained=False, **kwargs):
    model = TransConvMixer(128, 4, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def transconvmixer_windowres_128_4(pretrained=False, **kwargs):
    model = TransConvMixer_WindowRes(128, 4, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def poolmixer_128_4_max_2_2(pretrained=False, **kwargs):
    model = PoolMixer(128, 4, kernel_size=2, pooltype=nn.MaxPool2d, poolstride=2, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def poolmixer_128_4_max_3_1(pretrained=False, **kwargs):
    model = PoolMixer(128, 4, kernel_size=3, pooltype=nn.MaxPool2d, poolstride=1, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def poolmixer_128_4_max_5_1(pretrained=False, **kwargs):
    model = PoolMixer(128, 4, kernel_size=5, pooltype=nn.MaxPool2d, poolstride=1, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def poolmixer_128_4_avg_2_2(pretrained=False, **kwargs):
    model = PoolMixer(128, 4, kernel_size=2, pooltype=nn.AvgPool2d, poolstride=2, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def poolmixer_128_4_avg_3_1(pretrained=False, **kwargs):
    model = PoolMixer(128, 4, kernel_size=3, pooltype=nn.AvgPool2d, poolstride=1, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def poolmixer_128_4_avg_5_1(pretrained=False, **kwargs):
    model = PoolMixer(128, 4, kernel_size=5, pooltype=nn.AvgPool2d, poolstride=1, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_64_4(pretrained=False, **kwargs):
    model = ConvMixer(64, 4, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_32_4(pretrained=False, **kwargs):
    model = ConvMixer(32, 4, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def gap_convmixer_32_16(pretrained=False, **kwargs):
    model = GAP_ConvMixer(32, 16, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_32_16(pretrained=False, **kwargs):
    model = ConvMixer(32, 16, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def gap_convmixer_32_4(pretrained=False, **kwargs):
    model = GAP_ConvMixer(32, 4, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def gap_convmixer_256_8(pretrained=False, **kwargs):
    model = GAP_ConvMixer(256, 8, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_16_4(pretrained=False, **kwargs):
    model = ConvMixer(16, 4, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_8_4(pretrained=False, **kwargs):
    model = ConvMixer(8, 4, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def embmixer_128_4(pretrained=False, **kwargs):
    model = EmbMixer(128, 4, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer2_128_4(pretrained=False, **kwargs):
    model = ConvMixer2(128, 4, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_np_128_4(pretrained=False, **kwargs):
    model = ConvMixer_nopooling(128, 4, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_bothpath_v2_128_4(pretrained=False, **kwargs):
    model = BothPath_v2(128, 4, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

@register_model
def convmixer_bothpath_v5_128_12(pretrained=False, **kwargs):
    model = BothPath_v5(128, 12, kernel_size=8, patch_size=1, n_classes=10)
    model.default_cfg = _cfg
    return model

# if __name__ == "__main__":
#     print(1)