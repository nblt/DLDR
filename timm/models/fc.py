import sys
sys.path.append('../../../')
import fc1
from fc1 import deconv_toy8, deconv_toy9, deconv_toy10, deconv_toy11
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model


_cfg = {
    'url': '',
    'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
    'crop_pct': .96, 'interpolation': 'bicubic',
    'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head'
}

@register_model
def deconv_toy8_500_8_5(pretrained=False, **kwargs):
    model = deconv_toy8(dim=500, depth=8, num=5)
    model.default_cfg = _cfg
    return model

@register_model
def deconv_toy9_500_8_5(pretrained=False, **kwargs):
    model = deconv_toy9(dim=500, depth=8, num=5)
    model.default_cfg = _cfg
    return model

@register_model
def deconv_toy10_500_8_5(pretrained=False, **kwargs):
    model = deconv_toy10(dim=500, depth=8, num=5)
    model.default_cfg = _cfg
    return model

@register_model
def deconv_toy10_1000_8_5(pretrained=False, **kwargs):
    model = deconv_toy10(dim=1000, depth=8, num=5)
    model.default_cfg = _cfg
    return model

@register_model
def deconv_toy10_500_8_5_05(pretrained=False, **kwargs):
    model = deconv_toy10(dim=500, depth=8, num=5, clip=0.5)
    model.default_cfg = _cfg
    return model

@register_model
def deconv_toy11_500_8_5(pretrained=False, **kwargs):
    model = deconv_toy11(dim=500, depth=8, num=5)
    model.default_cfg = _cfg
    return model