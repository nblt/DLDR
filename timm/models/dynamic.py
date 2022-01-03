import sys
import torch.nn as nn
sys.path.append('../../../')
import dynamic
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model

_cfg = {
    'url': '',
    'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
    'crop_pct': .96, 'interpolation': 'bicubic',
    'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head'
}

@register_model
def fc_30_10_relu(pretrained=False, **kwargs):
    model = dynamic.FC(hidden_features=30, hidden_layers=10, act_layer=nn.ReLU())
    model.default_cfg = _cfg
    return model

@register_model
def fc_30_10_silu(pretrained=False, **kwargs):
    model = dynamic.FC(hidden_features=30, hidden_layers=10, act_layer=nn.SiLU())
    model.default_cfg = _cfg
    return model

@register_model
def resfc_30_10_relu(pretrained=False, **kwargs):
    model = dynamic.ResFC(hidden_features=30, hidden_layers=10, act_layer=nn.ReLU())
    model.default_cfg = _cfg
    return model

@register_model
def resfc_30_10_silu(pretrained=False, **kwargs):
    model = dynamic.ResFC(hidden_features=30, hidden_layers=10, act_layer=nn.SiLU())
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_fc_10_8_relu(pretrained=False, **kwargs):
    model = dynamic.dynamic_FC(hidden_features=10, hidden_layers=8, act_layer=nn.ReLU(), bias=True)
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_fc_10_8_silu(pretrained=False, **kwargs):
    model = dynamic.dynamic_FC(hidden_features=10, hidden_layers=8, act_layer=nn.SiLU(), bias=True)
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_resfc_10_32_relu(pretrained=False, **kwargs):
    model = dynamic.dynamic_ResFC(hidden_features=10, hidden_layers=32, act_layer=nn.ReLU(), bias=True)
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_resfc_10_32_silu(pretrained=False, **kwargs):
    model = dynamic.dynamic_ResFC(hidden_features=10, hidden_layers=32, act_layer=nn.SiLU(), bias=True)
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_fc_10_2_silu(pretrained=False, **kwargs):
    model = dynamic.dynamic_FC(hidden_features=10, hidden_layers=2, act_layer=nn.SiLU(), bias=True)
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_fc2_10_2_silu(pretrained=False, **kwargs):
    model = dynamic.dynamic_FC2(hidden_features=10, hidden_layers=2, act_layer=nn.SiLU(), bias=True)
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_fc_10_2_silu_nobias(pretrained=False, **kwargs):
    model = dynamic.dynamic_FC(hidden_features=10, hidden_layers=2, act_layer=nn.SiLU(), bias=False)
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_fc_v2_10_4_silu(pretrained=False, **kwargs):
    model = dynamic.dynamic_FC_v2(hidden_features=10, hidden_layers=4, act_layer=nn.SiLU(), bias=False)
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_resfc_v2_10_4_silu(pretrained=False, **kwargs):
    model = dynamic.dynamic_ResFC_v2(hidden_features=10, hidden_layers=4, act_layer=nn.SiLU(), bias=False)
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_resfc_v2_10_32_silu(pretrained=False, **kwargs):
    model = dynamic.dynamic_ResFC_v2(hidden_features=10, hidden_layers=32, act_layer=nn.SiLU(), bias=False)
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_resfc_v2_10_32_silu_val(pretrained=False, **kwargs):
    model = dynamic.dynamic_ResFC_v2(hidden_features=10, hidden_layers=32, act_layer=nn.SiLU(), bias=False, mode='val')
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_fc_v3_10_4_silu(pretrained=False, **kwargs):
    model = dynamic.dynamic_FC_v3(hidden_features=10, hidden_layers=4, act_layer=nn.SiLU(), bias=False)
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_fc_v3_10_4_silu_val(pretrained=False, **kwargs):
    model = dynamic.dynamic_FC_v3(hidden_features=10, hidden_layers=4, act_layer=nn.SiLU(), bias=False, mode='val')
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_resfc_v3_10_4_silu(pretrained=False, **kwargs):
    model = dynamic.dynamic_ResFC_v3(hidden_features=10, hidden_layers=4, act_layer=nn.SiLU(), bias=False)
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_resfc_v3_10_4_silu_val(pretrained=False, **kwargs):
    model = dynamic.dynamic_ResFC_v3(hidden_features=10, hidden_layers=4, act_layer=nn.SiLU(), bias=False, mode='val')
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_resfc_v3_10_32_silu(pretrained=False, **kwargs):
    model = dynamic.dynamic_ResFC_v3(hidden_features=10, hidden_layers=32, act_layer=nn.SiLU(), bias=False)
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_fc_v4_10_4_silu(pretrained=False, **kwargs):
    model = dynamic.dynamic_FC_v4(hidden_features=10, hidden_layers=4, act_layer=nn.SiLU(), bias=False)
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_resfc_v4_10_4_silu(pretrained=False, **kwargs):
    model = dynamic.dynamic_ResFC_v4(hidden_features=10, hidden_layers=4, act_layer=nn.SiLU(), bias=False)
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_fc_v5_10_4_silu(pretrained=False, **kwargs):
    model = dynamic.dynamic_FC_v5(hidden_features=10, hidden_layers=4, act_layer=nn.SiLU(), bias=False)
    model.default_cfg = _cfg
    return model

@register_model
def dynamic_resfc_v5_10_4_silu(pretrained=False, **kwargs):
    model = dynamic.dynamic_ResFC_v5(hidden_features=10, hidden_layers=4, act_layer=nn.SiLU(), bias=False)
    model.default_cfg = _cfg
    return model