"""dense net in pytorch



[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.

    Densely Connected Convolutional Networks
    https://arxiv.org/abs/1608.06993v5
"""
import math
import sys
sys.path.append('../../../')
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
import torch
import torch.nn as nn
from torchstat import stat

_cfg = {
    'url': '',
    'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
    'crop_pct': .96, 'interpolation': 'bicubic',
    'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head'
}

#"""Bottleneck layers. Although each layer only produces k
#output feature-maps, it typically has many more inputs. It
#has been noted in [37, 11] that a 1×1 convolution can be in-
#troduced as bottleneck layer before each 3×3 convolution
#to reduce the number of input feature-maps, and thus to
#improve computational efficiency."""
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        #"""In  our experiments, we let each 1×1 convolution
        #produce 4k feature-maps."""
        inner_channel = 4 * growth_rate

        #"""We find this design especially effective for DenseNet and
        #we refer to our network with such a bottleneck layer, i.e.,
        #to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` ,
        #as DenseNet-B."""
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)

class BottleneckWithoutCat(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        #"""In  our experiments, we let each 1×1 convolution
        #produce 4k feature-maps."""
        inner_channel = 4 * growth_rate

        #"""We find this design especially effective for DenseNet and
        #we refer to our network with such a bottleneck layer, i.e.,
        #to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` ,
        #as DenseNet-B."""
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.bottle_neck(x)

def create_index_map(growth_rate, in_channels, nblocks):
    index_map = {
        i:{"features_slice":[i], "in_channels":in_channels if i==0 else growth_rate} 
        for i in range(nblocks+1)
    }
    segment_num = int(math.log(nblocks, 2))
    interval = 1
    for i in range(segment_num+1):
        for index in range(0, nblocks+1, interval):
            if index == 0: continue
            index_map[index]["features_slice"].append(index-interval)
            index_map[index]["in_channels"] += in_channels if index-interval==0 else growth_rate
        interval *= 2
    return index_map

class SegmentDenseLayers(nn.Module):
    def __init__(self, block, growth_rate, in_channels, nblocks):
        super().__init__()
        assert nblocks&(nblocks-1) == 0 # 判断是否是2的幂次
        self.growth_rate = growth_rate
        self.in_channels = in_channels
        self.nblocks = nblocks
        self.index_map = create_index_map(growth_rate, in_channels, nblocks)
        self.out_channels = self.index_map[self.nblocks]["in_channels"]
        #    print(self.index_map)
        self.dense_block = nn.ModuleList()
        for index in range(nblocks):
            self.dense_block.append(block(self.index_map[index]["in_channels"], growth_rate))

    def forward(self, x):
        features = [x]
        for index in range(self.nblocks):
            cur_input = torch.cat([features[i] for i in self.index_map[index]["features_slice"]], 1)
            # print(cur_input.shape)
            new_features = self.dense_block[index](cur_input)
            features.append(new_features)
        output = torch.cat([features[i] for i in self.index_map[self.nblocks]["features_slice"]], 1)
        return output

class DynamicDenseLayers(nn.Module):
    def __init__(self, block, growth_rate, in_channels, nblocks, quota):
        super().__init__()
        self.growth_rate = growth_rate
        self.in_channels = in_channels
        self.nblocks = nblocks
        self.quota = quota 
        self.out_channels = in_channels+growth_rate*min(quota+1, nblocks)
        #    print(self.index_map)
        self.dense_block = nn.ModuleList()
        for index in range(nblocks):
            # print(growth_rate*min(quota+1, index))
            self.dense_block.append(block(in_channels+growth_rate*min(quota+1, index), growth_rate))

    def forward(self, x):
        input = x
        cand_features = []
        cand_mean = []
        new_features = None
        for index in range(self.nblocks):
            features = [input]
            # print(len(cand_features))
            if len(cand_features) > 0 and len(cand_mean) > 0:
                _, indices = torch.topk(torch.tensor(cand_mean), min(len(cand_mean), self.quota))
                features = features+[cand_features[i] for i in indices]

            if new_features is not None:
                features.append(new_features)
                cand_features.append(new_features)
                cand_mean.append(new_features.mean().cpu().float())
            
            features = torch.cat(features, 1)
            # print(features.shape)
            new_features = self.dense_block[index](features)

        output = [input]
        if len(cand_features) > 0 and len(cand_mean) > 0:
            _, indices = torch.topk(torch.tensor(cand_mean), min(len(cand_mean), self.quota))
            output = output+[cand_features[i] for i in indices]

        if new_features is not None:
            output.append(new_features)
            cand_features.append(new_features)

        output = torch.cat(output, 1)
        return output

#"""We refer to layers between blocks as transition
#layers, which do convolution and pooling."""
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #"""The transition layers used in our experiments
        #consist of a batch normalization layer and an 1×1
        #convolutional layer followed by a 2×2 average pooling
        #layer""".
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100):
        super().__init__()
        self.growth_rate = growth_rate

        #"""Before entering the first dense block, a convolution
        #with 16 (or twice the growth rate for DenseNet-BC)
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        #For convolutional layers with kernel size 3×3, each
        #side of the inputs is zero-padded by one pixel to keep
        #the feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module("dense_block_layer_{}".format(index), self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            #"""If a dense block contains m feature-maps, we let the
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_layer_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module("dense_block{}".format(len(nblocks) - 1), self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module('bottle_neck_layer_{}'.format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

def densenet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def densenet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def densenet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def densenet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

@register_model
def densenet_bc_100_12(pretrained=False, **kwargs):
    model = DenseNet(Bottleneck, [16,16,16], growth_rate=12, num_class=10)
    model.default_cfg = _cfg
    return model

@register_model
def densenet_bc_196_12(pretrained=False, **kwargs):
    model = DenseNet(Bottleneck, [32,32,32], growth_rate=12, num_class=10)
    model.default_cfg = _cfg
    return model

@register_model
def densenet_bc_388_12(pretrained=False, **kwargs):
    model = DenseNet(Bottleneck, [64,64,64], growth_rate=12, num_class=10)
    model.default_cfg = _cfg
    return model

@register_model
def densenet_bc_121_32(pretrained=False, **kwargs):
    model = DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, num_class=10)
    model.default_cfg = _cfg
    return model

@register_model
def densenet_bc_169_32(pretrained=False, **kwargs):
    model = DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, num_class=10)
    model.default_cfg = _cfg
    return model

@register_model
def densenet_bc_201_32(pretrained=False, **kwargs):
    model = DenseNet(Bottleneck, [6,12,48,32], growth_rate=32, num_class=10)
    model.default_cfg = _cfg
    return model

class SegmentDenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100):
        super().__init__()
        self.growth_rate = growth_rate

        #"""Before entering the first dense block, a convolution
        #with 16 (or twice the growth rate for DenseNet-BC)
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        #For convolutional layers with kernel size 3×3, each
        #side of the inputs is zero-padded by one pixel to keep
        #the feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            new_layer = SegmentDenseLayers(block, growth_rate, inner_channels, nblocks[index])
            self.features.add_module("segment_dense_{}".format(index), new_layer)
            inner_channels = new_layer.out_channels

            #"""If a dense block contains m feature-maps, we let the
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        new_layer = SegmentDenseLayers(block, growth_rate, inner_channels, nblocks[len(nblocks)-1])
        self.features.add_module("segment_dense_{}".format(len(nblocks) - 1), new_layer)
        inner_channels = new_layer.out_channels
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output

@register_model
def segment_densenet_bc_100_12(pretrained=False, **kwargs):
    model = SegmentDenseNet(BottleneckWithoutCat, [16,16,16], growth_rate=12, num_class=10)
    model.default_cfg = _cfg
    return model

@register_model
def segment_densenet_bc_196_12(pretrained=False, **kwargs):
    model = SegmentDenseNet(BottleneckWithoutCat, [32,32,32], growth_rate=12, num_class=10)
    model.default_cfg = _cfg
    return model

@register_model
def segment_densenet_bc_388_12(pretrained=False, **kwargs):
    model = SegmentDenseNet(BottleneckWithoutCat, [64,64,64], growth_rate=12, num_class=10)
    model.default_cfg = _cfg
    return model

@register_model
def segment_densenet_bc_121_32(pretrained=False, **kwargs):
    model = SegmentDenseNet(BottleneckWithoutCat, [8,16,32,16], growth_rate=32, num_class=10)
    model.default_cfg = _cfg
    return model

@register_model
def segment_densenet_bc_169_32(pretrained=False, **kwargs):
    model = SegmentDenseNet(BottleneckWithoutCat, [8,16,32,32], growth_rate=32, num_class=10)
    model.default_cfg = _cfg
    return model

@register_model
def segment_densenet_bc_201_32(pretrained=False, **kwargs):
    model = SegmentDenseNet(BottleneckWithoutCat, [8,16,64,32], growth_rate=32, num_class=10)
    model.default_cfg = _cfg
    return model


class DynamicDenseNet(nn.Module):
    def __init__(self, block, nblocks, quota, growth_rate=12, reduction=0.5, num_class=10):
        super().__init__()
        self.growth_rate = growth_rate
        self.quota = quota
        #"""Before entering the first dense block, a convolution
        #with 16 (or twice the growth rate for DenseNet-BC)
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        #For convolutional layers with kernel size 3×3, each
        #side of the inputs is zero-padded by one pixel to keep
        #the feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            new_layer = DynamicDenseLayers(block, growth_rate, inner_channels, nblocks[index], quota)
            self.features.add_module("dynamic_dense_{}".format(index), new_layer)
            inner_channels = new_layer.out_channels

            #"""If a dense block contains m feature-maps, we let the
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.add_module("transition_{}".format(index), Transition(inner_channels, out_channels))
            inner_channels = out_channels

        new_layer = DynamicDenseLayers(block, growth_rate, inner_channels, nblocks[len(nblocks)-1], quota)
        self.features.add_module("dynamic_dense_{}".format(len(nblocks) - 1), new_layer)
        inner_channels = new_layer.out_channels
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(inner_channels, num_class)

    def forward(self, x):
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output)
        output = output.view(output.size()[0], -1)
        output = self.linear(output)
        return output

@register_model
def dynamic_densenet_bc_100_12_2(pretrained=False, **kwargs):
    model = DynamicDenseNet(BottleneckWithoutCat, [16,16,16], growth_rate=12, quota=2, num_class=100)
    model.default_cfg = _cfg
    return model