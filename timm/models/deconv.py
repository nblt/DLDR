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
from typing import TypeVar, Optional, List, Tuple, Union
import collections
from itertools import repeat
from typing import List, Dict, Any

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))

T = TypeVar('T')
_scalar_or_tuple_any_t = Union[T, Tuple[T, ...]]
_scalar_or_tuple_1_t = Union[T, Tuple[T]]
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]

# For arguments which represent size parameters (eg, kernel size, padding)
_size_any_t = _scalar_or_tuple_any_t[int]
_size_1_t = _scalar_or_tuple_1_t[int]
_size_2_t = _scalar_or_tuple_2_t[int]


class _deConvNd(nn.Module):

    __constants__ = ['num', 'stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        ...

    _in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 num: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 weight_init='kaiming_uniform_5',
                 alpha_init='uniform_0_1',
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_deConvNd, self).__init__()
        self.num = num
        self.expend_num = 2**num
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        self.alpha = Parameter(torch.empty((num, 1, 1, 1, 1), **factory_kwargs))
        self.alpha_init = alpha_init
        self.weight_init = weight_init

        if transposed:
            self.weight = torch.empty(
                (num, in_channels, out_channels // groups, *kernel_size), **factory_kwargs).cuda()
        else:
            self.weight = torch.empty(
                (num, out_channels, in_channels // groups, *kernel_size), **factory_kwargs).cuda()
        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        if self.weight_init == 'uniform_-1_1':
            init.uniform_(self.weight, -1, 1)
        elif self.weight_init == 'normal_0_1':
            init.normal_(self.weight, 0, 1)
        elif self.weight_init == 'xavier_uniform_5':
            init.xavier_uniform_(self.weight, a=math.sqrt(5))
        elif self.weight_init == 'xavier_normal_5':
            init.xavier_normal_(self.weight, a=math.sqrt(5))
        elif self.weight_init == 'kaiming_uniform_5':
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        elif self.weight_init == 'kaiming_normal_5':
            init.kaiming_normal_(self.weight, a=math.sqrt(5))
        elif self.weight_init == 'orthogonal':
            init.orthogonal_(self.weight, gain=math.sqrt(5))
                # print(self.weight[i].mean(), self.weight[i].std())
        # elif self.weight_init == 'sparse_01':
        #     for i in range(self.expend_num):
        #         init.sparse_(self.weight[i], 0.1, 0.01)
        # elif self.weight_init == 'sparse_03':
        #     for i in range(self.expend_num):
        #         init.sparse_(self.weight[i], 0.3, 0.01)
        # elif self.weight_init == 'sparse_06':
        #     for i in range(self.expend_num):
        #         init.sparse_(self.weight[i], 0.6, 0.01)
        # elif self.weight_init == 'sparse_09':
        #     print(self.weight.shape)
        #     for i in range(self.in_channels):
        #         for j in range(self.out_channels):
        #             init.sparse_(self.weight[i][j], 0.9, 0.01)
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # init.uniform_(self.alpha, 0.5, 2)

        if self.alpha_init == 'constant_1':
            init.constant_(self.alpha, 1)
        elif self.alpha_init == 'uniform_0_1':
            init.uniform_(self.alpha, 0, 1)
        elif self.alpha_init == 'uniform_05_2':
            init.uniform_(self.alpha, 0.5, 2)
        elif self.alpha_init == 'normal_1_1':
            init.normal_(self.alpha, 1, 1)
        elif self.alpha_init == 'normal_0_1':
            init.normal_(self.alpha, 0, 1)
        elif self.alpha_init == 'lognormal':
            self.alpha.data = torch.clamp(
                torch.from_numpy(np.random.lognormal(mean=0, sigma=0.1, size=self.alpha.shape)), 0, 2)
            # print(self.alpha)
            


        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_deConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

class deConv2d(_deConvNd):

    def __init__(
        self,
        num: int,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        weight_init='kaiming_uniform_5',
        alpha_init='uniform_0_1',
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(deConv2d, self).__init__(
            num, in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, weight_init, alpha_init, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        # print("conv input", input.mean())
        # dummy_weight = torch.ones((1, 1, 1, 1, 1)).cuda()
        # for i in range(self.num):
        #     new_weight = self.alpha[i]*dummy_weight 
        #     # print("new", new_weight)
        #     dummy_weight = torch.cat([dummy_weight, new_weight])
        # # print(dummy_weight, weight)
        new_weight = torch.sum(weight*self.alpha, dim=0).float()
        # print(new_weight.shape)
        if self.padding_mode != 'zeros':
            res = F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            new_weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        else: res = F.conv2d(input, new_weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)
        # print(self.alpha)
        # print(res.mean(), res.std())
        return res

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)