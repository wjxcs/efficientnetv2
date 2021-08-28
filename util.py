from itertools import repeat
from typing import Union, Sequence
import torch

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs


# From PyTorch internals (with little change)
def ntuple(x, n):
    def parse(x, n):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse(x, n)


# Calculation as per formula given at
# https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
def calc_pad(img_size: Union[int, Sequence[int]],
             kernel_size: Union[int, Sequence[int]] = 3,
             stride: Union[int, Sequence[int]] = 1,
             dilation: Union[int, Sequence[int]] = 1):
    img_size = ntuple(img_size, 2)
    k = ntuple(kernel_size, 2)
    s = ntuple(stride, 2)
    d = ntuple(dilation, 2)

    h, w = img_size

    pad0 = (h - 1) * (s[0] - 1) + d[0] * (k[0] - 1)
    pad1 = (w - 1) * (s[1] - 1) + d[1] * (k[1] - 1)

    return (pad0//2, pad0-pad0//2, pad1//2, pad1-pad1//2)


def activation_fn(name, inplace=True):
    name = name.lower()
    if name in ('swish', 'silu'):
        return torch.nn.SiLU(inplace=inplace)
    elif name == 'relu':
        return torch.nn.ReLU(inplace=inplace)
    elif name == 'relu6':
        return torch.nn.ReLU6(inplace=inplace)
    elif name == 'elu':
        return torch.nn.ELU(inplace=inplace)
    elif name == 'leaky_relu':
        return torch.nn.LeakyReLU(inplace=inplace)
    elif name == 'selu':
        return torch.nn.SELU(inplace=inplace)
    elif name == 'mish':
        return torch.nn.Mish(inplace=inplace)
    elif name == 'hswish':
        return torch.nn.Hardswish(inplace=inplace)
    else:
        raise ValueError("Unsupported act_fn {}".format(name))


def get_actn_fn(name, inplace=True):
    if not name:
        return torch.nn.SELU(inplace=inplace)
    if isinstance(name, str):
        return activation_fn(name, inplace)
    return name
