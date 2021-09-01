from itertools import repeat
import torch
import torch.nn as nn
from typing import Union, Sequence


TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])

if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs


if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU(inplace=True)
else:
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


class SiLUImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        res = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        x = ctx.saved_tensors[0]
        sig_x = torch.sigmoid(x)
        return grad * (sig_x * (1 + x * (1 - sig_x)))


class MemoryEfficientSiLU(nn.Module):
    def forward(self, x):
        return SiLUImpl.apply(x)


# From:
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
def make_divisible(v, divisor, min_value=None):
    """
    This function taken from tf code repository.
    It ensures that the return value is divisible by 8.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


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


# Converted tf implementation from
# https://github.com/google/automl/blob/be5340b61e84ae998765ad3340b633fcf57da87a/efficientnetv2/utils.py#L292
# to pytorch.
def drop_connect(inputs, prob, training):
    assert 0.0 <= prob <= 1.0, "Drop probability should" \
                            + " be in range [0, 1]."

    if not training:
        return inputs

    bs = inputs.shape[0]
    keep_prob = 1 - prob

    random_tensor = keep_prob
    random_tensor += torch.rand(size=(bs, 1, 1, 1), dtype=inputs.dtype,
                                device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output
