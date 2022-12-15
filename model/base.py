import torch.nn as nn
import torch.nn.functional as F

from .util import calc_pad, drop_connect
from .util import make_divisible, MemoryEfficientSiLU


class Stem(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1,
                 actn_layer=None, skip_init=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                              stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        if actn_layer:
            self.actn = actn_layer
        else:
            self.actn = MemoryEfficientSiLU()

        self.k = kernel_size
        self.s = stride
        self.d = 1

        self.init_weights(skip_init)

    def init_weights(self, skip=False):
        if not skip:
            for _, module in self.named_modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                            nonlinearity='relu')
                if isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        pad = calc_pad((x.shape[2], x.shape[3]), self.k, self.s, self.d)
        x = F.pad(x, pad, mode='reflect')
        return self.actn(self.bn(self.conv(x)))


class Head(nn.Module):
    def __init__(self, in_ch, out_ch, actn_layer=None, skip_init=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1,
                              stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        if actn_layer:
            self.actn = actn_layer
        else:
            self.actn = MemoryEfficientSiLU()

        self.k, self.s, self.d = 1, 1, 1
        self.init_weights(skip_init)

    def init_weights(self, skip=False):
        if not skip:
            for _, module in self.named_modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                            nonlinearity='relu')
                if isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        pad = calc_pad((x.shape[2], x.shape[3]), self.k, self.s, self.d)
        x = F.pad(x, pad, mode='reflect')
        return self.actn(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.

    Args:
        in_ch(int): Input channels.
        reduction_ratio(float, optional): The reduction ratio used to
                                          calculate out channels in squeeze
                                          conv layer to nearest integer value.
                                          Value is between 0. and 1.0.
                                          Default: 0.25
        act_layer(nn.<activation_layer>, optional):
                                          Activation layer.
                                          If None, default is nn.ReLU
                                          Default: None
        skip_init(boolean, optional): Skip weight initialisation for the
                                      squeeze and excitation block.
                                      Default: False

    Examples:
        >> se = SEBlock(64)
        >> x = torch.randn((1, 64, 20, 30))
        >> output = se(x)

    """
    def __init__(self, in_ch, reduction_ratio=0.25, act_layer=None,
                 skip_init=False, use_avgpool=False):
        super().__init__()
        reduced_channels = make_divisible(in_ch * reduction_ratio, 8)
        if act_layer:
            self.actn = act_layer
        else:
            self.actn = nn.ReLU(inplace=True)

        self.k = 1
        self.s = 1
        self.d = 1
        self.squeeze = nn.Conv2d(in_ch, reduced_channels, kernel_size=self.k,
                                 stride=self.s, dilation=self.d, bias=True)
        self.excite = nn.Conv2d(reduced_channels, in_ch, kernel_size=self.k,
                                stride=self.s, dilation=self.d, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.use_avgpool = use_avgpool
        if use_avgpool:
            self.pool = nn.AdaptiveAvgPool2d(1)
        self.init_weights(skip_init)

    def init_weights(self, skip=False):
        if not skip:
            for _, module in self.named_modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                            nonlinearity='relu')
                if isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        orig = x
        pad = calc_pad((x.shape[2], x.shape[3]), self.k, self.s, self.d)
        x = F.pad(x, pad, mode='reflect')
        if self.use_avgpool:
            x = self.pool(x)
        else:
            x = x.mean((2, 3), keepdim=True)
        x = self.squeeze(x)
        x = self.actn(x)
        x = self.excite(x)
        x = self.sigmoid(x)
        return orig * x


class MBConv(nn.Module):
    """
    MBConv : Mobile Inverted Residual Bottleneck.

    Implementation of MBConv block.
    If fused option is set, Fussed MBConv is block is created.
    The only difference betwen MBConv and Fused MBConv is Fused MBConv.
    uses 3*3 conv2d layer inplace of 3*3 depthwise conv2d followed by 1*1
    conv.

    Args:
        in_ch(int):                            Input channels.

        out_ch(int):                           Output channels.

        fused(boolean, optional):              Create Fused MBConv block.
                                               Default: False

        expansion(int, optional):              Expansion ratio to be used to expand number
                                               of channels in conv layers.
                                               Default: 4

        kernel_size(int, optional):            Kernel size for conv layer.
                                               Default: 3

        stride(int, optional):                 Stride for conv layer.
                                               Default: 1

        norm_layer(nn.<norm_layer>, optional): Normalisation layer.
                                               Default: nn.BatchNorm2d

        dropout_ratio(float, optional):        Dropout ratio within range [0, 1]
                                               Default: 0.0

        reduction_ratio(float, optional):      The reduction ratio used to
                                               calculate out channels in squeeze
                                               conv layer to nearest integer value.
                                               Value is between 0. and 1.0.
                                               Default: 0.25

        drop_connect_ratio(float, optional):   Drop probability for drop connect.
                                               Default: 0.0

        act_layer(nn.<activation_layer>, optional):
                                               Activation layer.
                                               If input is None, we default to
                                               MemoryEfficientSiLU.
                                               Default: None

        use_se(boolean, optional):             Use Squeeze and Excitation layer.
                                               If True, SEBlock is added to model.
                                               If False, Identity layer is used instead.
                                               Default:True

        skip_init(boolean, optional):          Skip weight initialisation for the
                                               squeeze and excitation block.
                                               Default: False

    Examples:
        >> conv = MBConv(64, 64)
        >> x = torch.randn((1, 64, 20, 30))
        >> output = conv(x)

    """
    def __init__(self, in_ch: int, out_ch: int, fused: bool=False, expansion: int=4,
                 kernel_size: int=3, stride: int=1, norm_layer=nn.BatchNorm2d, dropout_ratio: float=0.0,
                 reduction_ratio: float=0.25, drop_connect_ratio: float=0.0, actn_layer=None,
                 use_se: bool=True, skip_init: bool=False) -> None:
        super().__init__()
        self.expansion = expansion
        hidden_ch = in_ch * self.expansion
        if actn_layer:
            self.actn = actn_layer
        else:
            self.actn = MemoryEfficientSiLU()
        self.identity = stride == 1 and in_ch == out_ch
        self.d = 1

        if dropout_ratio != 0.0:
            self.dropout = nn.Dropout()
        else:
            self.dropout = nn.Identity()

        self.drop_connect_prob = drop_connect_ratio

        self.fused = fused

        if self.fused:
            self.k1 = kernel_size
            self.s1 = stride
            self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=kernel_size,
                                   stride=stride, bias=False)
            self.bn1 = norm_layer(hidden_ch)
        else:
            self.k1 = 1
            self.s1 = 1
            self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=1,
                                   stride=1, bias=False)
            self.bn1 = norm_layer(hidden_ch)

            # Depthwise conv
            self.k2 = kernel_size
            self.s2 = stride
            self.conv2 = nn.Conv2d(hidden_ch, hidden_ch,
                                   kernel_size=kernel_size, stride=stride,
                                   groups=hidden_ch, bias=False)
            self.bn2 = norm_layer(hidden_ch)

        if self.expansion == 1:
            kernel_size, stride = 1, 1

        self.k3 = kernel_size
        self.s3 = stride

        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=kernel_size,
                               stride=stride, bias=False)
        self.bn3 = norm_layer(out_ch)

        if use_se:
            self.se = SEBlock(hidden_ch, reduction_ratio=reduction_ratio,
                              act_layer=self.actn)
        else:
            self.se = nn.Identity()

        self.init_weights(skip_init)

    def init_weights(self, skip=False):
        if not skip:
            for _, module in self.named_modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                            nonlinearity='relu')
                if isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        orig = x
        pad = calc_pad((x.shape[2], x.shape[3]), self.k1, self.s1, self.d)
        x = F.pad(x, pad, mode='reflect')
        x = self.actn(self.bn1(self.conv1(x)))

        if not self.fused:
            pad = calc_pad((x.shape[2], x.shape[3]), self.k2, self.s2, self.d)
            x = F.pad(x, pad, mode='reflect')
            x = self.actn(self.bn2(self.conv2(x)))

        x = self.dropout(x)
        x = self.se(x)

        pad = calc_pad((x.shape[2], x.shape[3]), self.k3, self.s3, self.d)
        x = F.pad(x, pad, mode='reflect')
        x = self.bn3(self.conv3(x))
        if self.expansion == 1:
            x = self.actn(x)

        if self.identity:
            if self.drop_connect_prob > 0.0:
                x = drop_connect(x, self.drop_connect_prob, self.training)
            return orig + x
        return x
