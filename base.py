import torch.nn as nn
import torch.nn.functional as F
from util import calc_pad, drop_connect
from util import make_divisible, MemoryEfficientSiLU


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
                                          Default: nn.ReLU
        norm_layer(nn.<norm_layer>, optional): Normalisation layer.
                                               If not provided any value,
                                               identity layer is added.
                                               Default: None
        skip_init(boolean, optional): Skip weight initialisation for the
                                      squeeze and excitation block.
                                      Default: False

    Examples:
        >> se = SEBlock(64)
        >> x = torch.randn((1, 64, 20, 30))
        >> output = se(x)

    """
    def __init__(self, in_ch, reduction_ratio=0.25, act_layer=nn.ReLU,
                 norm_layer=None, skip_init=False, use_avgpool=False):
        super().__init__()
        reduced_channels = make_divisible(in_ch * reduction_ratio, 8)
        self.actn = act_layer(inplace=True)
        self.k = 1
        self.s = 1
        self.d = 1
        self.squeeze = nn.Conv2d(in_ch, reduced_channels, kernel_size=self.k,
                                 stride=self.s, dilation=self.d, bias=True)
        self.excite = nn.Conv2d(reduced_channels, in_ch, kernel_size=self.k,
                                stride=self.s, dilation=self.d, bias=True)
        self.norm_layer = norm_layer if norm_layer is not None \
            else nn.Identity()
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
            b, c, _, _ = x.size()
            x = self.pool(x).view(b, c)
        else:
            x = x.mean((2, 3), keepdim=True)
        x = self.squeeze(x)
        x = self.norm_layer(x)
        x = self.actn(x)
        x = self.excite(x)
        x = self.sigmoid(x)
        return orig * x


class FusedMBConv(nn.Module):
    """
    Fused MBConve : Fused Mobile Inverted Residual Bottleneck.

    The only difference betwen MBConv and Fused MBConv is Fused MBConv.
    uses 3*3 conv2d layer inplace of 1*1 conv2d + 3*3 depthwise conv2d.

    Args:
        in_ch(int): Input channels.
        out_ch(int): output channels.
        expansion(int, optional): Expansion ratio to be used to expand number
                                  of channels in conv layers.
                                  Default: 4
        kernel_size(int, optional): Kernel size for conv layer.
                                   Default: 3
        stride(int, optional): Stride for conv layer.
                               Default: 1
        norm_layer(nn.<norm_layer>, optional): Normalisation layer.
                                               Default: nn.BatchNorm2d
        dropout_ratio(float, optional): Dropout ratio within range [0, 1]
                                        Default: 0.0
        reduction_ratio(float, optional): The reduction ratio used to
                                          calculate out channels in squeeze
                                          conv layer to nearest integer value.
                                          Value is between 0. and 1.0.
                                          Default: 0.25
        drop_connect_ratio(float, optional): Drop probability for drop connect.
                                             Default: 0.0
        act_layer(nn.<activation_layer>, optional):
                                          Activation layer.
                                          Default: MemoryEfficientSiLU
        use_se(boolean, optional): Use Squeeze and Excitation layer.
                                   If True, SEBlock is added to model.
                                   If False, Identity layer is used instead.
                                   Default:True
        skip_init(boolean, optional): Skip weight initialisation for the
                                      squeeze and excitation block.
                                      Default: False

    Examples:
        >> se = SEBlock(64)
        >> x = torch.randn((1, 64, 20, 30))
        >> output = se(x)

    """
    def __init__(self, in_ch, out_ch, expansion=4, kernel_size=3, stride=1,
                 norm_layer=nn.BatchNorm2d, dropout_ratio=0.0,
                 reduction_ratio=0.25, drop_connect_ratio=0.0,
                 actn_layer=MemoryEfficientSiLU,
                 use_se=True, skip_init=False):
        super().__init__()
        self.expansion = expansion
        hidden_ch = in_ch * self.expansion
        self.actn = actn_layer()
        self.identity = stride == 1 and in_ch == out_ch

        self.k1 = kernel_size
        self.s1 = stride
        self.d = 1

        if dropout_ratio != 0.0:
            self.dropout = nn.Dropout()
        else:
            self.dropout = nn.Identity()

        self.drop_connect_prob = drop_connect_ratio

        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=kernel_size,
                               stride=stride, bias=False)
        self.bn1 = norm_layer(hidden_ch)

        if self.expansion == 1:
            kernel_size, stride = 1, 1

        self.k2 = kernel_size
        self.s2 = stride

        self.conv2 = nn.Conv2d(hidden_ch, out_ch, kernel_size=kernel_size,
                               stride=stride, bias=False)
        self.bn2 = norm_layer(out_ch)

        if use_se:
            self.se = SEBlock(out_ch, reduction_ratio=reduction_ratio,
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
        x = self.dropout(x)
        x = self.se(x)

        pad = calc_pad((x.shape[2], x.shape[3]), self.k2, self.s2, self.d)
        x = F.pad(x, pad, mode='reflect')
        x = self.bn2(self.conv2(x))
        if self.expansion == 1:
            x = self.actn(x)

        if self.identity:
            if self.drop_connect_prob > 0.0:
                x = drop_connect(x, self.drop_connect_prob, self.training)
            return orig + x
        else:
            return x
