import torch.nn as nn
import torch.nn.functional as F
from util import calc_pad


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.

    Args:
        in_ch(float): Input channels.
        reduction_ratio(float, optional): The reduction ratio used to
                                          calculate out channels in squeeze
                                          conv layer to nearest integer value.
                                          Value is between 0. and 1.0.
                                          Default: 0.25
        act_layer(nn.<activation_layer>, optional):
                                          Activation layer.
                                          Default: nn.ReLU
        norm_layer(nn.<norm_layer>, optional): normalisation layer.
                                               If not provided any value,
                                               identity layer is added.
                                               Default: None
        skip(boolean, optional): Skip weight initialisation for the squeeze and
                                 excitation block.
                                 Default: False

    Examples:
        >> se = SEBlock(64)
        >> x = torch.randn((1, 64, 20, 30))
        >> output = se(x)

    """
    def __init__(self, in_ch, reduction_ratio=0.25, act_layer=nn.ReLU,
                 norm_layer=None, skip=False):
        super().__init__()
        reduced_channels = max(1, int(in_ch * reduction_ratio))
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
        self.init_weights(skip)

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
        orig = x
        x = x.mean((2, 3), keepdim=True)
        x = self.squeeze(x)
        x = self.norm_layer(x)
        x = self.actn(x)
        x = self.excite(x)
        x = self.sigmoid(x)
        return orig * x
