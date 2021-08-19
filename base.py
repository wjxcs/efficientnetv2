import torch.nn as nn


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
        skip(boolean, optional): Skip weight initialisation for the squeeze and
                                 excitation block.
                                 Default: False

    Examples:
        >> se = SEBlock(64)
        >> x = torch.randn((1, 64, 20, 30))
        >> output = se(x)

    """
    def __init__(self, in_ch, reduction_ratio=0.25, act_layer=nn.ReLU,
                 skip=False):
        super().__init__()
        reduced_channels = int(in_ch * reduction_ratio)
        self.actn = act_layer(inplace=True)
        self.squeeze = nn.Conv2d(in_ch, reduced_channels, kernel_size=1,
                                 stride=1, bias=True)
        self.excite = nn.Conv2d(reduced_channels, in_ch, kernel_size=1,
                                stride=1, bias=True)
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
        orig = x
        x = x.mean((2, 3), keepdim=True)
        x = self.squeeze(x)
        x = self.actn(x)
        x = self.excite(x)
        x = self.sigmoid(x)
        return orig * x
