import copy
import torch.nn as nn

from .base import Stem, Head, MBConv
from .util import make_divisible


class efficientnetv2(nn.Module):
    def __init__(self, cfg, num_classes=1000):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        out_ch = make_divisible(self.cfg['out_ch'] * self.cfg['width_mult'],
                                self.cfg['divisor'])

        features = []
        features.append(Stem(self.cfg['in_ch'], out_ch,
                             kernel_size=self.cfg['kernel_size'],
                             stride=self.cfg['stride'],
                             actn_layer=self.cfg['actn_layer'],
                             skip_init=True))

        layers = self.cfg['layers']
        input_ch = out_ch
        for layer in layers:
            out_ch = make_divisible(layer['channels'] * self.cfg['width_mult'],
                                    self.cfg['divisor'])
            ft = []
            for i in range(layer['nums']):
                stride = 1 if i == 0 else layer['stride']
                norm_layer = layer['norm_layer'] if layer['norm_layer'] \
                    else nn.BatchNorm2d
                ft.append(MBConv(input_ch, out_ch, fused=layer['fused'],
                                 expansion=layer['expansion'],
                                 kernel_size=layer['kernel_size'],
                                 stride=stride,
                                 norm_layer=norm_layer,
                                 dropout_ratio=layer['dropout_ratio'],
                                 reduction_ratio=layer['reduction_ratio'],
                                 drop_connect_ratio=layer['dc_ratio'],
                                 actn_layer=layer['actn_layer'],
                                 use_se=layer['use_se'], skip_init=True))
                input_ch = out_ch
            features.append(nn.Sequential(*ft))

        self.features = nn.Sequential(*features)
        out_ch = make_divisible(1280 * self.cfg['width_mult'],
                                self.cfg['divisor'])
        self.conv = Head(input_ch, out_ch, actn_layer=self.cfg['actn_layer'],
                         skip_init=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_ch, num_classes)
        self.init_weights()

    def init_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0, 0.001)
                module.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
