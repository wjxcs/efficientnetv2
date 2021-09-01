CFG = {
    'in_ch': 3,
    'out_ch': 24,
    'kernel_size': 3,
    'stride': 2,
    'width_mult': 1,
    'divisor': 8,
    'actn_layer': None,
    'layers': [
        {'channels': 24, 'expansion': 1, 'kernel_size': 3, 'stride': 1,
         'nums': 2, 'norm_layer': None, 'dropout_ratio': 0.1, 'dc_ratio': 0.2,
         'reduction_ratio': 0.25, 'actn_layer': None, 'fused': True,
         'use_se': False},
        {'channels': 48, 'expansion': 4, 'kernel_size': 3, 'stride': 2,
         'nums': 4, 'norm_layer': None, 'dropout_ratio': 0.1, 'dc_ratio': 0.2,
         'reduction_ratio': 0.25, 'actn_layer': None, 'fused': True,
         'use_se': False},
        {'channels': 64, 'expansion': 4, 'kernel_size': 3, 'stride': 2,
         'nums': 4, 'norm_layer': None, 'dropout_ratio': 0.1, 'dc_ratio': 0.2,
         'reduction_ratio': 0.25, 'actn_layer': None, 'fused': True,
         'use_se': False},
        {'channels': 128, 'expansion': 4, 'kernel_size': 3, 'stride': 2,
         'nums': 6, 'norm_layer': None, 'dropout_ratio': 0.1, 'dc_ratio': 0.2,
         'reduction_ratio': 0.25, 'actn_layer': None, 'fused': False,
         'use_se': True},
        {'channels': 160, 'expansion': 6, 'kernel_size': 3, 'stride': 1,
         'nums': 9, 'norm_layer': None, 'dropout_ratio': 0.1, 'dc_ratio': 0.2,
         'reduction_ratio': 0.25, 'actn_layer': None, 'fused': False,
         'use_se': True},
        {'channels': 256, 'expansion': 6, 'kernel_size': 3, 'stride': 2,
         'nums': 15, 'norm_layer': None, 'dropout_ratio': 0.1, 'dc_ratio': 0.2,
         'reduction_ratio': 0.25, 'actn_layer': None, 'fused': False,
         'use_se': True}
    ]
}


def get_default_cfg():
    return CFG


def get_cfg(name='efficientnetv2_s'):
    name = name.lower()
    if name == 'efficientnetv2_s':
        cfg = get_default_cfg()
    elif name == 'efficientnetv2_m':
        cfg = get_default_cfg()
        cfg['layers'] = [
            {'channels': 24, 'expansion': 1, 'kernel_size': 3, 'stride': 1,
             'nums': 3, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': True, 'use_se': False},
            {'channels': 48, 'expansion': 4, 'kernel_size': 3, 'stride': 2,
             'nums': 5, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': True, 'use_se': False},
            {'channels': 80, 'expansion': 4, 'kernel_size': 3, 'stride': 2,
             'nums': 5, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': True, 'use_se': False},
            {'channels': 160, 'expansion': 4, 'kernel_size': 3, 'stride': 2,
             'nums': 7, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': False, 'use_se': True},
            {'channels': 176, 'expansion': 6, 'kernel_size': 3, 'stride': 1,
             'nums': 14, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': False, 'use_se': True},
            {'channels': 304, 'expansion': 6, 'kernel_size': 3, 'stride': 2,
             'nums': 18, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': False, 'use_se': True},
            {'channels': 512, 'expansion': 6, 'kernel_size': 3, 'stride': 1,
             'nums': 5, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': False, 'use_se': True}
        ]
    elif name == 'efficientnetv2_l':
        cfg = get_default_cfg()
        cfg['layers'] = [
            {'channels': 32, 'expansion': 1, 'kernel_size': 3, 'stride': 1,
             'nums': 4, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': True, 'use_se': False},
            {'channels': 64, 'expansion': 4, 'kernel_size': 3, 'stride': 2,
             'nums': 7, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': True, 'use_se': False},
            {'channels': 96, 'expansion': 4, 'kernel_size': 3, 'stride': 2,
             'nums': 7, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': True, 'use_se': False},
            {'channels': 192, 'expansion': 4, 'kernel_size': 3, 'stride': 2,
             'nums': 10, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': False, 'use_se': True},
            {'channels': 224, 'expansion': 6, 'kernel_size': 3, 'stride': 1,
             'nums': 19, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': False, 'use_se': True},
            {'channels': 384, 'expansion': 6, 'kernel_size': 3, 'stride': 2,
             'nums': 25, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': False, 'use_se': True},
            {'channels': 640, 'expansion': 6, 'kernel_size': 3, 'stride': 1,
             'nums': 7, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': False, 'use_se': True}
        ]
    elif name == 'efficientnetv2_xl':
        cfg = get_default_cfg()
        cfg['layers'] = [
            {'channels': 32, 'expansion': 1, 'kernel_size': 3, 'stride': 1,
             'nums': 4, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': True, 'use_se': False},
            {'channels': 64, 'expansion': 4, 'kernel_size': 3, 'stride': 2,
             'nums': 8, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': True, 'use_se': False},
            {'channels': 96, 'expansion': 4, 'kernel_size': 3, 'stride': 2,
             'nums': 8, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': True, 'use_se': False},
            {'channels': 192, 'expansion': 4, 'kernel_size': 3, 'stride': 2,
             'nums': 16, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': False, 'use_se': True},
            {'channels': 256, 'expansion': 6, 'kernel_size': 3, 'stride': 1,
             'nums': 24, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': False, 'use_se': True},
            {'channels': 512, 'expansion': 6, 'kernel_size': 3, 'stride': 2,
             'nums': 32, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': False, 'use_se': True},
            {'channels': 640, 'expansion': 6, 'kernel_size': 3, 'stride': 1,
             'nums': 8, 'norm_layer': None, 'dropout_ratio': 0.1,
             'dc_ratio': 0.2, 'reduction_ratio': 0.25, 'actn_layer': None,
             'fused': False, 'use_se': True}
        ]
    else:
        raise ValueError("No pretrained config available"
                         " for name {}".format(name))
    return cfg
