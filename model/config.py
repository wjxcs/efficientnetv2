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
