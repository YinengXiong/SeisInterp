import math

import torch
import torch.nn as nn

def get_norm(num_features, norm_type):
    if norm_type == 'BN':
        return nn.BatchNorm2d(num_features)
    elif norm_type == 'IN':
        return nn.InstanceNorm2d(num_features)

class BasicBlock(nn.Sequential):
    """
    Basic Conv-ReLU Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 bias=True, norm=None, act=nn.ReLU(inplace=True)):

        m = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                       padding=kernel_size//2, bias=bias)]
        if norm is not None:
            m.append(get_norm(out_channels, norm))

        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    """
    Residual Block
    Conv -> ReLU -> Conv
    """
    def __init__(self, num_features, kernel_size, bias=True, norm=None,
                 act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()

        m = []
        for i in range(2):
            m.append(nn.Conv2d(num_features, num_features, kernel_size,
                               stride=1, padding=kernel_size//2, bias=bias))
            if norm is not None:
                m.append(get_norm(num_features, norm))

            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

def Upblock2x(direction, num_features, upsample_type, bias):
    m = []
    if direction == 0:
        k1, k2 = 1, 4
        s1, s2 = 1, 2
        p1, p2 = 0, 1
    elif direction == 1:
        k1, k2 = 4, 1
        s1, s2 = 2, 1
        p1, p2 = 1, 0
    else:
        k1, k2 = 4, 4
        s1, s2 = 2, 2
        p1, p2 = 1, 1

    if upsample_type == 'deconv':
        m.append(nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, bias=bias))
        m.append(nn.ConvTranspose2d(num_features, num_features, kernel_size=(k1, k2),
                                    stride=(s1, s2), padding=(p1, p2)))
    else:
        m.append(nn.Conv2d(num_features, num_features * 4, kernel_size=3, stride=1, padding=1, bias=bias))
        m.append(nn.PixelShuffle(2))
    return nn.Sequential(*m)

class Upsample(nn.Sequential):
    def __init__(self, scale, direction, num_features, upsample_type='deconv',
                 norm=None, act=nn.ReLU(True), bias=True):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                # X2 block
                m += Upblock2x(direction, num_features, upsample_type, bias)

                if norm is not None:
                    m.append(get_norm(num_features, norm))

                if act is not None:
                    m.append(act)

        elif scale == 3 and upsample_type != 'deconv' and direction == 2:
            # X3 block only support pixelshuffle now
            m.append(nn.Conv2d(num_features, 9 * num_features, kernel_size=3, stride=1, padding=1, bias=bias))
            m.append(nn.PixelShuffle(3))

            if norm is not None:
                m.append(get_norm(num_features, norm))

            if act is not None:
                m.append(act)

        else:
            raise NotImplementedError
        super(Upsample, self).__init__(*m)

