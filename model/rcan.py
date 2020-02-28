from model import basic

import torch
import torch.nn as nn

# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        # global average pooling: feature -> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # feature channel downscale and upsample -> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, num_features, kernel_size, reduction, bias=True,
                 norm=None, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()

        body = []
        for i in range(2):
            body.append(nn.Conv2d(num_features, num_features, kernel_size, stride=1,
                                  padding=kernel_size//2, bias=bias))
            if norm is not None:
                body.append(basic.get_norm(num_features, norm))

            if i == 0:
                body.append(act)

        body.append(CALayer(num_features, reduction))

        self.body = nn.Sequential(*body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

# Residual Group
class ResidualGroup(nn.Module):
    def __init__(self, num_features, kernel_size, reduction, num_blocks,
                 bias=True, norm=None, act=nn.ReLU(True), res_scale=1):
        super(ResidualGroup, self).__init__()
        body = [
            RCAB(num_features, kernel_size, reduction, bias, norm, act, res_scale) \
            for _ in range(num_blocks)]

        body.append(nn.Conv2d(num_features, num_features, kernel_size,
                              stride=1, padding=kernel_size//2, bias=bias))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        num_groups = args.num_groups
        num_blocks = args.num_blocks
        num_features = args.num_features
        reduction = args.reduction
        res_scale = args.res_scale
        bias = not args.no_bias
        norm = args.norm
        nComp = args.nComp
        act = nn.ReLU(True)

        kernel_size = 3

        # Define head module
        self.head = nn.Conv2d(nComp, num_features, kernel_size, stride=1,
                            padding=kernel_size//2, bias=bias)

        # Define body module
        body = []
        for _ in range(num_groups):
            body.append(ResidualGroup(num_features, kernel_size, reduction, num_blocks,
                                      bias, norm, act, res_scale))
        body.append(nn.Conv2d(num_features, num_features, kernel_size, stride=1,
                              padding=kernel_size//2, bias=bias))
        self.body = nn.Sequential(*body)

        # Define tail module
        tail = [
            basic.Upsample(args.scale, args.direction, num_features,
                           act=None, bias=bias),
            nn.Conv2d(num_features, nComp, kernel_size,
                      padding=kernel_size//2, bias=bias)
        ]
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x
