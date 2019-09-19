import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=True,
                 norm=None, act=nn.ReLU(True)):

        m = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, groups=groups,
                       bias=bias)]

        if norm == 'BN':
            m.append(nn.BatchNorm2d(out_channels))
        elif norm == 'IN':
            m.append(nn.InstanceNorm2d(out_channels))
        else:
            pass

        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(self, conv, num_features, kernel_size, stride=1, groups=1, bias=True,
                 norm=None, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()

        m = []
        for i in range(2):
            m.append(nn.Conv2d(num_features, num_features, kernel_size,
                               stride=stride, groups=groups, bias=bias))
            if norm == 'BN':
                m.append(nn.BatchNorm2d(num_features))
            elif norm == 'IN':
                m.append(nn.InstanceNorm2d(num_features))
            else:
                pass

            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
