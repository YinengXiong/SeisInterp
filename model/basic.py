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
