import torch
import torch.nn as nn
from math import sqrt

class Conv_ReLU_Block(nn.Module):
    def __init__(self, num_features):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=num_features, out_channels=num_features,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))

class VDSR(nn.Module):
    def __init__(self, num_blocks=18, num_features=64, residual=True):
        super(VDSR, self).__init__()
        self.num_blocks = num_blocks
        self.residual = residual
        self.num_features = num_features

        self.residual_layer = self.make_layer(Conv_ReLU_Block, self.num_blocks, self.num_features)
        self.input = nn.Conv2d(in_channels=1, out_channels=self.num_features, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=self.num_features, out_channels=1, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')

    def make_layer(self, block, num_of_layer, num_features):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(num_features))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.residual:
            residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        if self.residual:
            out = torch.add(out,residual)
        return out

if __name__ == '__main__':
    x = torch.rand(1, 1, 96, 96)
    model = VDSR(num_blocks=6, residual=False, num_features=128)
    print(model)
    y = model(x)
    print(y.size())

