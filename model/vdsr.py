from model import basic
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        num_blocks = args.num_blocks
        num_features = args.num_features
        bias = not args.no_bias
        norm = args.norm
        nComp = args.nComp
        self.residual = args.residual

        kernel_size = 3
        self.head = nn.Conv2d(nComp, num_features, kernel_size,
                              stride=1, padding=1, bias=bias)
        self.act = nn.ReLU(inplace=True)

        body = []
        for _ in range(num_blocks):
            body.append(basic.BasicBlock(num_features, num_features, kernel_size,
                                         bias=bias, norm=norm))
        self.body = nn.Sequential(*body)

        self.tail = nn.Conv2d(num_features, nComp, kernel_size,
                              stride=1, padding=1, bias=bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        if self.residual:
            residual = x

        x = self.act(self.head(x))
        x = self.body(x)
        x = self.tail(x)

        if self.residual:
            x = torch.add(x, residual)
        return x
