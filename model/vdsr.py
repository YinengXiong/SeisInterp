from model import basic
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.num_blocks = args.num_blocks
        self.num_features = args.num_features
        self.residual = args.residual
        self.bias = not args.no_bias
        self.norm = args.norm

        kernel_size = 3
        self.head = nn.Conv2d(1, self.num_features, kernel_size,
                              stride=1, padding=1, bias=self.bias)
        self.act = nn.ReLU(inplace=True)

        body = []
        for _ in range(self.num_blocks):
            body.append(basic.BasicBlock(self.num_features, self.num_features, kernel_size,
                                         bias=self.bias, norm=self.norm))
        self.body = nn.Sequential(*body)

        self.tail = nn.Conv2d(self.num_features, 1, kernel_size,
                              stride=1, padding=1, bias=self.bias)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')

    def forward(self, x):
        if self.residual:
            identity = x
        x = self.act(self.head(x))
        x = self.body(x)
        x = self.tail(x)

        if self.residual:
            x = torch.add(x, identity)
        return x
