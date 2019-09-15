import basic
import torch.nn as nn

class VDSR(nn.Module):
    def __init__(self, args):
        super(VDSR, self).__init__()
        self.num_blocks = args.num_blocks
        self.num_features = args.num_features
        self.residual = args.residual
        self.use_bias = args.use_bias
        self.norm = args.norm

        kernel_size = 3
        body = []
        body.append(basic.BasicBlock(1, self.num_features, kernel_size,
                                     bias=self.use_bias, norm=self.norm))
        for _ in range(self.num_blocks-2):
            body.append(basic.BasicBlock(self.num_features, self.num_features, kernel_size,
                                         bias=self.use_bias, norm=self.norm))

        body.append(basic.BasicBlock(self.num_features, 1, kernel_size,
                                     bias=self.use_bias, norm=None, act=None))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        if self.residual:
            identity = x
        x = self.body(x)
        if self.residual:
            x += identity
        return x
