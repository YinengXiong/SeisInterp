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

        res_scale = args.res_scale

        kernel_size = 3
        act = nn.ReLU(True)
        self.residual = args.residual

        # Define head module
        self.head = nn.Conv2d(nComp, num_features, kernel_size,
                              padding=kernel_size//2, bias=bias)

        # Define body module
        body = []
        for _ in range(num_blocks):
            body.append(basic.ResBlock(num_features, kernel_size, bias=bias, norm=norm,
                                       act=act, res_scale=res_scale))

        body.append(nn.Conv2d(num_features, num_features, kernel_size,
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
        if self.residual:
            res = torch.add(res, x)

        x = self.tail(res)

        return x
