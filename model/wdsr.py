from model import basic
import torch
import torch.nn as nn

class wdsr_block(nn.Module):
    def __init__(self, num_features, kernel_size, wn,
                 bias=True, act=nn.ReLU(True), res_scale=1):
        super(wdsr_block, self).__init__()

        self.res_scale = res_scale
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv2d(num_features, num_features*expand, 1, padding=0, bias=bias)),
            act,
            wn(nn.Conv2d(num_features*expand, int(num_features*linear), 1, padding=0,
                         bias=bias)),
            wn(nn.Conv2d(int(num_features*linear), num_features, kernel_size,
                         padding=kernel_size//2, bias=bias))
        )

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res = torch.add(res, x)
        return x

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        num_blocks = args.num_blocks
        num_features = args.num_features
        bias = not args.no_bias
        nComp = args.nComp
        res_scale = args.res_scale

        kernel_size = 3
        self.act = nn.ReLU(True)

        wn = lambda x: torch.nn.utils.weight_norm(x)

        head = []
        head.append(wn(nn.Conv2d(nComp, num_features, kernel_size,
                                 padding=kernel_size//2, bias=bias)))
        self.head = nn.Sequential(*head)

        body = []
        for _ in range(num_blocks):
            body.append(wdsr_block(num_features, kernel_size, wn=wn, bias=bias,
                                   act=self.act, res_scale=res_scale))
        self.body = nn.Sequential(*body)

        tail = [
            basic.Upsample(args.scale, args.direction, num_features,
                           act=None, bias=bias),
            nn.Conv2d(num_features, nComp, kernel_size,
                      padding=kernel_size//2, bias=bias)
        ]
        self.tail = nn.Sequential(*tail)

        skip = [
            wn(nn.Conv2d(nComp, num_features, 5, padding=5//2)),
            basic.Upsample(args.scale, args.direction, num_features,
                           act=None, bias=bias),
            nn.Conv2d(num_features, nComp, 5, padding=5//2, bias=bias)
        ]
        self.skip = nn.Sequential(*skip)

    def forward(self, x):
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = torch.add(x, s)
        return x
