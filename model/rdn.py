from model import basic

import torch
import torch.nn as nn

class RDBConv(nn.Module):
    """
    Conv Layer in Residual-Dense Block
    """
    def __init__(self, in_channels, grow_rate, kernel_size=3, bias=True):
        super(RDBConv, self).__init__()
        Cin = in_channels
        G = grow_rate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kernel_size, stride=1, padding=kernel_size//2,
                      bias=bias),
            nn.ReLU(True)
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kernel_size=3, bias=True):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        conv = []
        for c in range(C):
            conv.append(RDBConv(G0 + c*G, G, kernel_size, bias))
        self.conv = nn.Sequential(*conv)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1, bias=bias)

    def forward(self, x):
        return self.LFF(self.conv(x)) + x

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        nComp = args.nComp
        bias = not args.no_bias

        self.D = args.num_blocks
        G0 = args.num_features
        G = args.num_features
        C = args.num_layers

        kernel_size = 3

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(nComp, G0, kernel_size, padding=1, stride=1, bias=bias)
        self.SFENet2 = nn.Conv2d(G0, G0, kernel_size, padding=1, stride=1, bias=bias)

        # Residual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for _ in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C,
                    kernel_size=kernel_size, bias=bias)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1, bias=bias),
            nn.Conv2d(G0, G0, 3, padding=1, stride=1, bias=bias)
        ])

        # Upsapling net
        tail = [
            basic.Upsample(args.scale, args.direction, G0,
                           act=None, bias=bias),
            nn.Conv2d(G0, nComp, kernel_size,
                      padding=kernel_size//2, bias=bias)
        ]
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        f_1 = self.SFENet1(x)
        x = self.SFENet2(f_1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x = torch.add(x, f_1)

        x = self.tail(x)

        return x

