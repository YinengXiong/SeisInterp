import argparse
import os
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--gpu', type=str, default='0', metavar='N', help='GPU ids')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoint', help='Path to save the checkpoint file')

        # input setting
        self.parser.add_argument('--dataroot', type=str, default='./Data/zeromean_new/', help='Path to dataset')
        self.parser.add_argument('--num_traces', type=int, default=150, help='Number of traces in each seismic data')
        self.parser.add_argument('--scale', type=int, default=4, help='Interpolation scale factor')
        self.parser.add_argument('--direction', type=int, default=0, help='Axis to interpolate. [0 (space) | 1 (time) | 2 (both)]')
        self.parser.add_argument('--patchSize', type=int, default=64, help='Size of croped seismic data')
        self.parser.add_argument('--repeat', type=int, default=50, help='Repeat time')
        self.parser.add_argument('--batchSize', type=int, default=1, help='Batch Size')
        self.parser.add_argument('--nThreads', type=int, default=4, help='Num of threads for loading data')

        # for network
        self.parser.add_argument('--arch', type=str, default='vdsr', help='Network architecture')
        self.parser.add_argument('--num_blocks', type=int, default=16, help='Number of blocks')
        self.parser.add_argument('--num_features', type=int, default=64, help='Number of features per block')
        self.parser.add_argument('--residual', action='store_true', help='Using residual shortcut or not')
        self.parser.add_argument('--use_bias', action='store_true', help='Using bias in Conv layers')
        self.parser.add_argument('--norm', type=str, default='None', help='Normalize type')


    def parse(self):
        if not self.initialized:
            self.initialize()
        self.args = self.parser.parse_args()
        self.args.isTrain = self.isTrain

        if self.args.norm.lower() == 'none':
            self.args.norm = None

        args = vars(self.args)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ---------------')

        return self.args

