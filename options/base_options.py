import argparse

class BaseOptions():
    """
    This class defines options used during training and test time.
    It also implements some helper function such as printing, saving and parsing the options.
    """
    def __init__(self):
        """
        Reset the class; inticates the class hasn't been initialized
        """
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        """
        Define the common options that are used in both training and test.
        """
        # experiment specifics
        self.parser.add_argument('--gpu', type=str, default='0', metavar='N', help='GPU ids')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoint', help='Path to save the checkpoint file')

        # input setting
        self.parser.add_argument('--dataroot', type=str, default='./Data/zeromean_new/', help='Path to dataset')
        self.parser.add_argument('--num_traces', type=int, default=150, help='Number of traces in each seismic data')
        self.parser.add_argument('--nComp', type=int, default=1, help='Number of components of seismic data')
        self.parser.add_argument('--prefix', type=str, default='', help='Prefix of seismic filename e.g. shot_p')
        self.parser.add_argument('--scale', type=int, default=4, help='Interpolation scale factor [0 (multi-scale, 2 ~ 4) | 2 | 3 | 4]')
        self.parser.add_argument('--tscale', type=int, default=1, help='Time scale')
        self.parser.add_argument('--direction', type=int, default=0, help='Axis to interpolate [0 (space) | 1 (time) | 2 (both)]')
        self.parser.add_argument('--batchSize', type=int, default=32, help='Batch Size')
        self.parser.add_argument('--nThreads', type=int, default=4, help='Num of threads for loading data')

        # for network
        self.parser.add_argument('--arch', type=str, default='vdsr', help='Network architecture')
        self.parser.add_argument('--num_blocks', type=int, default=16, help='Number of blocks')
        self.parser.add_argument('--num_features', type=int, default=64, help='Number of features per block')
        self.parser.add_argument('--num_layers', type=int, default=8, help='Number of layers in each block, required in some arch, e.g. rdb')
        self.parser.add_argument('--num_groups', type=int, default=8, help='Number of groups in network, required in some arch, e.g. rcan')
        self.parser.add_argument('--reduction', type=int, default=16, help='Number of feature maps reduction in channel attention layer')
        self.parser.add_argument('--res_scale', type=float, default=1.0, help='Res scale of each residual block')
        self.parser.add_argument('--residual', action='store_true', help='Using residual shortcut or not')
        self.parser.add_argument('--no_bias', action='store_true', help='Not using bias in Conv layers')
        self.parser.add_argument('--norm', type=str, default='None', help='Normalize type')

    def print_options(self, args):
        '''
        Pirint the options
        It will print both current options and default options
        '''
        message = ''
        message += ('\n' + '-'*30 + 'Options' + '-' * 30 + '\n')
        for k, v in sorted(vars(self.args).items()):
            default = self.parser.get_default(k)
            comment = '\t[default: %s]' %str(default)
            message += '{:>20}: {:<20}{}\n'.format(str(k), str(v), comment)
        message += ('-'*30 + '  End  ' + '-' * 30 + '\n')
        print(message)

    def parse(self):
        '''
        Parse the options
        '''
        if not self.initialized:
            self.initialize()
        self.args = self.parser.parse_args()
        self.args.isTrain = self.isTrain

        if self.args.norm.upper() not in ['BN', 'IN']:
            self.args.norm = None

        # Prefix
        prefix = self.args.prefix.split(',')
        self.args.prefix = []
        for pre in prefix:
            self.args.prefix.append(pre)

        self.print_options(self.args)
        '''
        args = vars(self.args)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ---------------')
        '''

        return self.args

