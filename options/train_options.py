from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--print_freq', type=int, default=100, help='Frequency of showing training results on console')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='Frequency of saving checkpoints at the end of epochs')

        # for training
        self.parser.add_argument('--resume', type=str, default='', help='Continue training: load the latest model')
        self.parser.add_argument('--pretrained', type=str, default='', help='Load the pretrained model from the specified location')
        self.parser.add_argument('--nEpochs', type=int, default=100, help='Num of epochs')
        self.parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
        self.parser.add_argument('--lr_mode', type=str, default='poly', help='Learning rate schedule')
        self.parser.add_argument('--step', type=int, default=100, help='Num of step to decay learning rate')
        self.parser.add_argument('--loss', type=str, default='l2', help='Loss type [L2 | L1]')

        self.parser.add_argument('--log', action='store_true', help='Use log')

        self.isTrain = True
