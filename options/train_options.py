from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        # for displays
        self.parser.add_argument('--print_freq', type=int, default=10, help='Frequency of showing training results on console')
        self.parser.add_argument('--val_freq', type=int, default=1, help='Frequency of evaluating on validation set')

        # for training
        self.parser.add_argument('--patchSize', type=int, default=64, help='Size of croped seismic data')
        self.parser.add_argument('--repeat', type=int, default=50, help='Repeat time')

        self.parser.add_argument('--resume', type=str, default='', help='Continue training: load the latest model')
        self.parser.add_argument('--pretrained', type=str, default='', help='Load the pretrained model from the specified location')
        self.parser.add_argument('--nEpochs', type=int, default=100, help='Num of epochs')
        self.parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer [adam | sgd]')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
        self.parser.add_argument('--lr_mode', type=str, default='poly', help='Learning rate schedule')
        self.parser.add_argument('--clip', type=float, default=10.0, help='Gradient Clipping')
        self.parser.add_argument('--clip_grad', action='store_true', help='Whether use Gradient clipping')
        self.parser.add_argument('--step', type=int, default=100, help='Num of step to decay learning rate')
        self.parser.add_argument('--loss', type=str, default='l2', help='Loss type [l2 | l1]')

        self.isTrain = True
