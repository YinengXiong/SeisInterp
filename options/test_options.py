from .base_options import BaseOptions

class TestOptions(BaseOptions):
    """
    This class includes test options
    """
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--model', type=str, help='file path of trained model')
        self.parser.add_argument('--testSize', type=int, default=-1, help='testing patch size, -1 means whole data')
        self.parser.add_argument('--sample_dir', type=str, default='', help='file path to store test results')
        self.parser.add_argument('--respective', action='store_true', help='Calculate SNR every channel respectively')

        self.isTrain = False
