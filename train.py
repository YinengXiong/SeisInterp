import os
import time
import logging
import importlib
from options.train_options import TrainOptions
from model import *

args = TrainOptions().parse()

# Generate Checkpoints Path
checkpath = os.path.join(args.checkpoints_dir, '{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}'.format(
    time.strftime("%m-%d_%H:%M", time.localtime()), args.arch,
    args.batchSize, args.patchSize, args.num_blocks, args.num_features,
    'res' if args.residual else 'nores', args.lr, args.lr_mode)
)

if not os.path.exists(checkpath):
    os.makedirs(checkpath)

# Generate Log
if args.log:
    FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
    logging.basicConfig(format=FORMAT, filemode='a')
    logger=logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(checkpath, 'train.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    for k, v in args.__dict__.items():
        logger.info('{}:{}'.format(k, v))

try:
    model = importlib.import_module("model.{}".format(args.arch)).Model(args)
    if args.log:
        logger.info('\nModel Created')
        logger.info(model)
    else:
        print('\nModel Created')
        print(model)

except ModuleNotFoundError:
    print("Model not found")
#print('Model Created')

if args.log:
    logger.info(model)

