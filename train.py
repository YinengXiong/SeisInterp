import os
import time
import logging
import importlib

import torch
import torch.nn as nn
from torch.autograd import Variable

from options.train_options import TrainOptions
from utils import *

def train_model(train_data_loader, mnodel, criterion, optimizer, epoch,
                eval_score=None, args=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()

    start = time.time()

    hist = 0.
    count = 0.

    for i, (input, target) in enumerate(train_data_loader):
        data_time.update(time.time() - start)
        time_input = time.time()

        input = Variable(input.cuda())
        target = Variable(target.cuda())

        output = model(input)
        loss = criterion(output, target)

        losses.update(loss.data.item(), input.size(0))

        for index in range(target.size(0)):
            pred = output[index, :, :, :].cpu().data.numpy()
            label = target[index, :, :, :].cpu().data.numpy()
            hist += SNR(pred, label)
            count += 1

        if eval_score is not None:
            scores.update(eval_score(output, target), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)
        start = time.time()

        if i % args.print_freq == 0:
            if args.log:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                                epoch, i, len(train_data_loader), batch_time=batch_time,
                                data_time=data_time, loss=losses, top1=scores
                            ))
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                          epoch, i, len(train_data_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, top1=scores
                      ))

    if args.log:
        logger.info(hist / count * 1.)
    else:
        print(hist / count * 1.)

args = TrainOptions().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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

# Create Model
start_time = time.time()
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

# Load pretrained
if args.pretrained:
    if os.path.isfile(args.pretrained):
        print('=> loading pretrained "{}"'.format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('=> no checkpoint found at "{}"'.format(args.pretrained))

# Use GPUs
print('Found', torch.cuda.device_count(), 'GPUs')
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.cuda()

print('After parallel object, the time is {:.3f} s'.format(time.time() - start_time))

# Define Loss
if args.loss == 'l2':
    criterion = nn.MSELoss()
elif args.loss == 'l1':
    criterion = nn.L1Loss()
criterion.cuda()

if args.arch == 'vdsr':
    # need preinterp
    from data.preinterp_dataset import PreInterpDataset
    dataset = PreInterpDataset(args, phase='Train')
else:
    ###
    # under construction
    pass
    ###

train_data_loader = torch.utils.data.DataLoader(
    dataset=dataset, num_workers=args.nThreads, batch_size=args.batchSize, shuffle=True,
    pin_memory=True, drop_last=False
)

if args.optimizer == 'adam':
    print('Using Adam Optimizer')
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), amsgrad=True)
else:
    print('Using SGD with momentum(0.9)')
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)

best_prec1 = 0.
start_epoch = 0

if not args.log:
    print('Start Training')
    print('*' * 50)

for epoch in range(start_epoch, args.nEpochs):
    # Adjust learning rate
    lr = adjust_learning_rate(args, optimizer, epoch)
    if args.log:
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))
    else:
        print('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))

    # Train Process
    train_model(train_data_loader, model, criterion, optimizer, epoch,
                accuracy, args)

    #if epoch % args.val_freq == 0:
        #prec = validate(args.val_dir, model)


