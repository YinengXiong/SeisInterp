import os
import time
import shutil
import logging
import importlib

import numpy as np
import cv2

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
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                            epoch, i, len(train_data_loader), batch_time=batch_time,
                            data_time=data_time, loss=losses, top1=scores
                        ))

    logger.info(hist / count * 1.)

def validate(model, criterion, eval_score=None, args=None):
    batch_time = AverageMeter()
    score = AverageMeter()

    model.eval()
    start = time.time()

    val_dir = os.path.join(args.dataroot, 'Test')
    vallists = sorted(os.listdir(val_dir))
    valSNR = 0.
    valCount = len(vallists)

    for i, ff in enumerate(vallists):
        datafile = os.path.join(val_dir, ff)
        data = np.fromfile(datafile, 'float32')

        data.shape = (args.num_traces, -1)
        data = data.T

        # Subsample & pre-interpolate
        if args.direction == 0:
            subsampled = cv2.resize(data, (data.shape[1] // args.scale, data.shape[0]),
                                    cv2.INTER_CUBIC)
        elif args.direction == 1:
            subsampled = cv2.resize(data, (data.shape[1], data.shape[0] // args.scale),
                                    cv2.INTER_CUBIC)
        else:
            subsampled = cv2.resize(data, (data.shape[1] // args.scale, data.shape[0] // args.scale),
                                    cv2.INTER_CUBIC)
        subsampled = cv2.resize(subsampled, (data.shape[1], data.shape[0]), cv2.INTER_CUBIC)

        # Validate on whole image
        input_tensor = np.expand_dims(subsampled, axis=0)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = torch.from_numpy(input_tensor.copy()).float().cuda()

        # Forward
        with torch.no_grad():
            output = model(input_tensor)
        sr = output.squeeze(0).detach().cpu().numpy()

        # Evaluate performance
        if eval_score is not None:
            score.update(eval_score(sr, data), 1)

        # Update time
        batch_time.update(time.time() - start)
        start = time.time()

        valSNR += SNR(sr, data)

        if i % args.print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Score {score.val:.3f} ({score.avg:.3f})'.format(
                            i, len(vallists), batch_time=batch_time, score=score
                        ))

    finalscore = valSNR / valCount
    print('*' * 50)
    logger.info(finalscore)
    return score.avg

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copy(filename, os.path.join(args.checkpath, 'model_best.pth.tar'))


#######################################
#                Start                #
#######################################

# Parse Options
args = TrainOptions().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Generate Checkpoints Path
checkpath = os.path.join(args.checkpoints_dir, '{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}'.format(
    time.strftime("%m-%d_%H:%M", time.localtime()), args.arch,
    args.batchSize, args.patchSize, args.num_blocks, args.num_features,
    'res' if args.residual else 'nores', args.lr, args.lr_mode)
)
args.checkpath = checkpath

if not os.path.exists(checkpath):
    os.makedirs(checkpath)

# Generate Log
FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT, filemode='a')
logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(os.path.join(checkpath, 'train.log'))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

for k, v in args.__dict__.items():
    logger.info('{}:{}'.format(k, v))

# Prepare Dataset
print('Creating Dataset...')
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

# Create Model
print('Creating Model...')
start_time = time.time()
try:
    model = importlib.import_module("model.{}".format(args.arch)).Model(args)
    logger.info(model)

except ModuleNotFoundError:
    print("Model not found")

# Load pretrained model
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

# Define Loss Function
if args.loss == 'l2':
    criterion = nn.MSELoss()
elif args.loss == 'l1':
    criterion = nn.L1Loss()
criterion.cuda()

print('After parallel object, the time is {:.3f} s'.format(time.time() - start_time))

# Define Optimizer
if args.optimizer == 'adam':
    print('Using Adam Optimizer')
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), amsgrad=True)
else:
    print('Using SGD with momentum(0.9)')
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)

best_prec = 0.
start_epoch = 0

# Start Training
print('Start Training')
print('*' * 50)

for epoch in range(start_epoch, args.nEpochs):
    # Adjust learning rate
    lr = adjust_learning_rate(args, optimizer, epoch)
    logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))

    # Train Process
    train_model(train_data_loader, model, criterion, optimizer, epoch,
                accuracy, args)

    # Validate and Save Checkpoint
    if (epoch + 1) % args.val_freq == 0:
        prec = validate(model, criterion, SNR, args)

        is_best = prec > best_prec
        best_prec = max(best_prec, prec)

        ckpt_path = os.path.join(args.checkpath, 'checkpoint_latesest.pth.tar')
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
            'best_prec': best_prec
        }, is_best, filename=ckpt_path)

        history_path = os.path.join(args.checkpath, 'checkpoint_{:03d}.pth.tar'.format(epoch+1))
        shutil.copyfile(ckpt_path, history_path)

logger.info('Done !')
logger.info('Best SNR: {prec:.6f}'.format(prec=best_prec))
