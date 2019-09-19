import os
import time
import math
import sys
import shutil
import argparse
import logging

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable

from dataset import *


FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT, filemode='a')
logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def SNR(pred, gt):
    imdff = pred - gt
    return 10. * np.log10(np.sum(gt ** 2) / (np.sum(imdff ** 2) + 1e-8))

def accuracy(output, target):
    """
    To be Done
    """
    SNRsum = 0.
    for ii in range(target.size(0)):
        pred = output[ii, :, :, :].squeeze().cpu().data.numpy()
        label = target[ii, :, :, :].squeeze().cpu().data.numpy()
        SNRsum += SNR(pred, label)
    return SNRsum / target.size(0) * 1.

def adjust_learning_rate(args, optimizer, epoch):
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.epochs) ** 0.9
    elif args.lr_mode == 'none':
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class PatchList(torch.utils.data.Dataset):
    def __init__(self, Data, lenx, leny, patch_size=96):
        self.Data = Data
        self.size = (int(lenx), int(leny))
        self.patch_size = patch_size
        self.patch_list = []
        self.get_lists()

    def get_lists(self):
        x = 0
        while (x < self.size[0]):
            y = 0
            while (y < self.size[1]):
                X = 0
                Y = 0
                if (x + self.patch_size) < self.size[0]:
                    X = x
                else:
                    X = self.size[0] - self.patch_size

                if (y + self.patch_size) < self.size[1]:
                    Y = y
                else:
                    Y = self.size[1] - self.patch_size

                X = int(X)
                Y = int(Y)
                self.patch_list.append([X, Y])

                y = y + self.patch_size * 0.75
            x = x + self.patch_size * 0.75

    def __getitem__(self, index):
        coordx = self.patch_list[index][0]
        coordy = self.patch_list[index][1]
        data_ = self.Data[coordx: coordx + self.patch_size,
                          coordy: coordy + self.patch_size]
        bicubic = cv2.resize(data_, (data_.shape[1] // 4, data_.shape[0]),
                             cv2.INTER_CUBIC)
        data = cv2.resize(bicubic, (data_.shape[1], data_.shape[0]),
                          cv2.INTER_CUBIC)
        data_tensor = torch.tensor(np.array(data), requires_grad=False)
        data_tensor = data_tensor.unsqueeze(0)
        data_final = [data_tensor, coordx, coordy]
        return tuple(data_final)

    def __len__(self):
        return len(self.patch_list)

def test(eval_data_loader, model, patch_size, output_size):
    model.eval()
    output = np.zeros((output_size[0], output_size[1]), dtype='float32')
    count = np.zeros((output_size[0], output_size[1]))
    for iter, (image, coordx, coordy) in enumerate(eval_data_loader):
        image_var = torch.tensor(image.float(), requires_grad=False).cuda(async=True)
        with torch.no_grad():
            out = model(image_var)

        for ind in range(len(coordx)):
            out_np = out[ind].squeeze().detach().cpu().numpy()
            x = int(coordx[ind].detach().numpy())
            y = int(coordy[ind].detach().numpy())

            output[x: x+patch_size, y: y+patch_size] += out_np
            count[x: x+patch_size, y: y+patch_size] += 1.0
    return output, count

def validate(val_dir, model, criterion, eval_score=None, print_freq=5, args=None):
    batch_time = AverageMeter()
    score = AverageMeter()

    model.eval()
    end = time.time()

    vallists = sorted(os.listdir(val_dir))
    valSNR = 0.
    valCount = len(vallists)
    for i, ff in enumerate(vallists):
        datafile = os.path.join(val_dir, ff)
        data = np.fromfile(datafile, 'float32')

        ###
        # WYY Dataset
        data.shape = (150, -1)
        data = data.T
        ###

        ### Field Data
        #data.shape = (2500, -1)
        ###

        # Val Dataset
        '''
        dataset = PatchList(data, data.shape[0], data.shape[1],
                            patch_size=args.val_size)
        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size*4, shuffle=False,
            num_workers=args.workers, pin_memory=False, drop_last=False
        )
        output_size = (data.shape[0], data.shape[1])

        output, count = test(val_loader, model, args.val_size, output_size)
        hr = output / count
        '''

        # whole image inference
        bicubic = cv2.resize(data, (data.shape[1] // 4, data.shape[0]), cv2.INTER_CUBIC)
        bicubic = cv2.resize(bicubic, (data.shape[1], data.shape[0]), cv2.INTER_CUBIC)

        data_tensor = np.expand_dims(bicubic, axis=0)
        data_tensor = np.expand_dims(data_tensor, axis=0)
        data_tensor = torch.from_numpy(data_tensor.copy()).float().cuda()
        with torch.no_grad():
            output = model(data_tensor)
        hr = output.squeeze().detach().cpu().numpy()

        if eval_score is not None:
            score.update(eval_score(hr, data), 1)
        batch_time.update(time.time() - end)
        end = time.time()

        valSNR += SNR(hr, data)
        if i % print_freq == 0:
            logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Score {score.val:.3f} ({score.avg:.3f})'.format(
                            i, len(vallists), batch_time=batch_time,
                            score=score))
    finalscore = valSNR / valCount
    logger.info(finalscore)
    return score.avg

def train_model(train_loader, model, criterion, optimizer, epoch,
                eval_score=None, print_freq=10, args=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    data_time = AverageMeter()
    model.train()

    end = time.time()

    hist = 0.
    count = 0.

    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
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

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {top1.val:.3f} ({top1.avg:.3f})'.format(
                            epoch, i, len(train_loader), batch_time=batch_time,
                            data_time=data_time, loss=losses, top1=scores))
    logger.info(hist / count * 1.)

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.checkpath+'/model_best.pth.tar'))

def train(args):
    batch_size = args.batch_size
    num_workers = args.workers

    print(time.strftime("%m-%d %H:%M", time.localtime()))
    checkpath = args.checkpointpath + '/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}'.format(
        time.strftime("%m-%d_%H:%M", time.localtime()), args.arch,  #01
        args.batch_size, args.input_size, args.num_blocks, args.num_features, #2345
        'residual' if args.residual else 'noresidual', args.lr, args.lr_mode # 678
    )
    os.makedirs(checkpath)

    fh = logging.FileHandler(checkpath + '/{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}.log'.format(
        'train', args.arch, args.batch_size, args.input_size,
        args.num_blocks, args.num_features, 'residual' if args.residual else 'noresidual',
        args.lr, args.lr_mode))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    print(' '.join(sys.argv))
    end_test = time.time()
    for k, v in args.__dict__.items():
        print(k, ':', v)
        logger.info('{}:{}'.format(k, v))

    if args.arch == 'vdsr':
        import vdsr
        model = vdsr.VDSR(num_blocks=args.num_blocks,
                          residual=args.residual)
    logger.info(model)

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pretrained '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    print('Found', torch.cuda.device_count(), 'GPUs')
    print('*' * 50)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()
    print('Use GPUs')
    print('*' * 50)

    print('after parallel object, the time is "{:.6f}"'.format(time.time() - end_test))

    if args.loss == 'l2':
        criterion = nn.MSELoss()
    elif args.loss == 'l1':
        criterion = nn.L1Loss()
    criterion.cuda()

    seis = SeisDataset(args.data_dir,
                       repeat=args.repeat, image_size=args.input_size)

    seis_train_loader = torch.utils.data.DataLoader(
        dataset=seis, num_workers=args.workers,
        batch_size=args.batch_size, shuffle=True,
        pin_memory=True, drop_last=False
    )
    if args.optim == 'adam':
        print('Using Adam Optimizer')
        optimizer = torch.optim.Adam(model.parameters(),
                                     args.lr, betas=(0.9, 0.999),
                                     weight_decay=1e-8, amsgrad=True)
    else:
        print('Using SGD Optimizer')
        optimizer = torch.optim.SGD(model.parameters(),
                                    args.lr, momentum=0.9, weight_decay=1e-4)

    best_prec1 = 0.
    start_epoch = 0
    print('Start Training')
    print('*' * 50)

    for epoch in range(start_epoch, args.epochs):
        lr = adjust_learning_rate(args, optimizer, epoch)
        logger.info('Epoch: [{0}]\tlr {1:.06f}'.format(epoch, lr))

        # Train Process
        train_model(seis_train_loader, model, criterion, optimizer, epoch,
                    eval_score=accuracy, args=args)

        # Validation Process
        prec = validate(args.val_dir, model, criterion, eval_score=SNR,
                        print_freq=10, args=args)
        logger.info(' * Avg SNR: {prec1:.3f}'.format(prec1=prec))
        is_best = prec > best_prec1
        best_prec1 = max(prec, best_prec1)
        args.checkpath = checkpath
        checkpoint_path = checkpath + '/checkpoint_latest.path.tar'
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        if (epoch + 1) % 1 == 0:
            history_path = checkpath + '/checkpoint_{:03d}.pth.tar'.format(epoch+1)
            shutil.copyfile(checkpoint_path, history_path)
    logger.info('Done !')
    logger.info('Best SNR: {prec1:.6f}'.format(prec1=best_prec1))


if __name__ == '__main__':
    """Define the options."""
    parser = argparse.ArgumentParser(description='Seismic Data Interpolation using CNN')
    parser.add_argument('--data_dir', type=str, default='./Data/zeromean_new/Train/', help='Path to training data')
    parser.add_argument('--val_dir', type=str, default='./Data/zeromean_new/Test/', help='Path to testing data')
    parser.add_argument('--gpu', type=str, default='0,1,2,3', type=str, metavar='N', help='GPU ids')
    parser.add_argument('--arch', type=str, default='vdsr', help='Network architectures')
    parser.add_argument('--num_blocks', type=int, default=16, help='Number of blocks')
    parser.add_argument('--num_features', type=int, default=64, help='Number of features per block')
    parser.add_argument('--residual', action='store_true', help='Using residual shortcut or not')
    parser.add_argument('--pretrained', type=str, default='', help='Pretrained model')
    parser.add_argument('--lr', type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument('--lr_mode', type=str, default='poly', help='Learning rate schedule')
    parser.add_argument('--step', type=int, default=100, help='Num of step to decay learning rate')
    parser.add_argument('--optim', type=str, default='adam', help='Optimizer')
    parser.add_argument('--loss', type=str, default='l2', help='Loss Type')
    parser.add_argument('--epochs', type=int, default=100, help="Num of epochs")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--workers', type=int, help='Num of workers')
    parser.add_argument('--repeat', type=int, default=50, help='Repeat times in one epoch')
    parser.add_argument('--input_size', type=int, default=96, help='Input patch size')
    parser.add_argument('--val_size', type=int, default=96, help='Patch size when testing')
    parser.add_argument('--checkpointpath', type=str, default='./checkpoint',
                        metavar='PATH', help='path to save the checkpoint file')
    parser.add_argument('--checkpath', default='./', type=str, metavar='PATH',
                        help='path to save the checkpoint file')

    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    train(args)
