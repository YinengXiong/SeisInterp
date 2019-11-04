import torch
import numpy as np

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
    calculate SNR
    """
    SNRsum = 0.
    #print(target.size(0))
    for ii in range(target.size(0)):
        pred = output[ii, :, :, :].squeeze().cpu().data.numpy()
        label = target[ii, :, :, :].squeeze().cpu().data.numpy()
        SNRsum += SNR(pred, label)
    return SNRsum / target.size(0) * 1.

def adjust_learning_rate(args, optimizer, epoch):
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step))
    elif args.lr_mode == 'poly':
        lr = args.lr * (1 - epoch / args.nEpochs) ** 0.9
    elif args.lr_mode =='none':
        lr = args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
