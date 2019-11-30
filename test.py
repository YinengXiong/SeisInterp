import os
import time
import importlib

import numpy as np
import cv2

import torch
import torch.utils.data

from options.test_options import TestOptions
from utils import *

class PatchList(torch.utils.data.Dataset):
    def __init__(self, data, nComp, lenx, leny, patch_size=64):
        self.data = data
        self.size = (int(lenx), int(leny))
        self.nComp = nComp
        self.patch_size = patch_size
        self.patch_list = []
        self.get_list()

    def get_list(self):
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

                y += self.patch_size * 0.75
            x += self.patch_size * 0.75

    def __getitem__(self, index):
        coordx = self.patch_list[index][0]
        coordy = self.patch_list[index][1]
        data_ = self.data[coordx: coordx + self.patch_size,
                          coordy: coordy + self.patch_size]
        if self.nComp == 1:
            data_ = np.expand_dims(axis=2)
        data_tensor = torch.tensor(np.array(data_), requires_grad=False)
        #data_tensor = data_tensor.unsqueeze(0)
        data_final = [data_tensor, coordx, coordy]
        return tuple(data_final)

    def __len__(self):
        return len(self.patch_list)

def test_model(test_loader, model, patch_size, output_size):
    output = np.zeros((output_size[0], output_size[1]), dtype='float32')
    count = np.zeros((output_size[0], output_size[1]), dtype='float32')

    for i, (input_tensor, coordx, coordy) in enumerate(test_loader):
        input_var = input_tensor.clone().detach().requires_grad_(False).float().cuda()
        with torch.no_grad():
            out = model(input_var)

        for ind in range(len(coordx)):
            out_np = out[ind].squeeze().detach().cpu().numpy()
            x = int(coordx[ind].detach().numpy())
            y = int(coordy[ind].detach().numpy())

            output[x: x+patch_size, y: y+patch_size] += out_np
            count[x: x+patch_size, y: y+patch_size] += 1.0
    return output, count


#######################################
#                Start                #
#######################################

# Parse Options
args = TestOptions().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Generate Results dir
if args.sample_dir:
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

# Create Model
print('Creating Model...')
model = importlib.import_module("model.{}".format(args.arch)).Model(args)


# Load pretrained model
if args.model:
    if os.path.exists(args.model):
        print('=> loading pretrained "{}"'.format(args.model))
        checkpoint = torch.load(args.model)
        best_prec = checkpoint['best_prec']
        model.load_state_dict(checkpoint['state_dict'])
        print('best performance', best_prec)
    else:
        print('=> no checkpoint found at "{}"'.format(args.model))

# Use GPUs
print('Found', torch.cuda.device_count(), 'GPUs')
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.cuda()
model.eval()

# Start Testing
print('Start Testing')
print('*' * 50)


start = time.time()

test_dir = os.path.join(args.dataroot, 'Test')
testlists_ = sorted(os.listdir(test_dir))
testlists = []
for t in testlists_:
    if args.prefix[0] in t:
        testlists.append(t)

testSNR = 0.
testCount = len(testlists)
testTime = 0.

for i, ff in enumerate(testlists):
    if args.nComp == 1:
        datafile = os.path.join(test_dir, ff)
        hr = np.fromfile(datafile, 'float32')

        hr.shape = (-1, args.num_traces)
        hr = np.expand_dims(hr, axis=2)
    else:
        for icomp in range(args.nComp):
            if icomp == 0:
                datafile = os.path.join(test_dir, ff)
            else:
                datafile = os.path.join(test_dir, ff.replace(
                    args.prefix[icomp-1], args.prefix[icomp]))
            hr_ = np.fromfile(datafile, 'float32')
            hr_.shape = (-1, args.num_traces)

            if icomp == 0:
                hr = np.zeros((hr_.shape[0], hr_.shape[1], args.nComp), 'float32')

            hr[:, :, icomp] = hr_

    # Subsample & pre-interpolate
    if args.scale == 0:
        ss = 4
    else:
        ss = args.scale

    if args.direction == 0:
        if args.arch != 'vdsr':
            hr = hr[:, :hr.shape[1]//ss*ss]
        lr = cv2.resize(hr, (hr.shape[1] // ss, hr.shape[0]), cv2.INTER_CUBIC)
    elif args.direction == 1:
        if args.arch != 'vdsr':
            hr = hr[:hr.shape[0]//ss*ss, :]
        lr = cv2.resize(hr, (hr.shape[1], hr.shape[0] // ss), cv2.INTER_CUBIC)
    else:
        if args.arch != 'vdsr':
            hr = hr[:hr.shape[0]//ss*ss, :hr.shape[1]//ss*ss]
        lr = cv2.resize(hr, (hr.shape[1] // ss, hr.shape[0] // ss), cv2.INTER_CUBIC)

    # only vdsr need per-interpolation
    if args.arch == 'vdsr':
        lr = cv2.resize(lr, (hr.shape[1], hr.shape[0]), cv2.INTER_CUBIC)

    if args.nComp == 1:
        lr = np.expand_dims(lr, axis=2)

    start = time.time()
    if args.testSize == -1:
        # Test on whole data
        hr = np.transpose(hr, (2, 0, 1))
        lr = np.transpose(lr, (2, 0, 1))
        lr = torch.from_numpy(lr.copy()).float().cuda().unsqueeze(0)
        with torch.no_grad():
            output = model(lr)
        sr = output.squeeze(0).detach().cpu().numpy()
    else:
        # TODO: arch edsr & multi-component
        # Extract patches
        dataset = PatchList(lr, lr.shape[0], lr.shape[1],
                            patch_size=args.testSize)
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batchSize, shuffle=False,
            num_workers=args.nThreads, pin_memory=False, drop_last=False
        )
        output_size = (hr.shape[0], hr.shape[1])

        output, count = test_model(test_loader, model, args.testSize, output_size)
        sr = output / count

    testTime += (time.time() - start)

    result = SNR(sr, hr)
    testSNR += result

    if args.sample_dir:
        filename = args.arch + '_' + ff.replace('.dat', '.npy')
        np.save(os.path.join(args.sample_dir, filename), sr)

    print('{:40} SNR = {:.4f}'.format(datafile, result))

print('*' * 50)
print('Average SNR {:.4f} dB'.format(testSNR / testCount))
print('Average Inference time {:.4f} s'.format(testTime / testCount))
if args.sample_dir:
    print('Results saved to ', args.sample_dir)
