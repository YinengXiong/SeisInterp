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
    def __init__(self, data, lenx, leny, patch_size=64):
        self.data = data
        self.size = (int(lenx), int(leny))
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
        data_tensor = torch.tensor(np.array(data_), requires_grad=False)
        data_tensor = data_tensor.unsqueeze(0)
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
#print(model)


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
testlists = sorted(os.listdir(test_dir))
testSNR = 0.
testCount = len(testlists)
testTime = 0.

for i, ff in enumerate(testlists):
    datafile = os.path.join(test_dir, ff)
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
    subsampled = cv2.resize(subsampled, (data.shape[1], data.shape[0]),
                            cv2.INTER_CUBIC)

    start = time.time()
    if args.testSize == -1:
        # Test on whole data
        input_tensor = np.expand_dims(subsampled, axis=0)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = torch.from_numpy(input_tensor.copy()).float().cuda()
        with torch.no_grad():
            output = model(input_tensor)
        sr = output.squeeze(0).detach().cpu().numpy()
    else:
        # Extract patches
        dataset = PatchList(subsampled, subsampled.shape[0], subsampled.shape[1],
                            patch_size=args.testSize)
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batchSize, shuffle=False,
            num_workers=args.nThreads, pin_memory=False, drop_last=False
        )
        output_size = (subsampled.shape[0], subsampled.shape[1])

        output, count = test_model(test_loader, model, args.testSize, output_size)
        sr = output / count

    result = SNR(sr, data)
    testSNR += result
    testTime += (time.time() - start)
    print('{:40} SNR = {:.4f}'.format(datafile, result))

print('*' * 50)
print(testSNR / testCount)
