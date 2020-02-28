import os
import random
import numpy as np
import cv2

import torch
from torch.utils.data.dataset import Dataset

# H, W, C when data augmentation
def random_crop(hr, lr, size):
    h, w = lr.shape[0], lr.shape[1]
    while True:
        crop_x = random.randint(0, h - size)
        crop_y = random.randint(0, w - size)

        crop_lr = lr[crop_x: crop_x + size,
                     crop_y: crop_y + size].copy()
        crop_hr = hr[crop_x: crop_x + size,
                     crop_y: crop_y + size].copy()

        if (np.max(crop_hr) > 0):
            break
    return crop_hr, crop_lr

def random_flip_and_rotate(im1, im2, direction):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    if direction == 2:
        angle = random.choice([0, 1, 2, 3])
    else:
        angle = random.choice([0, 2])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    return im1.copy(), im2.copy()

class PreInterpDataset(Dataset):
    """
    A dataset class for seismic interpolation dataset.
    This dataset class is used for interpolation model which needs pre-interpolation methods.
    It assumes that the directory '/path/to/data/train' contains the seismic data to train.
    During test time, you need to prepare another directory '/path/to/data/test/'.
    """
    def __init__(self, args, phase='Train'):
        """
        Initialize the dataset class.
        Parameters:
            data_path -- /path/to/data/train/
            num_traces -- number of traces
            scale -- upsample scale factor
            direction -- direction which needs interpolation, 0(space) or 1(time) or 2(both)
            patchSize -- size of croped seismic data (lr patch size)
            repeat -- number of repeat in one epoch
            nComp -- number of components for training
            prefix -- prefix of seismic data file
        """

        self.data_path = os.path.join(args.dataroot, phase)
        self.repeat = args.repeat

        self.prefix = args.prefix
        self.nComp = args.nComp

        data_arr = sorted(os.listdir(self.data_path) * self.repeat)
        self.data_arr = []
        for dd in data_arr:
            if self.prefix[0] in dd:
                self.data_arr.append(dd)

        self.data_len = len(self.data_arr)

        self.num_traces= args.num_traces
        self.scale = args.scale
        self.tscale = args.tscale

        self.direction = args.direction
        self.patchSize = args.patchSize

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.
        Parameters:
            index -- a random integer for data indexing
        Returns two tensors (input, target)
            hr (tensor) -- high-resolution (ground-truth) seismic data patch
            lr (tensor) -- low-resolution (pre-interpolated input) seismic data patch
        """

        # Read binary data files
        if self.nComp == 1:
            data_name = self.data_arr[index]
            hr = np.fromfile(os.path.join(self.data_path, data_name), 'float32')

            hr.shape = (-1, self.num_traces)
            # tscale
            hr = hr[::self.tscale, :]
            hr = np.expand_dims(hr, axis=2)
        else:
            for icomp in range(self.nComp):
                if icomp == 0:
                    data_name = self.data_arr[index]
                else:
                    data_name = self.data_arr[index].replace(
                        self.prefix[icomp-1], self.prefix[icomp])
                hr_ = np.fromfile(os.path.join(self.data_path, data_name), 'float32')
                hr_.shape = (-1, self.num_traces)
                # tscale
                hr_ = hr_[::self.tscale, :]

                if icomp == 0:
                    hr = np.zeros((hr_.shape[0], hr_.shape[1], self.nComp), 'float32')

                hr[:, :, icomp] = hr_

        # For Multi-scale training
        if self.scale == 0:
            ss = random.randint(2, 4)
        else:
            ss = self.scale

        # Subsample & pre-interpolate
        if self.direction ==0:
            lr = cv2.resize(hr, (hr.shape[1] // ss, hr.shape[0]),
                            cv2.INTER_CUBIC)
        elif self.direction == 1:
            lr = cv2.resize(hr, (hr.shape[1], hr.shape[0] // ss),
                            cv2.INTER_CUBIC)
        else:
            lr = cv2.resize(hr, (hr.shape[1] // ss, hr.shape[0] // ss),
                            cv2.INTER_CUBIC)
        lr = cv2.resize(lr, (hr.shape[1], hr.shape[0]),
                        cv2.INTER_CUBIC)

        if self.nComp == 1:
            lr = np.expand_dims(lr, axis=2)

        # Data Augmentation
        hr, lr = random_crop(hr, lr, self.patchSize)
        hr, lr = random_flip_and_rotate(hr, lr, self.direction)

        hr = np.transpose(hr, (2, 0, 1))
        lr = np.transpose(lr, (2, 0, 1))
        hr = torch.from_numpy(hr.copy()).float()
        lr = torch.from_numpy(lr.copy()).float()

        return hr, lr

    def __len__(self):
        """
        Return the total number of data in the dataset.
        """
        return self.data_len
