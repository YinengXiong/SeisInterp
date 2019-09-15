import os
import random
import numpy as np
import cv2

from utils import *

import torch
from torch.utils.data.dataset import Dataset

class PreInterpDataset(Dataset):
    """
    A dataset class for seismic interpolation dataset.
    This dataset class is used for interpolation model which needs pre-interpolation methods.
    It assumes that the directory '/path/to/data/train' contains the seismic data to train.
    During test time, you need to prepare another directory '/path/to/data/test/'.
    """

    def __init__(self, data_path, num_traces, scale=4, direction=1, patch_size=64, repeat=50):
        """
        Initialize the dataset class.
        Parameters:
            data_path -- /path/to/data/train/
            num_traces -- number of traces
            scale -- upsample scale factor
            direction -- direction which needs interpolation, 0(space) or 1(time) or 2(both)
            patch_size -- size of croped seismic data
            repeat -- number of repeat in one epoch
        """
        self.data_path = data_path
        self.data_arr = sorted(os.listdir(self.data_path) * repeat)
        self.data_len = len(self.data_arr)

        self.num_traces = num_traces
        assert scale > 1
        self.scale = scale
        assert direction >= 0 and direction <= 2
        self.direction = direction
        self.patch_size = patch_size

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns two tensors (input, target)
            input (tensor) -- input pre-interpolated seismic data patch
            target (tensor) -- target (ground-truth) seismic data patch
        """

        # Read binary data files
        data_name = self.data_arr[index]
        data = np.fromfile(self.data_path + data_name, 'float32')
        data.shape = (self.num_traces, -1)
        data = data.T
        print(data.shape)

        # Subsample & pre-interpolate
        if self.direction == 0:
            subsampled = cv2.resize(data, (data.shape[1] // self.scale, data.shape[0]),
                                    cv2.INTER_CIUBIC)
        elif self.direction == 1:
            subsampled = cv2.resize(data, (data.shape[1], data.shape[0] // self.scale),
                                    cv2.INTER_CUBIC)
        else:
            subsampled = cv2.resize(data, (data.shape[1] // self.scale, data.shape[0] // self.scale),
                                    cv2.INTER_CUBIC)
        print(subsampled.shape)
        subsampled = cv2.resize(subsampled, (data.shape[1], data.shape[0]), cv2.INTER_CUBIC)

        # Random Crop
        h, w = data.shape[0], data.shape[1]
        while True:
            crop_x = random.randint(0, max(0, h - self.patch_size - 1))
            crop_y = random.randint(0, max(0, w - self.patch_size - 1))
            data_crop = data[crop_x: crop_x + self.patch_size,
                             crop_y: crop_y + self.patch_size]
            input_crop = subsampled[crop_x: crop_x + self.patch_size,
                                    crop_y: crop_y + self.patch_size]

            if (np.max(data_crop) > 0):
                break

        # Data Augmentation
        flip_num = randint(0, 3)
        data_crop = flip(data_crop, flip_num)
        input_crop = flip(input_crop, flip_num)

        data_crop = np.expand_dims(data_crop, axis=0)
        input_crop = np.expand_dims(input_crop, axis=0)

        data_crop = torch.from_numpy(data_crop.copy()).float()
        input_crop = torch.from_numpy(input_crop.copy()).float()

        return input_crop, data_crop

    def __len__(self):
        return self.data_len

if __name__ == '__main__':
    preinterpdata = PreInterpDataset('./Data/zeromean_new/Train/', 150)
    lr, gt = preinterpdata.__getitem__(0)
    print(lr.size())
    print(gt.size())
