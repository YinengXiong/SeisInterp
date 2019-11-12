import os
import random
import numpy as np
import cv2

from torch.utils.data.dataset import Dataset

def random_crop(hr, lr, size, scale):
    h, w = lr.shape[0], lr.shape[1]
    while True:
        crop_x = random.randint(0, h - size)
        crop_y = random.randint(0, w - size)

        hsize = size * scale
        hx, hy = crop_x * scale, crop_y * scale

        crop_lr = lr[crop_x: crop_x + size,
                     crop_y: crop_y + size].copy()
        crop_hr = hr[hx: hx + hsize,
                     hy: hy + hsize].copy()

        if (np.max(crop_hr) > 0):
            break
    return crop_hr, crop_lr

def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = np.random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    return im1.copy(), im2.copy()

class InteroDataset(Dataset):
    """
    A dataset class for seismic interpolation dataset.
    This dataset class is used for interpolation model which directly
    upsample the low-resolution data without pre-interpolation methods.
    It assumes that the directory '/path/to/data/trrain/' contains the seismic data to train.
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
        """

        self.data_path = os.path.join(args.dataroot, phase)
        self.data_arr = sorted(os.listdir(self.data_path) * self.repeat)
        self.data_len = len(self.data_arr)
        self.repeat = args.repeat

        self.num_traces = args.num_traces
        self.sacle = args.scale

        self.direction = args.direction
        self.patchSize = args.patchSize

    def __getitem__(self, index):
        """
        Return a data point and its metadata information.
        Parameters:
            index -- a random integer for data indexing
        Returns two tensors
            hr (tensor) -- high-resolution (ground-truth) seismic data patch
            lr (tensor) -- low-resolution (input) seismic data patch
        """

        # Read binary data files
        data_name = self.data_arr[index]
        hr = np.fromfile(os.path.join(self.data_path, data_name), 'float32')

        # To 2-D
        hr.shape = (self.num_traces, -1)
        hr = hr.T

        # For Multi-scale training
        if self.scale == 0:
            ss = random.randint(2, 4)
        else:
            ss = self.scale

        # Subsample
        if self.direction == 0:
            hr = hr[:, :hr.shape[1]//ss*ss]
            lr = cv2.resize(hr, (hr.shape[1] // ss, hr.shape[0]), cv2.INTER_CUBIC)
        elif self.direction == 1:
            hr = hr[:hr.shape[0]//ss*ss, :]
            lr = cv2.resize(hr, (hr.shape[1], hr.shape[0] // ss), cv2.INTER_CUBIC)
        else:
            hr = hr[:hr.shape[0]//ss*ss, :hr.shape[1]//ss*ss]
            lr = cv2.resize(hr, (hr.shape[1] // ss, hr.shape[0] // ss), cv2.INTER_CUBIC)

        # Data Augmentation
        hr, lr = random_crop(hr, lr, self.patchSize)
        hr, lr = random_flip_and_rotate(hr, lr)

        # to 3-D tensor todo: multi-components data
        hr = np.expand_dims(hr, axis=0)
        lr = np.expand_dims(lr, axis=0)

        return hr, lr

    def __len__(self):
        """
        Return the total number of data in the dataset
        """
        return self.data_len
