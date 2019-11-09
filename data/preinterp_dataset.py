import os
import random
import numpy as np
import cv2

#from data.data_transforms import *

from torch.utils.data.dataset import Dataset

def random_crop(hr, lr, size):
    h, w = hr.shape[0], hr.shape[1]
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

def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3])
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

    #def __init__(self, data_path, num_traces, scale=4, direction=1, patch_size=64, repeat=50):
    def __init__(self, args, phase='Train'):
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
        #self.data_path = args.dataroot
        self.data_path = os.path.join(args.dataroot, phase)
        self.repeat = args.repeat
        self.data_arr = sorted(os.listdir(self.data_path) * self.repeat)
        self.data_len = len(self.data_arr)

        self.num_traces= args.num_traces
        self.scale = args.scale

        '''
        if args.scale == 0:
            self.scale = [2, 3, 4]
        else:
            self.scale = [args.scale]
        '''

        assert args.direction >= 0 and args.direction <= 2
        self.direction = args.direction
        self.patchSize = args.patchSize

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
        hr = np.fromfile(os.path.join(self.data_path, data_name), 'float32')

        # To 2-D
        hr.shape = (self.num_traces, -1)
        hr = hr.T # only needed in WYY data!

        # Subsample & pre-interpolate
        if self.scale == 0:
            ss = random.randint(2, 4)
        else:
            ss = self.scale

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

        hr, lr = random_crop(hr, lr, self.patchSize)
        hr, lr = random_flip_and_rotate(hr, lr)

        hr = np.expand_dims(hr, axis=0)
        lr = np.expand_dims(lr, axis=0)

        return hr, lr

    def __len__(self):
        """
        Return the total number of data in the dataset.
        """
        return self.data_len
