import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from pre_processing import *
import random

class SeisDataset(Dataset):
    def __init__(self,
                 #data_path='../SEAM/singleshot/zeromean/Train/',
                 data_path = './Data/zeromean_new/Train/',
                 image_size=64,
                 repeat=100):
        self.data_path = data_path
        self.data_arr = sorted(os.listdir(self.data_path) * repeat)
        self.data_len = len(self.data_arr)
        self.image_size = image_size
        #print(self.data_len)

    def __getitem__(self, index):
        data_name = self.data_arr[index]
        data = np.fromfile(self.data_path + data_name, 'float32')
        #'''
        ### for WYY Dataset
        #data.shape = (150, -1)
        #data = data.T
        # 501 * 150
        ###
        #'''

        ### Field Data
        #data.shape = (2500, -1)
        ###

        #data = data[200:, :]        # SEAM
        '''
        while True:
            w, h = data.shape[0], data.shape[1]
            crop_x = random.randint(0, max(0, w - self.image_size - 1))
            crop_y = random.randint(0, max(0, h - self.image_size - 1))
            data_crop = data[crop_x : crop_x + self.image_size,
                             crop_y : crop_y + self.image_size]

            if (np.max(data_crop) > 1e-4) and (np.max(data_crop) > 1e-4):
                break

        flip_num = randint(0, 3)
        data_crop = flip(data_crop, flip_num)
        bicubic = cv2.resize(data_crop, (data_crop.shape[1] // 4, data_crop.shape[0]),
                             cv2.INTER_CUBIC)
        lrinput = cv2.resize(bicubic, (data_crop.shape[1], data_crop.shape[0]),
                             cv2.INTER_CUBIC)

        lrinput = np.expand_dims(lrinput, axis=0)
        lrinput = torch.from_numpy(lrinput.copy()).float()
        label = np.expand_dims(data_crop, axis=0)
        label = torch.from_numpy(label.copy()).float()
        return lrinput, label
        '''

        #'''
        data.shape = (150, -1)
        data = data.T

        bicubic = cv2.resize(data, (data.shape[1] // 4, data.shape[0]), cv2.INTER_CUBIC)
        bicubic = cv2.resize(bicubic, (data.shape[1], data.shape[0]), cv2.INTER_CUBIC)

        w, h = data.shape[0], data.shape[1]
        while True:

            crop_x = random.randint(0, max(0, w - self.image_size - 1))
            crop_y = random.randint(0, max(0, h - self.image_size - 1))
            data_crop = data[crop_x: crop_x + self.image_size,
                             crop_y: crop_y + self.image_size]
            bicubic_crop = bicubic[crop_x: crop_x + self.image_size,
                                   crop_y: crop_y + self.image_size]
            if (np.max(data_crop) > 1e-4):
                break

        flip_num = randint(0, 3)
        data_crop = flip(data_crop, flip_num)
        bicubic_crop = flip(bicubic_crop, flip_num)

        bicubic_crop = np.expand_dims(bicubic_crop, axis=0)
        bicubic_crop = torch.from_numpy(bicubic_crop.copy()).float()
        data_crop = np.expand_dims(data_crop, axis=0)
        data_crop = torch.from_numpy(data_crop.copy()).float()
        return bicubic_crop, data_crop
        #'''

    def __len__(self):
        return self.data_len

if __name__ == '__main__':
    #seisdata = SeisDataset('../SEAM/singleshot/zeromean/Train/')
    seisdata = SeisDataset('./Data/zeromean_new/Train/')
    #seisdata = SeisDataset('/share/home/xyn/FieldData/Zeromean/Train/')
    lr, gt = seisdata.__getitem__(0)
    print(lr.size())
    print(gt.size())
