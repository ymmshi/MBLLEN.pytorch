from torchvision import transforms
import torch.utils.data as data
from glob import glob
import numpy as np
from PIL import Image
import random
import torch
from torchvision.transforms import functional as tf
import os


class MBLLENData(data.Dataset):
    def __init__(self, data_dir, mode1, mode2, transform):
        super().__init__()
        self.data_dir = data_dir
        self.lows = glob(data_dir + '/%s_%s/*.*' % (mode1, mode2))
        self.highs = glob(data_dir + '/%s/*.*' % mode1)
        assert len(self.lows) == len(self.highs)
        self.transform = transform

    def my_transform(self, low_image, high_image, input_size=256):
        w, h = low_image.size
        print(w, h)
        h_offset = random.randint(0, h - input_size - 1)
        w_offset = random.randint(0, w - input_size - 1)
        low_crop = tf.crop(low_image, h_offset, w_offset, input_size, input_size)
        high_crop = tf.crop(high_image, h_offset, w_offset, input_size, input_size)
        if random.random() > 0.5:
            low_crop = tf.hflip(low_crop)
            high_crop = tf.hflip(high_crop)
        if random.random() > 0.5:
            low_crop = tf.vflip(low_crop)
            high_crop = tf.vflip(high_crop)
        return low_crop, high_crop

    def __getitem__(self, index):
        low_path = self.lows[index]
        high_path = self.highs[index]
        low_img = Image.open(low_path)
        high_img = Image.open(high_path)
        low_img = self.transform(low_img)
        high_img = self.transform(high_img)
        return low_img, high_img

    def __len__(self):
        return len(self.lows)
