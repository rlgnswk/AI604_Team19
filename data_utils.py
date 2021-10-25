import torch
import torch.nn as nn #
from torch.utils import data
import torchvision
import torch.nn.functional as F
import random
import os
import numpy as np
import cv2
import os.path


def dataAug(imglr):
    if random.random() < 0.3:  # horizontal flip
        imglr = imglr[:, ::-1, :]


    if random.random() < 0.3:  # vertical flip
        imglr =  imglr[::-1, :, :]

    rot = random.randint(0, 3)  # rotate
    imglr = np.rot90(imglr, rot, (0, 1))

    crop_size_w=random.randint(0,40)
    crop_size_h=random.randint(0,40)

    input_size_h, input_size_w, _ = imglr[0].shape
    x_start = random.randint(0, input_size_w - crop_size_w)
    y_start = random.randint(0, input_size_h - crop_size_h)

    img_lr = imglr[y_start: y_start + crop_size_h, x_start: x_start + crop_size_w]

    return  img_lr

def RGB_np2Tensor(img_lr):

    # to Tensor
    ts = (2, 0, 1)
    img_lr = torch.Tensor(img_lr.transpose(ts).astype(float)).mul_(1.0)
    # normalization [-1,1]
    img_lr = (img_lr / 255.0 - 0.5) * 2
    return img_lr


class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, args):
        self.args=args
        self.dirpath = args.datasetPath
        self.datatype=args.datasettype
        self.final_path=os.path.join(self.dirpath, self.datatype)
        self.filelist = os.listdir(self.final_path)
        self.img_path = os.path.join(self.final_path, 'Test')
        self.filefin = os.listdir(self.img_path)


    def __len__(self):

        length=len(self.filefin)
        return length

    def __getitem__(self, idx):
        args = self.args
        imgIn1 = cv2.imread(self.img_path)
        imgIn1=RGB_np2Tensor(imgIn1)
        imgIn1 = downscale(imgIn1)

        return imgIn1

def downscale(img_aug):
    img_aug=F.interpolate(input=img_aug, scale_factor=0.5, mode='bicubic')
    return img_aug
