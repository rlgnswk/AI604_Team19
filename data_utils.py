import torch
import torch.nn.functional as F
import random
import os
import numpy as np
import cv2
import os.path
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn



def dataAug(imglr, imghr):
    imglr=RGB_np2Tensor(imglr)
    imghr=RGB_np2Tensor(imghr)
    a = imglr.shape[1]
    b = imglr.shape[2]
    c = min(a, b)


    train_lr=[]
    train_hr=[]

    for i in range(10):
        k = random.randint(130, c)
        img_lr = torch.unsqueeze(imglr, 0)
        img_hr = torch.unsqueeze(imghr, 0)
        img_lr=F.interpolate(img_lr, scale_factor=k/c, mode='bicubic')
        img_hr = F.interpolate(img_hr, scale_factor=k/c, mode='bicubic')
        img_lr = torch.squeeze(img_lr, dim=0)
        img_hr = torch.squeeze(img_hr, dim=0)
        train_lr.append(img_lr)
        train_hr.append(img_hr)
        for j in range(4):
            img_lr= torch.rot90(img_lr, j, (1, 2))
            img_hr = torch.rot90(img_hr, j, (1,2))
            for k in range(2):
                if k==0:
                    img_lr = img_lr.flip(1)
                    img_hr = img_hr.flip(1)
                if k==1:
                     img_lr = img_lr.flip(2)
                     img_hr = img_hr.flip(2)
                train_lr.append(img_lr)
                train_hr.append(img_hr)
    return train_lr, train_hr



def get_aug(data_dir):
    data_lr=os.path.join(data_dir, "LR")
    data_hr=os.path.join(data_dir, "ILR")
    fileList1 = os.listdir(data_lr)
    fileList2 = os.listdir(data_hr)
    nTrain = len(fileList1)

    for i in range(1):
        name = fileList1[1]
        nameTrain = os.path.join(data_lr, name)
        nameGT = os.path.join(data_hr, name)
        imgTrain = cv2.imread(nameTrain)
        imgGT=cv2.imread(nameGT)
        train_aug, GT_aug = dataAug(imgTrain, imgGT)


    return train_aug, GT_aug


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
        self.kerneltype=args.kerneltype
        self.data_dir = os.path.join(self.dirpath, self.datatype, self.kerneltype)  # /datasets/Set5/g02
        train_images, GT_images = get_aug(self.data_dir)
        self.tr_im=train_images
        self.gt_im=GT_images

    def __len__(self):

        length=len(self.tr_im)
        return length

    def __getitem__(self, idx):
        args = self.args
        img_lr = self.tr_im[idx]
        img_gt=self.gt_im[idx]
        crop_size_w = 128
        crop_size_h = 128
        crop_size_w1 = 256
        crop_size_h1 = 256
        input_size_h = img_lr.shape[1]
        input_size_w = img_lr.shape[2]

        x_start = random.randrange(0, input_size_w - crop_size_w)
        y_start = random.randint(0, input_size_h - crop_size_h)
        (x_gt, y_gt) = (2 * x_start, 2 * y_start)
        img_lr = img_lr[:, y_start: y_start + crop_size_h, x_start: x_start + crop_size_w]
        img_gt = img_gt[:, y_gt: y_gt + crop_size_h1, x_gt: x_gt + crop_size_w1]

        return img_lr, img_gt

class testDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, args):
        self.args=args
        self.dirpath = args.datasetPath
        self.datatype=args.datasettype
        self.kerneltype=args.kerneltype
        self.data_dir = os.path.join(self.dirpath, self.datatype, self.kerneltype, "ILR")  # /datasets/Set5/g02
        self.fileList = os.listdir(self. data_dir)
        self.imgTest=os.path.join(self.data_dir, self.fileList[1])



    def __len__(self):

        length=1
        return length

    def __getitem__(self, idx):
        args = self.args
        self.img = cv2.imread(self.imgTest)

        img_hr=RGB_np2Tensor(self.img)


        return img_hr

