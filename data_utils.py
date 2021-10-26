import torch
import torch.nn.functional as F
import random
import os
import numpy as np
import cv2
import os.path
from torch.autograd import Variable
from PIL import Image

def dataAug(imglr, imghr):
    if random.random() < 0.3:  # horizontal flip
        imglr = imglr[:, ::-1, :]
        imghr = imghr[:, ::-1, :]
    if random.random() < 0.3:  # vertical flip
        imglr = imglr[::-1, :, :]
        imghr = imghr[:, ::-1, :]

    rot = random.randint(0, 3)  # rotate
    imglr = np.rot90(imglr, rot, (0, 1))
    imghr = np.rot90(imghr, rot, (0, 1))

    # crop_size_w=random.randint((30,300))
    # crop_size_h=random.randint(30,300)
    crop_size_w = 128
    crop_size_h = 128
    crop_size_w1 = 256
    crop_size_h1 = 256

    input_size_h, input_size_w, _ = imglr.shape




    x_start=random.randrange(0, input_size_w-crop_size_w)
    y_start = random.randint(0, input_size_h -crop_size_h)
    (x_gt, y_gt)=(2*x_start, 2*y_start)
    imglr = imglr[y_start: y_start+crop_size_h,x_start : x_start + crop_size_w , :]
    imghr = imghr[y_gt: y_gt+crop_size_h1,x_gt : x_gt + crop_size_w1 , :]


    return imglr, imghr


def get_aug(data_dir):
    data_lr=os.path.join(data_dir, "LR")
    data_hr=os.path.join(data_dir, "ILR")
    fileList1 = os.listdir(data_lr)
    fileList2 = os.listdir(data_hr)
    nTrain = len(fileList1)
    train_images=[]
    GT_images=[]
    for i in range(1):
        name = fileList1[1]
        nameTrain = os.path.join(data_lr, name)
        nameGT = os.path.join(data_hr, name)
        imgTrain = cv2.imread(nameTrain)
        imgGT=cv2.imread(nameGT)
        for j in range(100):
            train_aug, GT_aug = dataAug(imgTrain, imgGT)
            train_images.append(train_aug)
            GT_images.append(GT_aug)

    return train_images, GT_images


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

        img_lr=RGB_np2Tensor(img_lr)
        img_gt = RGB_np2Tensor(img_gt)

        return img_lr, img_gt

