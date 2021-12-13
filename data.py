import torch
import random
import torch.nn.functional as F
from torchvision.utils import save_image

import numpy as np
import random
import cv2
from math import floor

from metric import *
from lpips import lpips

class dummy():
    def __init__(self):
        self.SR_ratio = 2
        self.patchSize = 128

def dataAug(lq, args):
    # generate aspect ratio between 64 and shorter side of image
    reshape_dims = []
    if lq.shape[2] <= lq.shape[3]:
        for i in range(64, lq.shape[2]+1):
            y = i
            x = round(lq.shape[3] / lq.shape[2] * y)
            if x%args.SR_ratio==0 and y%args.SR_ratio==0:
                reshape_dims.append((x, y))
    else:
        for i in range(64, lq.shape[3]+1):
            x = i
            y = round(lq.shape[2] / lq.shape[3] * x)
            if x%args.SR_ratio==0 and y%args.SR_ratio==0:
                reshape_dims.append((x, y))

    weights = np.float32([x * y / (lq.shape[2] * lq.shape[3]) for (x, y) in reshape_dims])
    pair_prob = weights / np.sum(weights)

    x, y = random.choices(reshape_dims, weights=pair_prob, k=1)[0]

    # generate hr father / lr son
    img_hr = F.interpolate(lq, size=[x,y], mode='bicubic', align_corners=True)
    img_lr = F.interpolate(img_hr, size=[x//args.SR_ratio, y//args.SR_ratio], mode='bicubic', align_corners=True)   # Downsample
    img_lr = F.interpolate(img_lr, size=[x,y], mode='bicubic', align_corners=True)                                  # Upsample

    # crop
    img_hr, img_lr = crop(img_hr, img_lr, min(args.patchSize, min(x,y)))

    # generate hr-lr pairs
    hr_fathers = img_hr
    lr_sons = img_lr
    
    for j in range(4):
        rot_hr = torch.rot90(img_hr, j, (2,3))
        rot_lr = torch.rot90(img_lr, j, (2,3))

        if j != 0:
            hr_fathers = torch.cat([hr_fathers, rot_hr], dim=0)
            lr_sons = torch.cat([lr_sons, rot_lr], dim=0)
        
        for k in range(2, 4):
            flip_hr = torch.flip(rot_hr, [k])
            flip_lr = torch.flip(rot_lr, [k])

            hr_fathers = torch.cat([hr_fathers, flip_hr], dim=0)
            lr_sons = torch.cat([lr_sons, flip_lr], dim=0)

    # shuffle
    idx = torch.randperm(hr_fathers.shape[0])
    hr_fathers = hr_fathers[idx].view(hr_fathers.size())
    lr_sons = lr_sons[idx].view(lr_sons.size())

    return hr_fathers, lr_sons

def RGB_np2Tensor(img):
    # to Tensor
    ts = (2, 0, 1)
    img = torch.Tensor(img.transpose(ts).astype(float)).mul_(1.0)
    # normalization [-1,1]
    img = (img / 255.0 - 0.5) * 2
    return img

def crop(img_hr, img_lr, patch_size):
    _, _, input_size_h, input_size_w = img_lr.shape

    x_start = random.randrange(0, input_size_w - patch_size + 1)
    y_start = random.randrange(0, input_size_h - patch_size + 1)

    img_hr = img_hr[: ,: , y_start:y_start+patch_size, x_start:x_start+patch_size]
    img_lr = img_lr[: ,: , y_start:y_start+patch_size, x_start:x_start+patch_size]

    return img_hr, img_lr

args = dummy()
img = torch.randn((1, 3, 256, 256))
hr_fathers, lr_sons = dataAug(img, args)
print(hr_fathers.shape)