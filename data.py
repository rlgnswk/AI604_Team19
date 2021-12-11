import torch
import random
import torch.nn.functional as F
from torchvision.utils import save_image
from math import floor
import cv2

class dummy():
    def __init__(self, SR_ratio):
        self.SR_ratio = SR_ratio

def dataAug(lq, args):
    # generate aspect ratio between 128 and shorter side of image
    while True:
        if lq.shape[2] <= lq.shape[3]:
            y = random.randint(64, lq.shape[2])
            x = floor(lq.shape[3] / lq.shape[2] * y)
            if x%args.SR_ratio==0 and y%args.SR_ratio==0:
                break
        else:
            x = random.randint(64, lq.shape[3])
            y = floor(lq.shape[2] / lq.shape[3] * x)
            if x%args.SR_ratio==0 and y%args.SR_ratio==0:
                break

    # downsample / generate hr father
    img_hr = F.interpolate(lq, size=[x, y])
    img_lr = F.interpolate(img_hr, size=[x//args.SR_ratio, y//args.SR_ratio])

    # crop
    img_hr, img_lr = crop(img_hr, img_lr, args.SR_ratio, args.patchSize)

    # get lr sons
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

def crop(img_hr, img_lr, SR_ratio, patch_size):
    _, _, input_size_h, input_size_w = img_lr.shape

    lr_patch_size = patch_size // SR_ratio
    x_start_lr = random.randrange(0, input_size_w - lr_patch_size + 1)
    y_start_lr = random.randrange(0, input_size_h - lr_patch_size + 1)

    hr_patch_size = patch_size
    x_start_hr = 2 * x_start_lr
    y_start_hr = 2 * y_start_lr

    img_hr = img_hr[: ,: , y_start_hr:y_start_hr+hr_patch_size, x_start_hr:x_start_hr+hr_patch_size]
    img_lr = img_lr[: ,: , y_start_lr:y_start_lr+lr_patch_size, x_start_lr:x_start_lr+lr_patch_size]

    return img_hr, img_lr