import torch
import random
import torch.nn.functional as F
from torchvision.utils import save_image
from math import floor
import cv2
import numpy as np

def dataAug(lq, args):
    # generate aspect ratio between 128 and shorter side of image
    #if original image is less than 128, the patch size is given to 64
    mi=min(lq.shape[2], lq.shape[3])
    if mi<=128:
        patch_size=64
    else:
        patch_size = args.patchSize


    sizes=[]
    if lq.shape[2] <= lq.shape[3]:
        for i in range(patch_size, lq.shape[2]):
            x=i
            y = floor(lq.shape[3] / lq.shape[2] * x)
            size=(x,y)
            sizes.append(size)

    else:
        for j in range(patch_size, lq.shape[3]):
            y = j
            x = floor(lq.shape[2] / lq.shape[3] * y)
            size = (x, y)
            sizes.append(size)

    weights = np.float32([x * y / (lq.shape[2] * lq.shape[3]) for (x, y) in sizes])
    pair_prob = weights / np.sum(weights)

    x, y = random.choices(sizes, weights=pair_prob, k=1)[0]
    # downsample / generate hr father
    img_hr = F.interpolate(lq, size=[x, y], mode='bicubic', align_corners=True)
    img_lr = F.interpolate(img_hr, size=[x // args.SR_ratio, y // args.SR_ratio], mode='bicubic',
                           align_corners=True)  # Downsample
    img_lr = F.interpolate(img_lr, size=[x, y], mode='bicubic', align_corners=True)

    # crop
    img_hr, img_lr = crop(img_hr, img_lr, min(args.patchSize, min(x,y)))

    # get lr sons
    hr_fathers = img_hr
    lr_sons = img_lr

    for j in range(4):
        rot_hr = torch.rot90(img_hr, j, (2, 3))
        rot_lr = torch.rot90(img_lr, j, (2, 3))

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
