import torch
import random
import torch.nn.functional as F

def dataAug(lq, args):
    a = lq.shape[2]
    b = lq.shape[3]
    a_mod = random.randint(a - 10, a - 2)
    b_mod = random.randint(b - 10, b - 2)
    img_hr=F.interpolate(lq, size=[a_mod,b_mod])
    img_lr=F.interpolate(lq,  size=[a_mod//args.SR_ratio,b_mod//args.SR_ratio])
    img_lr, img_hr=crop(img_lr,img_hr, 128,64)
    lr_sons=img_lr
    hr_fathers=img_hr
    for j in range(4):
        rot_lr=torch.rot90(img_lr, j, (2, 3))
        rot_hr = torch.rot90(img_hr, j, (2,3))


        for k in range(2):
            flip_lr = rot_lr.flip(k)
            flip_hr = rot_hr.flip(k)
            lr_sons = torch.cat([lr_sons, flip_lr], dim=0)
            hr_fathers = torch.cat([hr_fathers, flip_hr], dim=0)


    return hr_fathers, lr_sons


def RGB_np2Tensor(img_lr):
    # to Tensor
    ts = (2, 0, 1)
    img_lr = torch.Tensor(img_lr.transpose(ts).astype(float)).mul_(1.0)
    # normalization [-1,1]
    img_lr = (img_lr / 255.0 - 0.5) * 2
    return img_lr

def crop(imglr, imghr, cropgt, croplr):
    _,_,input_size_h, input_size_w= imglr.shape
    x_start = random.randrange(0, input_size_w - croplr)
    y_start = random.randint(0, input_size_h - croplr)
    (x_gt, y_gt) = (2 * x_start, 2 * y_start)
    imglr = imglr[:,:,y_start: y_start + croplr, x_start: x_start + croplr]
    imghr = imghr[:,:,y_gt: y_gt + cropgt, x_gt: x_gt + cropgt]

    return imglr, imghr
