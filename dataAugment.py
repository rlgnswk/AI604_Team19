import torch
import torch.nn.functional as F
import random
import os
import numpy as np
import cv2
import os.path

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

data_path='./datasets'
data_type='real'
data_dir=os.path.join(data_path, data_type)
fileList = os.listdir(data_dir)
nTrain = len(fileList)-1

def dataAug(imglr):
    if random.random() < 0.3:  # horizontal flip
        imglr = imglr[:, ::-1, :]


    if random.random() < 0.3:  # vertical flip
        imglr =  imglr[::-1, :, :]

    rot = random.randint(0, 3)  # rotate
    imglr = np.rot90(imglr, rot, (0, 1))

    crop_size_w=random.randint(0,20)
    crop_size_h=random.randint(0,20)

    input_size_h, input_size_w, _= imglr.shape
    x_start = random.randint(0, input_size_w - crop_size_w)
    y_start = random.randint(0, input_size_h - crop_size_h)

    img_lr = imglr[y_start: y_start + crop_size_h, x_start: x_start + crop_size_w,:]

    return  img_lr



for i in range(nTrain):
    name = fileList[1]
    nameTar = os.path.join(data_dir, name)
    saveTar=os.path.join(data_dir, 'Test')
    imgTar=cv2.imread(nameTar)
    for j in range(100):
        img_aug=dataAug(imgTar)
        ts = (2, 0, 1)
        tm = (1, 2, 0)
        img_aug = torch.Tensor(img_aug.transpose(ts).astype(float)).mul_(1.0)
        img_aug = torch.unsqueeze(img_aug, 0)
        img_aug = torch.squeeze(img_aug, 0)
        img_aug = img_aug.detach().cpu().numpy()
        img_aug = img_aug.transpose(tm).astype(float)


        k=os.path.join(saveTar, str(j), '.png')
        cv2.imwrite(k, img_aug)
