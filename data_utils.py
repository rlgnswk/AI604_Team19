import torch
import torch.nn.functional as F
import random
import os
import numpy as np
import cv2
import os.path
from torch.autograd import Variable
from PIL import Image

from torchvision.utils import save_image
import torchvision

from PIL import Image as PIL_image

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

def RGB_np2Tensor(img_lr):

    # to Tensor
    ts = (2, 0, 1)
    img_lr = torch.Tensor(img_lr.transpose(ts).astype(float)).mul_(1.0)
    # normalization [-1,1]
    img_lr = (img_lr / 255.0 - 0.5) * 2
    return img_lr

#it's not have to use Dataset library.
class New_trainDataset():
    def __init__(self, args, idx = 0):
        self.args=args
        self.GT_path = args.GT_path
        self.LR_path = args.LR_path
        self.gt_path = sorted(os.listdir(self.GT_path))
        #self.length = len(self.gt_path)

        self.batchSize = args.batchSize
        self.crop_gt = args.patchSize
        self.crop_lr = self.crop_gt // args.SR_ratio
        self.input_channel_size = args.input_channel
        
        gt = PIL_image.open(self.GT_path +"\\"+ self.gt_path[idx]).convert('RGB')
        lq = PIL_image.open(self.LR_path +"\\"+ self.gt_path[idx]).convert('RGB')        
        self.gt_array = np.array(gt)
        self.lq_array = np.array(lq)

    def getitem(self):

        img_gt_batch = torch.ones([self.batchSize, self.input_channel_size, self.crop_gt, self.crop_gt], dtype=torch.float32, requires_grad=True)
        img_lr_batch = torch.ones([self.batchSize, self.input_channel_size, self.crop_lr, self.crop_lr], dtype=torch.float32, requires_grad=True)
        
        for i in range(self.batchSize):        
            imglr, imghr  = dataAug(self.lq_array, self.gt_array)
            img_gt_batch[i,:,:,:] = RGB_np2Tensor(imghr)
            img_lr_batch[i,:,:,:] = RGB_np2Tensor(imglr)
        #print(img_gt_batch.shape)
        return img_lr_batch, img_gt_batch

class New_testDataset():
    def __init__(self, args, idx = 0):
        self.args=args
        self.GT_path = args.GT_path
        #self.LR_path = r'C:\Users\VML\Documents\GitHub\AI604_Team19\datasets\Set5\temp\LR'
        self.gt_path = sorted(os.listdir(self.GT_path))
         
        self.batchSize = 1
        self.input_channel_size = args.input_channel


        gt = PIL_image.open(self.GT_path +"\\"+ self.gt_path[idx]).convert('RGB')
     
        self.gt_array = np.array(gt)


    def getitem(self):

        test_image = torch.ones([self.batchSize, self.gt_array.shape[2], self.gt_array.shape[0], self.gt_array.shape[1]], dtype=torch.float32, requires_grad=True)
        
        for i in range(self.batchSize):        
            #imglr, imghr  = dataAug(self.lq_array, self.gt_array)
            test_image[i,:,:,:] = RGB_np2Tensor(self.gt_array)

        return test_image



if __name__ == '__main__':
    import argparse
    print("Start: check Dataloader")
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='test', help='save result')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--saveDir', default='./results', help='datasave directory')

    #dataPath
    parser.add_argument('--GT_path', type=str, default=r'.\dataset\GT')
    parser.add_argument('--LR_path', type=str, default=r'.\dataset\LR')

    #model parameters
    parser.add_argument('--input_channel', type=int, default=3, help='netSR and netD input channel')
    parser.add_argument('--mid_channel', type=int, default=64, help='netSR middle channel')
    parser.add_argument('--nThreads', type=int, default=0, help='number of threads for data loading')

    #training parameters
    parser.add_argument('--SR_ratio', type=int, default=2, help='SR ratio')
    parser.add_argument('--patchSize', type=int, default=256, help='patch size (GT)')
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lrDecay', type=int, default=100, help='epoch of half lr')
    parser.add_argument('--decayType', default='inv', help='lr decay function')
    parser.add_argument('--iter', type=int, default=2000, help='number of iterations to train')
    parser.add_argument('--period', type=int, default=100, help='period of evaluation')
    parser.add_argument('--kerneltype', default='g02', help='save result')
    args = parser.parse_args()
    

    data_train=New_trainDataset(args)
    data_test=New_testDataset(args)              
    #testdataloader = get_testdataset(args)
    train_batch = 16
    test_batch = 1

    for iter in range(1):
        img_lr_train, img_gt_train  = data_train.getitem()
        img_gt_test  = data_test.getitem()
        print( "train_batch: " ,train_batch, " test_batch: ",test_batch)
        print("img_gt_train.shape" ,img_gt_train.shape)
        print("img_lr_train.shape" ,img_lr_train.shape)
        print("img_gt_test.shape" ,img_gt_test.shape)
        #the color might be unreal because of scaling
        save_image(img_gt_train, './SR_img_train.png')
        save_image(img_gt_test, './SR_img_test.png')
        break
    print("Done: check Dataloader")