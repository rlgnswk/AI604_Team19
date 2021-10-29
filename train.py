import torch 
import torch.nn as nn #
import torch.nn.functional as F # various activation functions for model
import torchvision # You can load various Pretrained Model from this package 
import torchvision.datasets as vision_dsets
import torchvision.transforms as T # Transformation functions to manipulate images
import torch.optim as optim # various optimization functions for model
from torch.autograd import Variable 
from torch.utils import data
from torchvision.utils import save_image
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import models

from utils import *
import argparse
import time
from datetime import datetime 
import math
from math import log10
from torch.nn import init
import numpy as np
from data_utils import New_trainDataset, New_testDataset
from PIL import Image
from torchvision.utils import save_image

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

# xavier initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)

# setting learning rate decay
def set_lr(args, epoch, optimizer):
    lrDecay = args.lrDecay
    decayType = args.decayType
    if decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = args.lr / 2 ** epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = args.lr * math.exp(-k * epoch)
    elif decayType == 'inv':
        k = 1 / lrDecay
        lr = args.lr / (1 + k * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# parameter counter
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#test
def test(save, netG, testdataloader, i_th_image, iters = 0):
    netG.eval()
    with torch.no_grad():
        test_image = testdataloader.getitem()
        output = netG(test_image.cuda())
        output = output.cpu()
        output = output.data.squeeze(0)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        for t, m, s in zip(output, mean, std):
            t.mul_(s).add_(m)

        imagename = testdataloader.gt_path[i_th_image][:-4]
        save_image(output, save.save_dir_image +"/"+imagename+str(iters)+'.png')
    netG.train()


# training
def train(args):
    ################################# for gihoon's part start
    netD = models.netD(input_channel = args.input_channel, mid_channel = args.mid_channel)
    criterion_D = nn.BCELoss()
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr)
    netD.train()

    netG = models.netSR(input_channel = args.input_channel, mid_channel = args.mid_channel)
    criterion_G = nn.BCELoss()
    criterion_Recon = nn.L1Loss()
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr)
    netG.train()

    netD.apply(weights_init)
    netG.apply(weights_init)
    netD.cuda()
    netG.cuda()

    criterion_G.cuda()
    criterion_D.cuda()
    ##################################for gihoon's part end


    # load data
    ################################# for yujin's part start
    #trainDataLoader, validDataLoader , testDataLoader = data_utils.Load_Data(args.datapath)
    trainDataLoader = New_trainDataset(args)
    testdataloader = New_testDataset(args)
    ################################# for yujin's part end

    #writer = SummaryWriter()



    #learning_rate_G = set_lr(args, args.iter, optimizer_G)
    #learning_rate_D = set_lr(args, args.iter, optimizer_D)
    tot_loss_G=0
    tot_loss_D=0
    tot_loss_Recon=0

    start = datetime.now()
    for i_th_image in range(len(os.listdir(args.GT_path))): # num of data
        trainDataLoader = New_trainDataset(args, i_th_image)
        testdataloader = New_testDataset(args, i_th_image)
        # load saveData class from utils module
        imagename = testdataloader.gt_path[i_th_image][:-4]
        save = saveData(args, imagename)

        for iters in range(args.iter):
            
            im_lr, im_hr  = trainDataLoader.getitem()
            im_lr = Variable(im_lr.cuda())
            im_hr = Variable(im_hr.cuda())
            im_lr_up=F.interpolate(im_lr, scale_factor=args.SR_ratio, mode='bicubic', align_corners=True)
            #print(im_lr.shape)
            #print(im_lr.requires_grad) # true
            #print(im_hr.requires_grad) # true

            # --------------------
            # Train Generator
            # --------------------
            for p in netD.parameters():
                p.requires_grad = False

            optimizer_G.zero_grad()

            output_SR = netG(im_lr_up)
            output_fake=netD(output_SR)

            true_labels = Variable(torch.ones_like(output_fake))
            fake_labels = Variable(torch.zeros_like(output_fake))

            loss_G = criterion_G(output_fake, true_labels) # GAN Loss
            loss_Recon = criterion_Recon(output_SR, im_hr) # Reconstruction Loss

            alpha = 1.0 # I(gihoon) think It should be bigger.
            loss_G_total = loss_G + loss_Recon
            #loss_G_total = loss_G
            loss_G_total.backward()
            optimizer_G.step()

            # --------------------
            # Train Discriminator
            # --------------------
            for p in netD.parameters():
                p.requires_grad = True
            optimizer_D.zero_grad()

            output_real=netD(im_hr.detach())
            output_fake=netD(output_SR.detach())
            
            loss_D_fake = criterion_D(output_fake, fake_labels)
            loss_D_real = criterion_D(output_real, true_labels)
            loss_D_total = loss_D_fake/2  + loss_D_real/2

            loss_D_total.backward()
            optimizer_D.step()

            tot_loss_Recon += loss_Recon
            tot_loss_G += loss_G
            tot_loss_D += loss_D_total

            if (iters + 1) % args.period == 0:
                #test
                test(save, netG, testdataloader, i_th_image, iters = iters)
                #print
                lossD = tot_loss_D / ((args.batchSize) * args.period)
                lossGAN = tot_loss_G / ((args.batchSize) * args.period)
                lossRecon = tot_loss_Recon / ((args.batchSize) * args.period)
                end = datetime.now()
                iter_time = (end - start)
                #total_time = total_time + iter_time    

                log = "[{} / {}] \tReconstruction Loss: {:.5f}\t Generator Loss: {:.4f} \t Discriminator Loss: {:.4f} \t Time: {}".format(iters + 1, args.iter, lossRecon, lossGAN, lossD, iter_time)
                print(log)
                save.save_log(log)
                save.save_model(netG, iters)
                
                tot_loss_Recon = 0
                tot_loss_G = 0 
                tot_loss_D = 0
    #writer.close()

if __name__ == '__main__':
    train(args)
