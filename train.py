import torch
import torch.nn as nn  #
import torch.nn.functional as F  # various activation functions for model
import torchvision  # You can load various Pretrained Model from this package
import torchvision.datasets as vision_dsets
import torchvision.transforms as T  # Transformation functions to manipulate images
import torch.optim as optim  # various optimization functions for model
from torch.autograd import Variable
from torch.utils import data
from torchvision.utils import save_image
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import models
import random
from utils import *
import argparse
import time
from datetime import datetime
import math
from math import log10
from torch.nn import init
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from PIL import Image as PIL_image
from data import *
from test_metric import *
parser = argparse.ArgumentParser()

parser.add_argument('--name', default='test', help='save result')
parser.add_argument('--gpu', type=int)
parser.add_argument('--saveDir', default='./results', help='datasave directory')

# dataPath
parser.add_argument('--GT_path', type=str, default='./datasets/GT')
parser.add_argument('--LR_path', type=str, default='./datasets/LR')
# model parameters
parser.add_argument('--input_channel', type=int, default=3, help='netSR and netD input channel')
parser.add_argument('--mid_channel', type=int, default=64, help='netSR middle channel')
parser.add_argument('--nThreads', type=int, default=0, help='number of threads for data loading')

# training parameters
parser.add_argument('--SR_ratio', type=int, default=2, help='SR ratio')
parser.add_argument('--patchSize', type=int, default=256, help='patch size (GT)')
parser.add_argument('--batchSize', type=int, default=9, help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lrDecay', type=int, default=100, help='epoch of half lr')
parser.add_argument('--decayType', default='inv', help='lr decay function')
parser.add_argument('--iter', type=int, default=2000, help='number of iterations to train')
parser.add_argument('--period', type=int, default=100, help='period of evaluation')
parser.add_argument('--kerneltype', default='g02', help='save result')
args = parser.parse_args()

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


def test(save, netG, lq, gt, img, iters):
    with torch.no_grad():
        input_img = F.interpolate(lq, scale_factor=2, mode='bicubic')

        output = netG(input_img)
        save_image(output, save.save_dir_image + "/out" + str(iters) + '.png')
        psnr = get_psnr(output, gt)
        ssim = get_ssim(output, gt)

    return psnr, ssim

def train(args):
    tot_loss_G = 0
    tot_loss_D = 0
    tot_loss_Recon = 0

    start = datetime.now()
    gt_path = sorted(os.listdir(args.GT_path))
    length = len(gt_path)

    for idx in range(length):  # num of data

        gt_pi = PIL_image.open(args.GT_path + "/" + gt_path[idx]).convert('RGB')
        lq_pi = PIL_image.open(args.LR_path + "/" + gt_path[idx]).convert('RGB')

        gt = RGB_np2Tensor(np.array(gt_pi)).cuda()
        lq = RGB_np2Tensor(np.array(lq_pi)).cuda()
        gt=torch.unsqueeze(gt, dim=0)
        lq=torch.unsqueeze(lq, dim=0)
        hr_fathers1, lr_sons1 = dataAug(lq)
        hr_fathers2, lr_sons2 = dataAug(lq)
        hr_fathers3, lr_sons3 = dataAug(lq)
        hr_fathers4, lr_sons4 = dataAug(lq)

        netD = models.netD(input_channel=args.input_channel, mid_channel=args.mid_channel)
        criterion_D = nn.BCELoss()
        optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr)
        netD.train()

        netG = models.netSR(input_channel=args.input_channel, mid_channel=args.mid_channel)
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
        save = saveData(args, gt_path[idx][:-4])

        for iters in range(args.iter):
            for j in range(4):
                if j==0:
                    im_lr=Variable(lr_sons1)
                    im_hr=Variable(hr_fathers1)
                elif j==1:
                    im_lr=Variable(lr_sons2)
                    im_hr=Variable(hr_fathers2)
                elif j==2:
                    im_lr=Variable(lr_sons3)
                    im_hr=Variable(hr_fathers3)
                else:
                    im_lr=Variable(lr_sons4)
                    im_hr=Variable(hr_fathers4)

            for p in netD.parameters():
                p.requires_grad = False


            optimizer_G.zero_grad()
            input_img = F.interpolate(im_lr, scale_factor=2, mode='bicubic')
            output_SR = netG(input_img)
            output_fake = netD(output_SR)

            true_labels = Variable(torch.ones_like(output_fake))
            fake_labels = Variable(torch.zeros_like(output_fake))

            loss_G = criterion_G(output_fake, true_labels)  # GAN Loss
            loss_Recon = criterion_Recon(output_SR, im_hr)  # Reconstruction Loss

            alpha = 1.0  # I(gihoon) think It should be bigger.
            loss_G_total = loss_G + loss_Recon
            loss_G_total.backward()
            optimizer_G.step()

            for p in netD.parameters():
                p.requires_grad = True
            optimizer_D.zero_grad()

            output_real = netD(im_hr)
            D_fake = output_fake.detach()
            loss_D_fake = criterion_D(D_fake, fake_labels)
            loss_D_real = criterion_D(output_real, true_labels)
            loss_D_total = loss_D_fake / 2 + loss_D_real / 2

            loss_D_total.backward()
            optimizer_D.step()

            tot_loss_Recon += loss_Recon
            tot_loss_G += loss_G
            tot_loss_D += loss_D_total

            if (iters + 1) % args.period == 0:
                # test

                netG.eval()
                psnr, ssim = test(save, netG, lq, gt, gt_path[idx], iters=iters)
                # print
                lossD = tot_loss_D / ((args.batchSize) * args.period)
                lossGAN = tot_loss_G / ((args.batchSize) * args.period)
                lossRecon = tot_loss_Recon / ((args.batchSize) * args.period)
                end = datetime.now()
                iter_time = (end - start)
                # total_time = total_time + iter_time

                log = "[{} / {}] \tReconstruction Loss: {:.5f}\t Generator Loss: {:.4f} \t Discriminator Loss: {:.4f} \t Time: {} \t PSNR: {:.4f} \t SSIM: {:.4f} ".format(
                    iters + 1, args.iter, lossRecon, lossGAN, lossD, iter_time, psnr, ssim)
                print(log)
                save.save_log(log)
                save.save_model(netG, iters, idx)
                netG.train()
                tot_loss_Recon = 0
                tot_loss_G = 0
                tot_loss_D = 0


if __name__ == '__main__':
    train(args)
