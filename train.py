import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torchvision.utils import save_image

import os
import argparse
import random
import cv2
import time
from datetime import datetime
import math
from math import log10
import numpy as np
from PIL import Image

from utils import *
import models
from data import *
from metric import *
from lpips import lpips


parser = argparse.ArgumentParser()

parser.add_argument('--name', default='test', help='save result')
parser.add_argument('--gpu', type=int)
parser.add_argument('--saveDir', default='./results', help='datasave directory')
parser.add_argument('--load', default='NetFinal', help='save result')

# dataPath
parser.add_argument('--data_dir', type=str, default='./MyDataset_AI604')
parser.add_argument('--dataset', type=str, default='MySet5x2')
parser.add_argument('--GT_path', type=str, default='HR')
parser.add_argument('--LR_path', type=str, default='g20_non_ideal_LR')

# model parameters
parser.add_argument('--input_channel', type=int, default=3, help='netSR and netD input channel')
parser.add_argument('--mid_channel', type=int, default=64, help='netSR middle channel')
parser.add_argument('--nThreads', type=int, default=0, help='number of threads for data loading')

# training parameters
parser.add_argument('--SR_ratio', type=int, default=2, help='SR ratio')
parser.add_argument('--patchSize', type=int, default=64, help='patch size (GT)')
parser.add_argument('--batchSize', type=int, default=12, help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lrDecay', type=int, default=50, help='epoch of half lr')
parser.add_argument('--decayType', default='step', help='lr decay function')
parser.add_argument('--iter', type=int, default=200, help='number of iterations to train')
parser.add_argument('--period', type=int, default=100, help='period of evaluation')
parser.add_argument('--kerneltype', default='g02', help='kernel type')

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

def test(save, netG, lq, gt, idx, iters):
    save_dir = os.path.join(args.saveDir, 'test_output')
    if not os.path.exists(os.path.join(save_dir)):
        os.makedirs(save_dir)

    with torch.no_grad():
        lq = Variable(lq.cuda(), volatile=False)
        gt = Variable(gt.cuda())
        input_img = F.interpolate(lq, scale_factor=args.SR_ratio, mode='bicubic')
        output = netG(input_img)

    psnr = get_psnr(output, gt)
    ssim = get_ssim(output, gt)
    lpips_score = lpips(output, gt, net_type='vgg').item()

    # saving image
    output = output.cpu()
    output = output.data.squeeze(0)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for t, m, s in zip(output, mean, std):
        t.mul_(s).add_(m)
    output = output.numpy()
    output *= 255.0
    output = output.clip(0, 255)
    sp_0, sp_1, sp_2 = output.shape
    output_rgb = np.zeros((sp_1, sp_2, sp_0))
    output_rgb[:, :, 0] = output[2]
    output_rgb[:, :, 1] = output[1]
    output_rgb[:, :, 2] = output[0]
    out = Image.fromarray(np.uint8(output_rgb), mode='RGB')
    out.save(save_dir + '/img_' + str(idx) + '_iter_' + str(iters) + '.png')

    return psnr, ssim, lpips_score

def train(args):
    gt_path = os.path.join(args.data_dir, args.dataset, args.GT_path)
    lr_path = os.path.join(args.data_dir, args.dataset, args.LR_path)

    gt_filelist = sorted([os.path.join(gt_path, img) for img in os.listdir(gt_path)])
    lr_filelist = sorted([os.path.join(lr_path, img) for img in os.listdir(lr_path)])

    tot_loss_G = 0
    tot_loss_D = 0
    tot_loss_Recon = 0
    tot_loss_Perc = 0
    idx = 0

    for gt_file, lr_file in zip(gt_filelist, lr_filelist):
        idx += 1
        print('Image {}:'.format(idx))

        gt_pi = cv2.imread(gt_file)
        lq_pi = cv2.imread(lr_file)

        gt = RGB_np2Tensor(gt_pi).cuda()
        lq = RGB_np2Tensor(lq_pi).cuda()

        gt = gt.unsqueeze(0)
        lq = lq.unsqueeze(0)

        netD = models.netD(input_channel=args.input_channel, mid_channel=args.mid_channel)
        optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
        criterion_D = nn.BCELoss()       

        netG = models.netSR(input_channel=args.input_channel, mid_channel=args.mid_channel)
        optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
        criterion_G = nn.BCELoss()
        criterion_Recon = nn.L1Loss()

        vgg = models.VGG16(requires_grad=False).cuda()
        criterion_vgg = nn.L1Loss()

        netD.apply(weights_init)
        netG.apply(weights_init)

        netD.cuda()
        netG.cuda()
        criterion_G.cuda()
        criterion_D.cuda()

        netD.train()
        netG.train()

        save = saveData(args)

        for iters in range(args.iter):
            # netG_lr = set_lr(args, iters, optimizer_G)
            # netD_lr = set_lr(args, iters, optimizer_D)

            hr_fathers, lr_sons = dataAug(lq, args)
            im_lr = Variable(lr_sons)
            im_hr = Variable(hr_fathers)

            input_img = F.interpolate(im_lr, scale_factor=args.SR_ratio, mode='bicubic')
            output_SR = netG(input_img)

            # update D
            for p in netD.parameters():
                p.requires_grad = True
            netD.zero_grad()

            # real image
            output_real = netD(im_hr)
            true_labels = Variable(torch.ones_like(output_real))
            loss_D_real = criterion_D(output_real, true_labels)

            # fake image
            fake_image = output_SR.detach()
            D_fake = netD(fake_image)
            fake_labels = Variable(torch.zeros_like(D_fake))
            loss_D_fake = criterion_D(D_fake, fake_labels)
            
            # total D loss
            loss_D_total = 0.5 * (loss_D_fake + loss_D_real)
            loss_D_total.backward()
            optimizer_D.step()

            # update G
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()
            
            loss_Recon = criterion_Recon(output_SR, im_hr)           # Reconstruction Loss
            loss_Perc = criterion_vgg(vgg(output_SR), vgg(im_hr))    # Perceptual Loss
            loss_G = criterion_G(netD(output_SR), true_labels)       # GAN Loss

            alpha = 0.01  # I(gihoon) think It should be bigger.
            loss_G_total = loss_Recon + alpha * loss_G + alpha * loss_Perc
            loss_G_total.backward()
            optimizer_G.step()

            tot_loss_Recon += loss_Recon
            tot_loss_Perc += loss_Perc
            tot_loss_G += loss_G
            tot_loss_D += loss_D_total

            if (iters + 1) % args.period == 0:
                # test
                netG.eval()
                psnr, ssim, lpips_score = test(save, netG, lq, gt, idx, iters)
                netG.train()
                
                lossD = tot_loss_D / args.period
                lossGAN = tot_loss_G / args.period
                lossRecon = tot_loss_Recon / args.period
                lossPerc = tot_loss_Perc / args.period

                # print
                log = "[{} / {}] \t Reconstruction Loss: {:.8f} \t Perceptual Loss: {:.8f} \t Generator Loss: {:.8f} \t Discriminator Loss: {:.8f} \t PSNR: {:.4f} \t SSIM: {:.4f} \t LPIPS: {:.4f}".format(iters + 1, args.iter, lossRecon, lossPerc, lossGAN, lossD, psnr, ssim, lpips_score)
                print(log)
                save.save_log(log)
                save.save_model(netG, iters)

                tot_loss_Recon = 0
                tot_loss_G = 0
                tot_loss_D = 0
                tot_loss_Perc = 0

if __name__ == '__main__':
    train(args)