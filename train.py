import torch
import torch.nn as nn  #

from torch.autograd import Variable
import model
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
from metric import *
from lpips import lpips

parser = argparse.ArgumentParser()

parser.add_argument('--name', default='test', help='save result')
parser.add_argument('--gpu', type=int)
parser.add_argument('--saveDir', default='./results', help='datasave directory')
parser.add_argument('--load', default='NetFinal', help='save result')

# dataPath
parser.add_argument('--datapath', type=str, default='./MyDataset_AI604')
parser.add_argument('--dataset', type=str, default='MySet5x2')
parser.add_argument('--kerneltype', default='g20_non_ideal_LR', help='save result')
# model parameters
parser.add_argument('--input_channel', type=int, default=3, help='netSR and netD input channel')
parser.add_argument('--mid_channel', type=int, default=64, help='netSR middle channel')
parser.add_argument('--nThreads', type=int, default=0, help='number of threads for data loading')

# training parameters
parser.add_argument('--SR_ratio', type=int, default=2, help='SR ratio')
parser.add_argument('--patchSize', type=int, default=128, help='patch size (GT)')
parser.add_argument('--batchSize', type=int, default=12, help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lrDecay', type=int, default=100, help='epoch of half lr')
parser.add_argument('--decayType', default='inv', help='lr decay function')
parser.add_argument('--iter', type=int, default=3000, help='number of iterations to train')
parser.add_argument('--period', type=int, default=10, help='period of evaluation')
parser.add_argument('--numhr', type=int, default=15, help='period of evaluation')
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

def Average_list(lst):
    return sum(lst) / len(lst)

def test(save, netG, input, gt, iters, num):
    netG.eval()
    input = F.interpolate(input, scale_factor=args.SR_ratio, mode='bicubic')
    with torch.no_grad():
        output = netG(input)

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
    output_rgb[:, :, 0] = output[0]
    output_rgb[:, :, 1] = output[1]
    output_rgb[:, :, 2] = output[2]
    out = PIL_image.fromarray(np.uint8(output_rgb), mode='RGB')
    out.save(args.saveDir + '/img_' + str(num) + '_iter_' + str(iters) + '.png')

    return psnr, ssim, lpips_score

def train(args):
    tot_loss_G = 0
    tot_loss_D = 0
    tot_loss_Recon = 0
    tot_loss_per=0
    GT_path = os.path.join(args.datapath, args.dataset, 'HR')
    LR_path = os.path.join(args.datapath, args.dataset, args.kerneltype)

    start = datetime.now()
    gt_path = sorted(os.listdir(GT_path))
    length = len(gt_path)

    tot_psnr_list = []
    tot_ssim_list = []
    tot_lpips_list = []

    total_min_lipis_index_list = []


    for num in range(length):  # num of data
        #load file
        gt_pi = PIL_image.open(GT_path + "/" + gt_path[num]).convert('RGB')
        lq_pi = PIL_image.open(LR_path + "/" + gt_path[num]).convert('RGB')

        gt = RGB_np2Tensor(np.array(gt_pi)).cuda()
        lq = RGB_np2Tensor(np.array(lq_pi)).cuda()

        gt = torch.unsqueeze(gt, dim=0)
        lq = torch.unsqueeze(lq, dim=0)

        #define network
        netD = model.netD(input_channel=args.input_channel, mid_channel=args.mid_channel)
        optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

        netG = model.netSR(input_channel=args.input_channel, mid_channel=args.mid_channel)
        optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))

        #define loss
        criterion_D = nn.BCELoss().cuda()
        criterion_G = nn.BCELoss().cuda()
        criterion_Recon = nn.L1Loss().cuda()
        criterion_mse = nn.L1Loss().cuda()


        #weight initialize
        netD.apply(weights_init)
        netG.apply(weights_init)

        #model to cuda
        vgg = model.Vgg16(requires_grad=False)
        vgg.cuda()

        netD.cuda()
        netG.cuda()

        #model to train
        netD.train()
        netG.train()

        save = saveData(args, gt_path[num])
        psnr_list = []
        ssim_list = []
        lpips_list = []


        for iters in range(args.iter):
            #set lr
            netG_lr = set_lr(args, iters, optimizer_G)
            netD_lr = set_lr(args, iters, optimizer_D)

            hr_fathers, lr_sons = dataAug(lq, args)

            output_SR = netG(lr_sons)

            # update D
            for p in netD.parameters():
                p.requires_grad = True
            netD.zero_grad()

            # real image
            output_real = netD(hr_fathers)
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

            loss_Recon = 1*criterion_Recon(output_SR, hr_fathers)          # Reconstruction Loss
            loss_Recon.backward(retain_graph=True)

            alpha = 1# I(gihoon) think It should be bigger

            loss_G = alpha*criterion_G(netD(output_SR), true_labels)      # GAN Loss
            loss_G.backward(retain_graph=True)

            loss_per =alpha* criterion_mse(vgg(output_SR), vgg(hr_fathers))
            loss_per.backward()
            optimizer_G.step()


            tot_loss_Recon += loss_Recon
            tot_loss_G += loss_G
            tot_loss_D += loss_D_total
            tot_loss_per += loss_per


            if (iters + 1) % args.period == 0:
                # test
                psnr, ssim, lpips_score = test(save, netG, lq, gt, iters,num)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
                lpips_list.append(lpips_score)
                netG.train()

                # loss
                lossD = tot_loss_D / args.period
                lossGAN = tot_loss_G / args.period
                lossRecon = tot_loss_Recon / args.period
                lossPercep=tot_loss_per / args.period


                end = datetime.now()
                iter_time = (end - start)

                #print
                log = "[{} / {}]Generator lr: {} \tReconstruction Loss: {:.8f}\t Generator Loss: {:.8f} \t Perceptual Loss: {:.8f} \t Discriminator Loss: {:.8f} \t PSNR: {:.4f} \t SSIM: {:.4f} \t LPIPS: {:.4f} \t Time: {}".format(
                    iters + 1, args.iter, netG_lr, lossRecon, lossGAN,lossPercep,  lossD, psnr, ssim, lpips_score, iter_time)
                print(log)
                save.save_log(log)
                save.save_model(netG, iters, gt_path[num])

                #initialize loss
                tot_loss_Recon = 0
                tot_loss_G = 0
                tot_loss_D = 0
                tot_loss_per=0

        tmp = min(lpips_list)
        min_lpips_index = lpips_list.index(tmp)

        tot_psnr_list.append(psnr_list[min_lpips_index])
        tot_ssim_list.append(ssim_list[min_lpips_index])
        tot_lpips_list.append(lpips_list[min_lpips_index])

        total_min_lipis_index_list.append((min_lpips_index+1)*args.period)

    log = "Avg Psnr: {}".format(Average_list(tot_psnr_list))
    save.save_log(log)
    print(log)

    log = "Avg ssim: {}".format(Average_list(tot_ssim_list))
    save.save_log(log)
    print(log)

    log = "Avg lpips: {}".format(Average_list(tot_lpips_list))
    save.save_log(log)
    print(log)

    for a in range(length):
        log = "{}th image index: {}".format(a+1, total_min_lipis_index_list[a])
        save.save_log(log)


if __name__ == '__main__':
    train(args)
