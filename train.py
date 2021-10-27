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
from data_utils import Dataset
import argparse
import time
import math
from math import log10
from torch.nn import init
import numpy as np
from data_utils import Dataset, testDataset
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
parser.add_argument('--gpu', type=int)
parser.add_argument('--datasetPath', type=str, default='./datasets')
parser.add_argument('--datasettype', type=str, default='Set5')

#model parameters
parser.add_argument('--input_channel', type=int, default=3, help='netSR and netD input channel')
parser.add_argument('--mid_channel', type=int, default=64, help='netSR middle channel')
parser.add_argument('--nThreads', type=int, default=8, help='number of threads for data loading')
parser.add_argument('--load', default='NetFinal', help='save result')
#training parameters
parser.add_argument('--patchSize', type=int, default=128, help='patch size (GT)')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lrDecay', type=int, default=100, help='epoch of half lr')
parser.add_argument('--decayType', default='inv', help='lr decay function')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
parser.add_argument('--period', type=int, default=10, help='period of evaluation')
parser.add_argument('--saveDir', default='./result', help='datasave directory')
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
#yj
def get_dataset(args):
    data_train=Dataset(args)
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batchSize,
                                             drop_last=True, shuffle=True, num_workers=int(args.nThreads), pin_memory=False)
    return dataloader

def get_testdataset(args):
    data_test=testDataset(args)
    dataloader = torch.utils.data.DataLoader(data_test, batch_size=args.batchSize,
                                             drop_last=True, shuffle=True, num_workers=int(args.nThreads), pin_memory=False)
    return dataloader


#test
def test(args, modelG, dataloader):
    count=0
    for batch, (im_lr, im_hr) in enumerate(dataloader):
        count = count + 1
        with torch.no_grad():
            im_lr = Variable(im_lr.cuda(), volatile=False)
            im_hr = Variable(im_hr.cuda())
            output = modelG(im_lr)

        # denormalizing the output
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
        output_rgb = np.zeros((sp_1,sp_2,sp_0))
        output_rgb[:, :, 0] = output[2]
        output_rgb[:, :, 1] = output[1]
        output_rgb[:, :, 2] = output[0]
        out = Image.fromarray(np.uint8(output_rgb), mode='RGB')
        out.save('result/NetFinal/SR_img_%03d.png' % (count))

# training
def train(args):
    ################################# for gihoon's part start
    netD = models.netD(input_channel = args.input_channel, mid_channel = args.mid_channel)
    criterion_D = nn.BCELoss()
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=args.lr)

    netG = models.netSR(input_channel = args.input_channel, mid_channel = args.mid_channel)
    criterion_G = nn.BCELoss()
    optimizer_G = torch.optim.Adam(netG.parameters(), lr=args.lr)

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
    trainDataLoader = get_dataset(args)
    testdataloader=get_testdataset(args)
    ################################# for yujin's part end


    writer = SummaryWriter()

    # load saveData class from utils module
    save = saveData(args)

    total_time = 0


    for epoch in range(args.epochs):
        start = time.time()

        learning_rate_G = set_lr(args, epoch, optimizer_G)
        learning_rate_D = set_lr(args, epoch, optimizer_D)
        tot_loss_G=0
        tot_loss_D=0

        for batch, (im_lr, im_hr) in enumerate(trainDataLoader):
            im_lr = Variable(im_lr.cuda())
            im_lr=F.interpolate(im_lr, scale_factor=2, mode='bicubic')
            im_hr = Variable(im_hr.cuda())
            output_SR = netG(im_lr)
            output_real=netD(im_hr)
            output_fake=netD(output_SR)
            # --------------------
            # Train Generator
            # --------------------

            valid_hr = Variable(torch.ones_like(output_real), requires_grad=False)
            fake = Variable(torch.zeros_like(output_fake), requires_grad=False)
            valid_lr = Variable(torch.ones_like(output_fake), requires_grad=False)

            loss_G = criterion_G(output_fake, valid_lr)

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # --------------------
            # Train Discriminator
            # --------------------


            loss_D = criterion_D( output_fake.detach(), fake)+criterion_D(output_real, valid_hr)
            tot_loss_G += loss_G
            tot_loss_D += loss_D

            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

        end = time.time()
        epoch_time = (end - start)
        total_time = total_time + epoch_time
        print(epoch)

        lossD = tot_loss_D / (batch + 1)
        lossG = tot_loss_G / (batch + 1)




        if (epoch + 1) % args.period == 0:
            netG.eval()
            test(args, netG, testdataloader)
            netG.train()
            log = "[{} / {}] \tLearning_rate: {:.5f}\t Generator Loss: {:.4f} \t Discriminator Loss: {:.4f} \t Time: {:.4f}".format(epoch + 1, args.epochs, learning_rate_G, lossG, lossD, total_time)
            print(log)
            save.save_log(log)
            save.save_model(netG, epoch)
            total_time = 0
    #writer.close()

if __name__ == '__main__':
    train(args)
