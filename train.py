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
from torch.utils.tensorboard import SummaryWriter

import models
import data_utils 
from utils import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
parser.add_argument('--gpu', type=int)
parser.add_argument('--datasetPath', type=str)

#training parameters
parser.add_argument('--patchSize', type=int, default=128, help='patch size (GT)')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lrDecay', type=int, default=100, help='epoch of half lr')
parser.add_argument('--decayType', default='inv', help='lr decay function')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--period', type=int, default=10, help='period of evaluation')

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
def test(args, model, dataloader):
    for batch, (im_lr, im_hr) in enumerate(testdataloader):
        count = count + 1
        with torch.no_grad():
            im_lr = Variable(im_lr.cuda(), volatile=False)
            im_hr = Variable(im_hr.cuda())
            output = my_model(im_lr)

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
    netD = models.netD()
    netG = models.netSR()

    criterion_G = 
    optimizer_G = 

    criterion_D = 
    optimizer_D = 

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
    trainDataLoader = data_utils.Load_Data(args.datasetPath)
    ################################# for yujin's part end


    writer = SummaryWriter()

    # load saveData class from utils module
    save = saveData(args)

    total_time = 0

    for epoch in range(args.epochs):
        start = time.time()

        learning_rate_G = set_lr(args, epoch, optimizer_G)
        learning_rate_D = set_lr(args, epoch, optimizer_D)

        for batch, (im_lr, im_hr) in enumerate(trainDataLoader):
            im_lr = Variable(im_lr.cuda())
            im_hr = Variable(im_hr.cuda())
            output_SR = netG(im_lr)

            # --------------------
            # Train Generator
            # --------------------

            loss_G = criterion_G(netD(output_SR), torch.ones_like(netD(output_SR)))
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # --------------------
            # Train Discriminator
            # --------------------

            real_loss_D = criterion_D(netD(im_hr), torch.ones_like(netD(im_hr)))
            fake_loss_D = criterion_D(netD(output_SR), torch.zeros_like(netD(output_SR)))
            loss_D = (real_loss_D + fake_loss_D) / 2
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

        end = time.time()
        epoch_time = (end - start)
        total_time = total_time + epoch_time

        if (epoch + 1) % args.period == 0:
            log = "[{} / {}] \tLearning_rate: {:.5f}\t Generator Loss: {:.4f} \t Discriminator Loss: {:.4f} \t Time: {:.4f}".format(epoch + 1, args.epochs, learning_rate, loss_G, loss_D, total_time)
            print(log)
            save.save_log(log)
            save.save_model(my_model, epoch)
            total_time = 0

    netG.eval()
    test(args, netG, testdataloader)
    netG.train()

    writer.close()

if __name__ == '__main__':
    train(args)