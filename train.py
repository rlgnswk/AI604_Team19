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
writer = SummaryWriter()

import models
import data_utils 
import utils 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
parser.add_argument('--gpu', type=int)
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--datapath', type=str)

args = parser.parse_args()

SR_outputPath= utils.file_generator(args.name)

trainDataLoader, validDataLoader , testDataLoader = data_utils.Load_Data(args.datapath)

netD = models.netD()
netG = models.netSR()

criterion = 
optimizer =

for epoch in range(args.num_epochs):
    for i, data in enumerate(trainDataLoader, 0):
        pass

writer.close()

