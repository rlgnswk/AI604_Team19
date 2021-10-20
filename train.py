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
import utils 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
parser.add_argument('--gpu', type=int)
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--datasetPath', type=str)
## add if you need. more option
args = parser.parse_args()

################################# for yujin's part start
#trainDataLoader, validDataLoader , testDataLoader = data_utils.Load_Data(args.datapath)
trainDataLoader = data_utils.Load_Data(args.datasetPath)
################################# for yujin's part end

################################# for gihoon's part start
netD = models.netD()
netG = models.netSR()

criterion_G = 
optimizer_G = 

criterion_D = 
optimizer_D = 
##################################for gihoon's part end

writer = SummaryWriter()
SR_outputPath= utils.file_generator(args.name)

for epoch in range(args.num_epochs):
    for i, data in enumerate(trainDataLoader, 0):

        '''
        training the model by GAN trainng, 
        do test, 
        write the log and tensorboard
        '''

        pass

writer.close()

