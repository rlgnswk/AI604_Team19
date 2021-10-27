import torch.nn.functional as F
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

from skimage.metrics import structural_similarity as ssim

parser = argparse.ArgumentParser(description='Super Resolution')

# validation data

parser.add_argument('--datasetPath', type=str, default='./datasets')
parser.add_argument('--datasettype', type=str, default='Set5')
parser.add_argument('--kerneltype', default='g02', help='save result')

parser.add_argument('--valDataroot', required=False, default='Dataet_VFI_HW2/ucf101_HW3') # modifying to your SR_data folder path
parser.add_argument('--valBatchSize', type=int, default=1)
parser.add_argument('--pretrained_model', default='./result/NetFinal/model/model_latest.pt', help='save result')
parser.add_argument('--nDenseBlock', type=int, default=3, help='number of DenseBlock')
parser.add_argument('--nRRDB', type=int, default=3, help='number of RRDB')
parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=3, help='number of color channels to use')
parser.add_argument('--patchSize', type=int, default=512, help='patch size') # entire size for test
parser.add_argument('--input_channel', type=int, default=3, help='netSR and netD input channel')
parser.add_argument('--mid_channel', type=int, default=64, help='netSR middle channel')
parser.add_argument('--nThreads', type=int, default=8, help='number of threads for data loading')

parser.add_argument('--scale', type=float, default=2, help='scale output size /input size')
parser.add_argument('--gpu', type=int, default=4, help='gpu index')

args = parser.parse_args()

if args.gpu == 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif args.gpu == 4:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)


def get_dataset(args):
    data_train=Dataset(args)
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=1,
                                             drop_last=True, shuffle=True, num_workers=int(args.nThreads), pin_memory=False)
    return dataloader

def get_testdataset(args):
    data_test=testDataset(args)
    dataloader = torch.utils.data.DataLoader(data_test, batch_size=1,
                                             drop_last=True, shuffle=True, num_workers=int(args.nThreads), pin_memory=False)
    return dataloader


def test(args):
    # SR network
    my_model = models.netSR(input_channel = args.input_channel, mid_channel = args.mid_channel)
    my_model.apply(weights_init)
    my_model.cuda()

    my_model.load_state_dict(torch.load(args.pretrained_model))

    testdataloader = get_testdataset(args)
    my_model.eval()

    avg_psnr = 0
    avg_ssim = 0
    count = 0
    for batch, im_1 in enumerate(testdataloader):
        count = count + 1
        with torch.no_grad():
            im_1 = Variable(im_1.cuda(), volatile=False)
            im_1 = F.interpolate(im_1, scale_factor=2, mode='bicubic')
            output =my_model(im_1)

        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]




        for t, m, s in zip(output, mean, std):
            t.mul_(s).add_(m)

        output = output.cpu()
        output = output.data.squeeze(0)

        output = output.numpy()
        output *= 255.0
        output = output.clip(0, 255)

        sp_0, sp_1, sp_2 = output.shape
        output_rgb = np.zeros((sp_1,sp_2,sp_0))
        output_rgb[:, :, 0] = output[2]
        output_rgb[:, :, 1] = output[1]
        output_rgb[:, :, 2] = output[0]
        out = Image.fromarray(np.uint8(output_rgb), mode='RGB')  # output of SRCNN

        out.save('result/NetFinal/FI_img_%03d.png' % (count))


        # =========== Target Image ===============




if __name__ == '__main__':
    test(args)
