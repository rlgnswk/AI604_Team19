import models

from utils import *
from torchvision.utils import save_image
import argparse
import time
import math
from math import log10
from torch.nn import init
import numpy as np
from PIL import Image as PIL_image
from data import *
from test_metric import *
from skimage.metrics import structural_similarity as ssim

parser = argparse.ArgumentParser(description='ZSSR-GAN')

# validation data
parser.add_argument('--name', default='test', help='save result')
parser.add_argument('--GT_path', type=str, default='./datasets/GT')
parser.add_argument('--LR_path', type=str, default='./datasets/LR')
parser.add_argument('--datasetPath', type=str, default='./datasets')
parser.add_argument('--datasettype', type=str, default='Set5')
parser.add_argument('--kerneltype', default='g02', help='save result')
parser.add_argument('--valBatchSize', type=int, default=1)
parser.add_argument('--pretrained_model', default='./results/test/baby/model/model0_latest.pt', help='save result')
parser.add_argument('--nDenseBlock', type=int, default=3, help='number of DenseBlock')
parser.add_argument('--nRRDB', type=int, default=3, help='number of RRDB')
parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=3, help='number of color channels to use')
parser.add_argument('--patchSize', type=int, default=512, help='patch size') # entire size for test
parser.add_argument('--input_channel', type=int, default=3, help='netSR and netD input channel')
parser.add_argument('--mid_channel', type=int, default=64, help='netSR middle channel')
parser.add_argument('--nThreads', type=int, default=8, help='number of threads for data loading')
parser.add_argument('--saveDir', default='./results', help='datasave directory')
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


def test(args):
    my_model = models.netSR(input_channel=args.input_channel, mid_channel=args.mid_channel)
    my_model.apply(weights_init)
    my_model.cuda()

    my_model.load_state_dict(torch.load(args.pretrained_model))
    my_model.eval()
    gt_path = sorted(os.listdir(args.GT_path))
    length = len(gt_path)
    gt_pi = PIL_image.open(args.GT_path + "/" + gt_path[0]).convert('RGB')
    lq_pi = PIL_image.open(args.LR_path + "/" + gt_path[0]).convert('RGB')

    gt = RGB_np2Tensor(np.array(gt_pi)).cuda()
    lq = RGB_np2Tensor(np.array(lq_pi)).cuda()
    gt = torch.unsqueeze(gt, dim=0)
    lq = torch.unsqueeze(lq, dim=0)
    save = saveData(args, gt_path[0][:-4])
    avg_psnr = 0
    avg_ssim = 0
    count = 0
    with torch.no_grad():
        input_img = F.interpolate(lq, scale_factor=2)
        #save_image(input_img, save.save_dir_image + "/" +  str(iters) + '.png')
        output= my_model(input_img)
        save_image(output, save.save_dir_image + "/out"  + '.png')




if __name__ == '__main__':
    test(args)
