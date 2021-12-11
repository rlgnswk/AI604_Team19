import torch
from torch.nn import init
from torch.autograd import Variable

import argparse
import math
from math import log10
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import models
from data import *
from lpips import lpips
from utils import *
from metric import *

parser = argparse.ArgumentParser(description='ZSSR-GAN')

# validation data
parser.add_argument('--name', default='test', help='save result')
parser.add_argument('--data_dir', type=str, default='./MyDataset_AI604')
parser.add_argument('--dataset', type=str, default='MySet5x2')
parser.add_argument('--GT_path', type=str, default='HR')
parser.add_argument('--LR_path', type=str, default='g20_non_ideal_LR')

parser.add_argument('--valBatchSize', type=int, default=1)
parser.add_argument('--pretrained_model', default='./results/NetFinal/model/model_latest.pt', help='save result')
parser.add_argument('--SR_ratio', type=int, default=2, help='SR ratio')
parser.add_argument('--nFeat', type=int, default=64, help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=3, help='number of color channels to use')
parser.add_argument('--patchSize', type=int, default=64, help='patch size')
parser.add_argument('--input_channel', type=int, default=3, help='netSR and netD input channel')
parser.add_argument('--mid_channel', type=int, default=64, help='netSR middle channel')
parser.add_argument('--nThreads', type=int, default=8, help='number of threads for data loading')
parser.add_argument('--saveDir', default='./results', help='datasave directory')
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
    gt_path = os.path.join(args.data_dir, args.dataset, args.GT_path)
    sr_path = './results/test_output'

    gt_filelist = sorted([os.path.join(gt_path, img) for img in os.listdir(gt_path)])
    sr_filelist = sorted([os.path.join(sr_path, img) for img in os.listdir(sr_path)])

    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    count = 0

    for gt_file, sr_file in zip(gt_filelist, sr_filelist):
        count += 1

        gt = cv2.imread(gt_file)
        sr = cv2.imread(sr_file)

        ssim_val = ssim(gt, gt, multichannel=True)

        gt = RGB_np2Tensor(gt)
        sr = RGB_np2Tensor(sr)
        lpips_val = lpips(sr, gt, net_type='vgg').item()
        psnr_val = get_psnr(sr, gt)  

        avg_psnr += psnr_val
        avg_ssim += ssim_val
        avg_lpips += lpips_val

        print('Image {} \t PSNR: {:.4f} \t SSIM: {:.4f} \t LPIPS: {:.4f}'.format(count, psnr_val, ssim_val, lpips_val))

    print()
    print('Average PSNR: {:.4f}'.format(avg_psnr/count))
    print('Average SSIM: {:.4f}'.format(avg_ssim/count))
    print('Average LPIPS: {:.4f}'.format(avg_lpips/count))

if __name__ == '__main__':
    test(args)