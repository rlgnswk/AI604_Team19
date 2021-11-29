import numpy as np
import math
from math import log10
from skimage.metrics import structural_similarity as ssim
import torch

def to_numpy_array(image):
    image = image.cpu()
    image = image.data.squeeze(0)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    image = image.numpy()
    image *= 255.0

    return image.clip(0, 255)

def get_psnr(sr_image, ground_truth):
    #inputs should be torch tensors

    # denormalize and convert to numpy array
    sr_image = to_numpy_array(sr_image)
    ground_truth = to_numpy_array(ground_truth)

    # psnr computation
    mse = ((ground_truth - sr_image) ** 2).mean()
    psnr_val = 10 * log10(255 * 255 / (mse + 10 ** (-10)))

    return psnr_val

def get_ssim(sr_image, ground_truth):
    # input should be torch tensors    

    # denormalize and convert to numpy array
    sr_image = to_numpy_array(sr_image)
    ground_truth = to_numpy_array(ground_truth)

    # ssim computation
    ssim_val = ssim(np.transpose(ground_truth, (1,2,0)),
                    np.transpose(sr_image, (1,2,0)),
                    multichannel=True)

    return ssim_val

# sr_image = torch.randn((1, 3, 32, 32))
# ground_truth = torch.randn((1, 3, 32, 32))

# print(get_psnr(sr_image, ground_truth))
# print(get_ssim(sr_image, ground_truth))