import torch
import cv2
import os

from lpips import lpips
from metric import get_psnr, get_ssim

def RGB_np2Tensor(img_input):
	# to tensor
	ts = (2, 0, 1)
	img_input = torch.Tensor(img_input.transpose(ts).astype(float)).mul_(1.0)

	# normalization
	img_input = (img_input / 255.0 - 0.5) * 2

	return img_input

def calculate_metirc(srpath ,gtpath):
    # load SR image list and sort
    SRpath = srpath
    SRfile_list = os.listdir(SRpath)
    # load GT image list and sort
    GTpath = gtpath
    GTfile_list = os.listdir(GTpath)

    PSNR_score = 0.0
    SSIM_score = 0.0
    LPIPS_score = 0.0

    #load each image and calculate loss

    print("SRfile_list -- ", SRfile_list[:5])
    print("GTfile_list -- ", GTfile_list[:5])

    
    # PSNR SSIM LPIPS
    for idx in range(len(SRfile_list)):
        if len(GTfile_list) != len(SRfile_list):
            print("the number of image is mismatched!")
            break
        #print(SRfile_list[idx])
        sr_image = cv2.imread(SRpath +"\\"+  SRfile_list[idx])
        gt_image = cv2.imread(GTpath +"\\"+  GTfile_list[idx])
        sr_image = RGB_np2Tensor(sr_image).cuda()
        gt_image = RGB_np2Tensor(gt_image).cuda()
        try:
            PSNR_score = PSNR_score + get_psnr(sr_image, gt_image)
        except ValueError:
            print("there is mismatch size- idx  :  ", idx)
            pass
        try:
            SSIM_score = SSIM_score + get_ssim(sr_image, gt_image)
        except ValueError:
            print("there is mismatch size- idx  :  ", idx)
            pass
        try:        
            LPIPS_score = LPIPS_score + lpips(sr_image, gt_image, net_type='vgg', version='0.1')
        except RuntimeError:
            print("there is mismatch size- idx  :  ", idx)
            pass
            
    #save and mean

    final_PSNR_score = PSNR_score / len(SRfile_list)
    final_SSIM_score = SSIM_score / len(SRfile_list)
    final_LPIPS_score = LPIPS_score / len(SRfile_list)

    #print out
    print("final_PSNR_score:  ", final_PSNR_score)
    print("final_SSIM_score:  ", final_SSIM_score)
    print("final_LPIPS_score:  ", final_LPIPS_score)

if __name__ == '__main__':
    gtpath = None
	srpath = None
	calculate_metirc(srpath ,gtpath)
