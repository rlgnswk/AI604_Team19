import torch
import cv2
from lpips import lpips

def RGB_np2Tensor(img_input):
	# to tensor
	ts = (2, 0, 1)
	img_input = torch.Tensor(img_input.transpose(ts).astype(float)).mul_(1.0)

	# normalization
	img_input = (img_input / 255.0 - 0.5) * 2

	return img_input

x = cv2.imread('lpips_test_images/baby.png')
y = cv2.imread('lpips_test_images/SR_img_baby.png')

x = RGB_np2Tensor(x)
y = RGB_np2Tensor(y)

# Identical Images
loss_alex = lpips(x, x, net_type='alex', version='0.1')
loss_squeeze = lpips(x, x, net_type='squeeze', version='0.1')
loss_vgg = lpips(x, x, net_type='vgg', version='0.1')
print('IDENTICAL IMAGES\tloss_alex: {:.5f}\tloss_squeeze: {:.5f}\tloss_vgg: {:.5f}'.format(loss_alex.item(), loss_squeeze.item(), loss_vgg.item()))

# Unidentical Images
loss_alex = lpips(x, y, net_type='alex', version='0.1')
loss_squeeze = lpips(x, y, net_type='squeeze', version='0.1')
loss_vgg = lpips(x, y, net_type='vgg', version='0.1')
print('UNIDENTICAL IMAGES\tloss_alex: {:.5f}\tloss_squeeze: {:.5f}\tloss_vgg: {:.5f}'.format(loss_alex.item(), loss_squeeze.item(), loss_vgg.item()))