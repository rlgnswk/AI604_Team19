import torch

from .modules.lpips import LPIPS

def lpips(x, y, net_type, version='0.1'):
	criterion = LPIPS(net_type, version)
	if torch.cuda.is_available():
		criterion.cuda()
	return criterion(x, y)