import torch
import torch.nn as nn
from torchvision import models

from typing import Sequence
from itertools import chain

from .utils import normalize_activation

def get_network(net_type):
	if net_type == 'alex':
		return AlexNet()
	elif net_type == 'squeeze':
		return SqueezeNet()
	elif net_type == 'vgg':
		return VGG16()

class LinLayers(nn.ModuleList):
	def __init__(self, n_channel_list):
		super(LinLayers, self).__init__([
			nn.Sequential(
				nn.Identity(),
				nn.Conv2d(nc, 1, 1, 1, 0, bias=False)
			) for nc in n_channel_list
		])

		for param in self.parameters():
			param.requires_grad = False

class BaseNet(nn.Module):
	def __init__(self):
		super(BaseNet, self).__init__()

		self.register_buffer('mean', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
		self.register_buffer('std', torch.Tensor([.458, .448, .450])[None, :, None, None])

	def set_requires_grad(self, state):
		for param in chain(self.parameters(), self.buffers()):
			param.requires_grad = state

	def z_score(self, x):
		return (x - self.mean) / self.std

	def forward(self, x):
		x = self.z_score(x)

		output = []
		for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
			x = layer(x)
			if i in self.target_layers:
				output.append(normalize_activation(x))
			if len(output) == len(self.target_layers):
				break

		return output

class SqueezeNet(BaseNet):
	def __init__(self):
		super(SqueezeNet, self).__init__()
		
		self.layers = models.squeezenet1_1(True).features
		self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]
		self.target_layers = [2, 5, 8, 10, 11, 12, 13]
		self.set_requires_grad(False)

class AlexNet(BaseNet):
	def __init__(self):
		super(AlexNet, self).__init__()

		self.layers = models.alexnet(True).features
		self.target_layers = [2, 5, 8, 10, 12]
		self.n_channels_list = [64, 192, 384, 256, 256]

		self.set_requires_grad(False)

class VGG16(BaseNet):
	def __init__(self):
		super(VGG16, self).__init__()

		self.layers = models.vgg16(True).features
		self.target_layers = [4, 9, 16, 23, 30]
		self.n_channels_list = [64, 128, 256, 512, 512]

		self.set_requires_grad(False)