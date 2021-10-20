import torch 
import torch.nn as nn #
import torch.nn.functional as F # various activation functions for model

class netD(nn.Module):
    def __init__(self):
        super(netD, self).__init__() 

    def forward(self, x):
        return x


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels = in_channels,
                       out_channels = out_channels,
                       kernel_size = 3,
                       padding= 1,
                       stride = 1)
        self.conv_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_bn(x)
        x = F.relu(x)
        return x


class netSR(nn.Module):
    def __init__(self, input_channel):
        self.input_channel = input_channel
        
        super(netSR, self).__init__()
        
        self.Conv_block1 = Conv_block(self.input_channel, 32)

        #...

    def forward(self, x):

        return x