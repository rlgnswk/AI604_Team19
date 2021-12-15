import torch
import torch.nn as nn  #
import torch.nn.functional as F  # various activation functions for model
from torchvision import models

class Conv_block4NetD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_block4NetD, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, padding=1, stride=2,
                              bias=False)
        self.conv_bn = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_bn(x)
        x = self.leaky_relu(x)
        return x


# input size 128×128
class netD(nn.Module):
    def __init__(self, input_channel=3, mid_channel=64):
        self.input_channel = input_channel
        self.mid_channel = mid_channel
        super(netD, self).__init__()

        # self.Conv_blockIn = Conv_block4NetD(self.input_channel , self.mid_channel//2)     # output size = self.mid_channel / 2 x 64 x 64
        self.Conv_block1 = Conv_block4NetD(self.input_channel,
                                           self.mid_channel)  # output size = self.mid_channel     x 32 x 32
        self.Conv_block2 = Conv_block4NetD(self.mid_channel * 1,
                                           self.mid_channel * 2)  # output size = self.mid_channel * 2 x 16 x 16
        self.Conv_block3 = Conv_block4NetD(self.mid_channel * 2,
                                           self.mid_channel * 4)  # output size = self.mid_channel * 4 x 8 x 8
        self.Conv_block4 = Conv_block4NetD(self.mid_channel * 4,
                                           self.mid_channel * 8)  # output size = self.mid_channel * 8 x 4 x 4
        self.Conv_last = nn.Conv2d(self.mid_channel * 8, 1, 4, 1, 0, bias=False)
        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):
        # x = self.Conv_blockIn(x)
        x = self.Conv_block1(x)
        x = self.Conv_block2(x)
        x = self.Conv_block3(x)
        x = self.Conv_block4(x)
        x = self.Conv_last(x)
        x = self.sigmoid_layer(x)
        return x


class Conv_block4NetSR_ds(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Conv_block4NetSR_ds, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride)
        # self.conv_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        # x = self.conv_bn(x)
        x = F.relu(x)
        return x


class Conv_block4NetSR_us(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Conv_block4NetSR_us, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=1)
        # self.conv_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        # x = self.conv_bn(x)
        x = F.relu(x)
        return x




class netSR(nn.Module):
    def __init__(self, input_channel=3, mid_channel=64):
        self.input_channel = input_channel
        self.mid_channel = mid_channel
        super(netSR, self).__init__()

        self.Conv_blockIn = Conv_block4NetSR_ds(self.input_channel, self.mid_channel,1)
        self.Conv_block1 = Conv_block4NetSR_ds(self.mid_channel, self.mid_channel,2)
        self.Conv_block2 = Conv_block4NetSR_ds(self.mid_channel, self.mid_channel,2)
        self.Conv_block3 = Conv_block4NetSR_ds(self.mid_channel, self.mid_channel,1)
        self.Conv_block4 = Conv_block4NetSR_ds(self.mid_channel, self.mid_channel,1)
        self.Conv_block5 = Conv_block4NetSR_us(self.mid_channel, self.mid_channel,2)
        self.Conv_block6 = Conv_block4NetSR_us(self.mid_channel, self.mid_channel,2)

        self.ConvOut = nn.Conv2d(in_channels=self.mid_channel, out_channels=self.input_channel, kernel_size=3,
                                 padding=1, stride=1)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_In = x
        x1 = self.Conv_blockIn(x) #64*64
        x2 = self.Conv_block1(x1) #32*32
        x3 = self.Conv_block2(x2) #16*16
        x4 = self.Conv_block3(x3) #16*16
        x5 = self.Conv_block4(x4) #16*16
        x5=x5+x3
        x6 = self.Conv_block5(x5)#32*32
        x6=x6+x2
        x7 = self.Conv_block6(x6) #64*64
        x7=x7+x1
        x8 = self.ConvOut(x7)
        return x8
class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.relu3_1 = nn.Sequential()
        for x in range(12):
            self.relu3_1.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        out = self.relu3_1(x)
        return out

if __name__ == '__main__':
    print("check forwarding model")

    netD = netD(input_channel=3, mid_channel=64)
    netSR = netSR(input_channel=3, mid_channel=64)

    temp_input = torch.randn(1, 3, 128, 128)

    netSR_output = netSR(temp_input)
    print("netSR_output: ", netSR_output.shape)
    netD_output = netD(temp_input)
    print("netD_output: ", netD_output.shape)
