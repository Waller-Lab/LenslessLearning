import torch
import torch.nn as nn
import torch.nn.functional as F

BN_EPS = 1e-4


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class StackEncoder(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=(3, 3)):
        super(StackEncoder, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x):
        x = self.encode(x)
        x_small = F.max_pool2d(x, kernel_size=2, stride=2)
        return x, x_small


class StackDecoder(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(StackDecoder, self).__init__()
        padding = (kernel_size - 1) // 2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x, down_tensor):
        _, channels, height, width = down_tensor.size()
        x = F.upsample(x, size=(height, width), mode='bilinear')
        x = torch.cat([x, down_tensor], 1)
        x = self.decode(x)
        return x

    # 32x32
    
class UNet270480(nn.Module):
    def __init__(self, in_shape):
        super(UNet270480, self).__init__()
        channels, height, width = in_shape

        self.down1 = StackEncoder(3, 24, kernel_size=3) ;# 256
        self.down2 = StackEncoder(24, 64, kernel_size=3)  # 128
        self.down3 = StackEncoder(64, 128, kernel_size=3)  # 64
        self.down4 = StackEncoder(128, 256, kernel_size=3)  # 32
        self.down5 = StackEncoder(256, 512, kernel_size=3)  # 16
        

        self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 128
        self.up2 = StackDecoder(64, 64, 24, kernel_size=3)  # 256
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)  # 512
        self.classify = nn.Conv2d(24, 3, kernel_size=1, bias=True)


        self.center = nn.Sequential(ConvBnRelu2d(512, 512, kernel_size=3, padding=1))


    def forward(self, x):
        out = x; 
        down1, out = self.down1(out); 
        down2, out = self.down2(out); 
        down3, out = self.down3(out); 
        down4, out = self.down4(out); 
        down5, out = self.down5(out); 

        out = self.center(out)
        out = self.up5(out, down5); 
        out = self.up4(out, down4); 
        out = self.up3(out, down3); 
        out = self.up2(out, down2); 
        out = self.up1(out, down1); 

        out = self.classify(out); 
        out = torch.squeeze(out, dim=1); 
        return out



    
class UNet_small(nn.Module):
    def __init__(self, in_shape):
        super(UNet_small, self).__init__()
        channels, height, width = in_shape

        self.down1 = StackEncoder(3, 24, kernel_size=3)  # 512

        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)  # 512
        self.classify = nn.Conv2d(24, 3, kernel_size=1, bias=True)

        self.center = nn.Sequential(
            ConvBnRelu2d(24, 24, kernel_size=3, padding=1),
        )


    def forward(self, x):
        out = x
        down1, out = self.down1(out)
        out = self.center(out)
        out = self.up1(out, down1)
        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out
    