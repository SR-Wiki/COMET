import torch
import torch.nn as nn
import numpy as np
#import tifffile
import os
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import skimage
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from pathlib import Path

class DoubleConv(nn.Module):
    """
    Convolution 3 x 3
    => 
    [BN] 
    => 
    Leaky ReLU
    Convolution 3 x 3 
    => 
    [BN] 
    => 
    Leaky ReLU
    """
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope = 0.02, inplace = True),
            nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope = 0.02, inplace = True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),#AdaptiveAvgPool2d
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear = True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size = 2, stride = 2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim = 1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        
    def forward(self, x):
        return self.conv(x)

class AUnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear = True):
        super(AUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)             
        self.down4 = Down(512, 1024 // 2)

        self.up1 = Up(1024, 512 // 2, bilinear)
        self.up2 = Up(512, 256 // 2, bilinear)
        self.up3 = Up(256, 128 // 2, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)


def normalize_(input_img):
    img = np.array(input_img).astype('float32')
    img = img - np.min(img)
    img = img / np.max(img)
    return img

def predict_(model, img_path, save_path, fname,ifGPU=True, reg=1):
    device = torch.device("cuda" if torch.cuda.is_available() and ifGPU else "cpu")
    if isinstance(img_path, str):
        img_path = Path(img_path)
    image_data = imread(str(img_path / fname))
    if image_data.dtype != np.uint8:
        max_val = np.iinfo(image_data.dtype).max
        image_data = (image_data.astype(np.float32) * 255.0 / max_val).clip(0, 255).astype(np.uint8)
    try:
        t, x, y = image_data.shape  
        image_data = image_data/255
        image_data = image_data.astype('float32')
        test_pred_np = np.zeros((t, x, y), dtype=np.float32)
    except ValueError:
        t = 1
        x, y = image_data.shape
        image_data = image_data/255
        image_data = image_data.astype('float32')
        image_data = np.expand_dims(image_data, axis=0)  # 添加帧维度
        test_pred_np = np.zeros((t, x, y), dtype=np.float32)
    for frame in range(t):
        single_frame = image_data[frame, :, :]
        #single_frame = normalize_(single_frame)
        single_frame = single_frame.astype('float32')
        single_frame = single_frame.reshape(1,x,y)
        #datatensor = torch.from_numpy(single_frame).unsqueeze(0).unsqueeze(0).to(device)
        imsize = (1,1,x,y)
        imagA = np.zeros(imsize)
        imagA[0,:,:,:] = single_frame
        imagA = torch.from_numpy(imagA)
        imagA = imagA.to(device, dtype = torch.float32)
        with torch.no_grad():
            test_pred = model(imagA)
            #test_pred = test_pred.squeeze().cpu().numpy()
        test_pred = test_pred.to(torch.device("cpu"))
        test_pred = test_pred.numpy()
        #test_pred = normalize_(test_pred)
        #plt.imshow(test_pred[0,0,:,:])
        #plt.show()

        test_pred_np[frame, :, :] = test_pred
        if frame % 100 == 0:
            print('predict %d images'% frame)
    #test_pred_np = normalize_(test_pred_np)

    os.makedirs(save_path, exist_ok=True)
    imsave(os.path.join(save_path, f'SN2N_{fname}_reg_{reg:.3f}.tif'), test_pred_np)



