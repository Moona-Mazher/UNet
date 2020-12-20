# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 19:57:34 2020

@author: moona
"""

#                           Basic UNET Model



#import pytorch libraries
import torch
import torch.nn

#generate random input (batch size, channel, height, width)
inp=torch.rand(1,1,224,224)
inp.shape

# Step 1- ------------------------------

# 1st convolutional layer (input channel, output filter, kernal size, stride, padding)
m = torch.nn.Conv2d(1, 64, 3, stride=1, padding=1)
out=m(inp)
out.shape
print("C1 Out = ",out.shape)

# 2nd convolutional layer 
#m2= torch.nn.Conv2d(64, 64, 3, stride=1, padding=0)
m2= torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
out2=m2(out)
out2.shape
print("C2 Out = ",out2.shape)

# Step 2-------------------------------

# max pooling layer
mp1 = torch.nn.MaxPool2d(2,2)
outmp1=mp1(out2)
outmp1.shape
print("Maxpool1 Out = ",outmp1.shape)

# Conv3 Layer

m3= torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
out3=m3(outmp1)
out3.shape
print("C3 Out = ",out3.shape)

# Conv4

m4= torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)
out4=m4(out3)
out4.shape
print("C4 Out = ",out4.shape)

# Step 3-----------------------------

# max pooling layer
mp2 = torch.nn.MaxPool2d(2,2)
outmp2=mp2(out4)
outmp2.shape
print("Maxpool2  Out = ",outmp2.shape)

# Conv5

m5= torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)
out5=m5(outmp2)
out5.shape
print("C5 Out = ",out5.shape)

# Conv6

m6= torch.nn.Conv2d(256, 256, 3, stride=1, padding=1)
out6=m6(out5)
out6.shape
print("C6 Out = ",out6.shape)

# Step 4----------------------------

# max pooling layer
mp3 = torch.nn.MaxPool2d(2,2)
outmp3=mp3(out6)
outmp3.shape
print("Maxpool 3 Out = ",outmp3.shape)

# Conv7 Layer

m7= torch.nn.Conv2d(256, 512, 3, stride=1, padding=1)
out7=m7(outmp3)
out7.shape
print("C7 Out = ",out7.shape)

# Conv8

m8= torch.nn.Conv2d(512, 512, 3, stride=1, padding=1)
out8=m8(out7)
out8.shape
print("C8 Out = ",out8.shape)

# Base Layer----------------------

# max pooling layer
mp4 = torch.nn.MaxPool2d(2,2)
outmp4=mp4(out8)
outmp4.shape
print("Maxpooling 4 Out = ",outmp4.shape)

# Conv9 base

m9= torch.nn.Conv2d(512, 1024, 3, stride=1, padding=1)
out9=m9(outmp4)
out9.shape
print("C9 Out = ",out9.shape)

# Conv10 base

m10= torch.nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
out10=m10(out9)
out10.shape
print("C10 Out = ",out10.shape)

# Decoder Side
# Step 1 Decoder --------------------------------------


#upsampling 1
up1=torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
outup1=up1(out10)
outup1.shape
print("Upsample 1 Out = ",outup1.shape)

# Concatenation 1
import torch.nn.functional as F
def forward(x1,x2):
  # input is CHW
  diffY = x2.size()[2] - x1.size()[2]
  diffX = x2.size()[3] - x1.size()[3]
  x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
  x = torch.cat([x2, x1], dim=1)
  return x

CF1=forward(out8,outup1)   
print("Concatenation 1 = ",CF1.shape)   

# 1D Conv layer to adjust the feature size i.e. 1536 to 1024

d1c= torch.nn.Conv2d(1536, 1024, kernel_size=1)
conv1=d1c(CF1)
print("1DC Out = ", conv1.shape)

# Deconv1
d1= torch.nn.Conv2d(1024, 512, 3, stride=1, padding=1)
Deconv1=d1(conv1)
Deconv1.shape
print("DC1 Out = ",Deconv1.shape)

# Deconv2
d2= torch.nn.Conv2d(512, 512, 3, stride=1, padding=1)
Deconv2=d2(Deconv1)
Deconv2.shape
print("DC2 Out = ",Deconv2.shape)

# Step 2 Decoder--------------------------

# upsampling 2

up2=torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
outup2=up1(Deconv2)
outup2.shape
print("Upsample 2 Out = ",outup2.shape)

# Concatenation 2

import torch.nn.functional as F
def forward(x1,x2):
  # input is CHW
  diffY = x2.size()[2] - x1.size()[2]
  diffX = x2.size()[3] - x1.size()[3]
  x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
  x = torch.cat([x2, x1], dim=1)
  return x
#print(output2.shape)
CF2=forward(out6,outup2)   
print("Concatenation 2 = ", CF2.shape)   

# 1D Conv layer to adjust the feature size i.e. 768 to 512

d2c= torch.nn.Conv2d(768, 512, kernel_size=1)
conv2=d2c(CF2)
print("1DC Out = ", conv2.shape)

# Deconv3
d3= torch.nn.Conv2d(512, 256, 3, stride=1, padding=1)
Deconv3=d3(conv2)
Deconv3.shape
print("DC3 Out = ",Deconv3.shape)

# Deconv4
d4= torch.nn.Conv2d(256, 256, 3, stride=1, padding=1)
Deconv4=d4(Deconv3)
Deconv4.shape
print("DC4 Out = ",Deconv4.shape)

# Step 3 Docoder-------------------------------

# upsampling 3

up3=torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
outup3=up3(Deconv4)
outup3.shape
print("Upsample 3 Out = ",outup3.shape)

# Concatenation 3

import torch.nn.functional as F
def forward(x1,x2):
  # input is CHW
  diffY = x2.size()[2] - x1.size()[2]
  diffX = x2.size()[3] - x1.size()[3]
  x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
  x = torch.cat([x2, x1], dim=1)
  return x

CF3=forward(out4,outup3)   
print("Concatenation 3 = ", CF3.shape)   

# 1D Conv layer to adjust the feature size i.e. 384 to 256

d3c= torch.nn.Conv2d(384, 256, kernel_size=1)
conv3=d3c(CF3)
print("1DC out = ", conv3.shape)

# Deconv5

d5= torch.nn.Conv2d(256, 128, 3, stride=1, padding=1)
Deconv5=d5(conv3)
Deconv5.shape
print("DC6 Out = ",Deconv5.shape)

# Deconv6

d6= torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)
Deconv6=d6(Deconv5)
Deconv6.shape
print("DC6 Out = ",Deconv6.shape)

# Step 4 Docoder-------------------------------

# upsampling 4

up4=torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
outup4=up4(Deconv6)
outup4.shape
print("Upsample 4 Out = ",outup4.shape)

# Concatenation 4

import torch.nn.functional as F
def forward(x1,x2):
  # input is CHW
  diffY = x2.size()[2] - x1.size()[2]
  diffX = x2.size()[3] - x1.size()[3]
  x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2])
  x = torch.cat([x2, x1], dim=1)
  return x

CF4=forward(out2,outup4)   
print("Concatenation 4 = ", CF4.shape)   

# 1D Conv layer to adjust the feature size i.e. 192 to 128

d4c= torch.nn.Conv2d(192, 128, kernel_size=1)
conv4=d4c(CF4)
print("1DC Out = ", conv4.shape)

# Deconv7

d7= torch.nn.Conv2d(128, 64, 3, stride=1, padding=1)
Deconv7=d7(conv4)
Deconv7.shape
print("DC8 Out = ",Deconv7.shape)

# Deconv8

d8= torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
Deconv8=d8(Deconv7)
Deconv8.shape
print("DC8 Out = ",Deconv8.shape)

# Last layer of Decoder 1D Conv layer 
d_out= torch.nn.Conv2d(64, 2, kernel_size=1)
Deconv_out=d_out(Deconv8)
print("DC Last Out = ", Deconv_out.shape)

