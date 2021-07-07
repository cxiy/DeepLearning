#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @file name  : Allconv6.py
# @author     : cenzy
# @date       : 2021-05-13
# @brief      : Allconv6网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(bottleneck,self).__init__()
        mid_channels = int(in_channels/2)
        self.conv_1_bottleneck = nn.Conv2d(in_channels,mid_channels , 1, 1)
        self.conv_2_bottleneck = nn.Conv2d(mid_channels, mid_channels, 3, 2, padding=1)
        self.conv_3_bottleneck = nn.Conv2d(mid_channels, out_channels, 1)
        
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.pool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        
    def forward(self, x):
        out1 = F.relu(self.conv_1_bottleneck(x))
        out1 = F.relu(self.conv_2_bottleneck(out1))
        out1 = F.relu(self.conv_3_bottleneck(out1))
        
        out2 = F.relu(self.conv(x))
        out2 = self.pool(out2)
        
        out = torch.cat([out1,out2], 1)
        return out

class Allconv6(nn.Module):
    def __init__(self , num_classes, bottleneck_layer = None):
        '''
        num_classes :类别数量
        bottleneck_layer : 添加瓶颈结构的卷积层:conv2,conv3,conv4,conv5
        '''
        super(Allconv6,self).__init__()
        if bottleneck_layer == "conv2":
            self.layers = self.make_layers_conv2()
        elif bottleneck_layer == "conv3":
            self.layers = self.make_layers_conv3()
        elif bottleneck_layer == "conv4":
            self.layers = self.make_layers_conv4()
        elif bottleneck_layer == "conv5":
            self.layers = self.make_layers_conv5()
        else:
            self.layers = self.make_layers_baseline()       
        
        self.pool6 = nn.AvgPool2d(7,stride=1)
        self.fc = nn.Linear(256*1,num_classes)
        
        #init
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0)

     
    def make_layers_baseline(self):
         layers = nn.Sequential(
             nn.Conv2d(3, 16, 3, 2, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(16, 32, 3, 2, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(32, 64, 3, 2, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(64, 128, 3, 2, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(128, 256, 3, 2, padding=1),
             nn.ReLU(inplace=True)
             )
         return layers
     
    def make_layers_conv2(self):
         layers = nn.Sequential(
             nn.Conv2d(3, 16, 3, 2, padding=1),
             nn.ReLU(inplace=True),
             bottleneck(16,16),
             nn.Conv2d(32, 64, 3, 2, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(64, 128, 3, 2, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(128, 256, 3, 2, padding=1),
             nn.ReLU(inplace=True)
             )
         return layers
     
    def make_layers_conv3(self):
         layers = nn.Sequential(
             nn.Conv2d(3, 16, 3, 2, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(16, 32, 3, 2, padding=1),
             nn.ReLU(inplace=True),
             bottleneck(32,32),
             nn.Conv2d(64, 128, 3, 2, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(128, 256, 3, 2, padding=1),
             nn.ReLU(inplace=True)
             )
         return layers
     
    def make_layers_conv4(self):
         layers = nn.Sequential(
             nn.Conv2d(3, 16, 3, 2, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(16, 32, 3, 2, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(32, 64, 3, 2, padding=1),
             nn.ReLU(inplace=True),
             bottleneck(64,64),
             nn.Conv2d(128, 256, 3, 2, padding=1),
             nn.ReLU(inplace=True)
             )
         return layers
     
    def make_layers_conv5(self):
         layers = nn.Sequential(
             nn.Conv2d(3, 16, 3, 2, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(16, 32, 3, 2, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(32, 64, 3, 2, padding=1),
             nn.ReLU(inplace=True),
             nn.Conv2d(64, 128, 3, 2, padding=1),
             nn.ReLU(inplace=True),
             bottleneck(128,128),
             )
         return layers
     
    def forward(self , x):
        out = self.layers(x)
        out = self.pool6(out)
        out = out.view(-1 , 256*1) 
        out = self.fc(out)
        return out
    

if __name__ == "__main__":
    
    from torch.autograd import Variable
      
    # bottlenecknet = bottleneck(16,32)
    # inputs = Variable(torch.randn(1,16,224,224))
    # out = bottlenecknet(inputs)
    
    # from torchsummary import summary
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = bottleneck(16,16).to(device)

    # summary(model, (16, 112, 112))
    
    print("*"*32)
    print("allconv6 baseline")
    allconv6 = Allconv6(20)
    print(allconv6)
    inputs = Variable(torch.randn(1,3,224,224))
    output = allconv6(inputs)
    print(output.shape)
    print("*"*32)

    print("*"*32)
    print("allconv6 conv2")
    allconv6 = Allconv6(20,bottleneck_layer="conv2")
    print(allconv6)
    inputs = Variable(torch.randn(1,3,224,224))
    output = allconv6(inputs)
    print(output.shape)
    print("*"*32)
    
    print("*"*32)
    print("allconv6 conv3")
    allconv6 = Allconv6(20,bottleneck_layer="conv3")
    print(allconv6)
    inputs = Variable(torch.randn(1,3,224,224))
    output = allconv6(inputs)
    print(output.shape)
    print("*"*32)
    
    print("*"*32)
    print("allconv6 conv4")
    allconv6 = Allconv6(20,bottleneck_layer="conv4")
    print(allconv6)
    inputs = Variable(torch.randn(1,3,224,224))
    output = allconv6(inputs)
    print(output.shape)
    print("*"*32)
    
    print("*"*32)
    print("allconv6 conv5")
    allconv6 = Allconv6(20,bottleneck_layer="conv5")
    print(allconv6)
    inputs = Variable(torch.randn(1,3,224,224))
    output = allconv6(inputs)
    print(output.shape)
    print("*"*32)
    

