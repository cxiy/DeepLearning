# -*- coding: utf-8 -*-
"""
# @file name  : MobileNet.py
# @author     : cenzy
# @date       : 2021-05-12
# @brief      : mobilenet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, 
                               padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, 
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out

class MobileNet(nn.Module):
    def __init__(self,class_num, mobile_count=10):
        '''
        class_num : 分类数量
        mobile_count : MobileNet网络深度:10,12,14,16,20,24,26
        '''
        super(MobileNet,self).__init__()
        self.mobile_count = mobile_count
        self.conv1 = nn.Conv2d(3, 32, 3, 2, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(32)   
        self.layers,self.linear_channels = self._maker_layer(32)          
        self.pool4 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.linear_channels, 20)
        
    #组合Bolck模块
    def _maker_layer(self, in_channels):
        #outchannels_stride列表:Block 模块的(输出通道数，stride）,输入通到数取上一层输出通道数
        if self.mobile_count == 10:
            outchannels_stride = [(4,2),(128,2),(128,2),(256,2)] 
        elif self.mobile_count == 12:
            outchannels_stride = [(64,1),(128,2),(128,2),(256,2),(256,2)]
        elif self.mobile_count == 14:
            outchannels_stride = [(64,1),(128,2),(128,1),(256,2),(256,2),(512,2)]      
        elif self.mobile_count == 16:
            outchannels_stride = [(64,1),(128,2),(128,1),(256,2),(256,1),(512,2),(512,2)]      
        elif self.mobile_count == 20:
            outchannels_stride = [(64,1),(128,2),(128,1),(256,2),(256,1),(512,2),(512,1),
                              (512,1),(512,2)]
        elif self.mobile_count == 24:
            outchannels_stride = [(64,1),(128,2),(128,1),(256,2),(256,1),(512,2), (512,1),(512,1),
                          (512,1),(512,1),(512,2)]   
        else: #self.mobile_count == 26:
            outchannels_stride = [(64,1),(128,2),(128,1),(256,2),(256,1),(512,2),(512,1),(512,1),
                          (512,1),(512,1),(512,1),(1024,2)]  
        layers = []
        in_planes = in_channels
        for i,cfg in enumerate(outchannels_stride):
            if i > 0:
                in_planes = outchannels_stride[i-1][0]
            layers.append(Block(in_planes,cfg[0],cfg[1])) #cfg[0] 上一层的输出是该层的输入
        return nn.Sequential(*layers),outchannels_stride[len(outchannels_stride)-1][0]
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.pool4(out)
        out = out.view(-1,self.linear_channels)
        out = self.fc(out)

        return out
    
if __name__ == "__main__":
    
    from torch.autograd import Variable
          
    net = MobileNet(20,mobile_count=26)
    print(net)
    inputs = Variable(torch.randn(1,3,224,224))
    output = net(inputs)
    print(net.linear_channels)
    print(output.shape)

