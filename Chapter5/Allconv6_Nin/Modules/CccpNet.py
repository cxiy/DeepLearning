#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @file name  : CccpNet.py
# @author     : cenzy
# @date       : 2021-05-13
# @brief      : cccp网络
"""
import torch
import torch.nn as nn

class CccpNet(nn.Module):
    def __init__(self ,cccp_count = 0):
        '''
        cccp_count : cccp个数，0或默认没有cccp

        '''
        super(CccpNet,self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, 96, 11, 4))
        layers.append(nn.ReLU())
        
        if cccp_count >= 1:
            layers.append(nn.Conv2d(96, 96, 1, 1))
            layers.append(nn.ReLU())
        if cccp_count >= 2:
            layers.append(nn.Conv2d(96, 96, 1, 1))
            layers.append(nn.ReLU())  
            
        layers.append(nn.MaxPool2d(3, 2,ceil_mode=True))
        layers.append(nn.Conv2d(96, 256, 5, 1, padding = 2))
        layers.append(nn.ReLU())
        
        if cccp_count >= 3:
            layers.append(nn.Conv2d(256, 256, 1, 1))
            layers.append(nn.ReLU())  
        if cccp_count >= 4:
            layers.append(nn.Conv2d(256, 256, 1, 1))
            layers.append(nn.ReLU())  
            
        layers.append(nn.MaxPool2d(3, 2))
        layers.append(nn.Conv2d(256, 384, 3, 1, padding = 1))
        layers.append(nn.ReLU())
        
        if cccp_count >= 5:
            layers.append(nn.Conv2d(384, 384, 1, 1))
            layers.append(nn.ReLU())  
        if cccp_count >= 6:
            layers.append(nn.Conv2d(384, 384, 1, 1))
            layers.append(nn.ReLU())  
            
        layers.append(nn.MaxPool2d(3, 2))
        layers.append(nn.Dropout(p = 0.5))
        layers.append(nn.Conv2d(384, 1024, 3, 1, padding = 1))
        layers.append(nn.ReLU())
        
        if cccp_count >= 7:
            layers.append(nn.Conv2d(1024, 1024, 1, 1))
            layers.append(nn.ReLU())  
            
        layers.append(nn.Conv2d(1024, 20, 1, 1))
        layers.append(nn.ReLU())
        layers.append(nn.AvgPool2d(6,1))
        
        self.layers = nn.Sequential(*layers)
        
        
    def forward(self, x):
        x = self.layers(x)
        out = x.view(-1,20)
        return out


if __name__ == "__main__":
    
    from torch.autograd import Variable
    
    print("*"*32)
    print("cccp 1")
    net = CccpNet(cccp_count=3)
    print(net)
    inputs = Variable(torch.randn(1,3,224,224))
    output = net(inputs)
    print(output.shape)
    print("*"*32)
    
    