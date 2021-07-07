#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @file name  : Allconv5.py
# @author     : cenzy
# @date       : 2021-05-12
# @brief      : Allconv5网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Allconv5(nn.Module):
    def __init__(self , num_classes):
        super(Allconv5,self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 4, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 2, padding=1)
        self.pool5 = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(128*1, num_classes)
        
        #init
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0)

                
    def forward(self , x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.pool5(out)
        out = out.view(-1 , 128*1) 
        out = self.fc(out)
        return out
    
    
if __name__ == "__main__":
    
    from torch.autograd import Variable
    
    allconv6 = Allconv5(20)
    print(allconv6)
    inputs = Variable(torch.randn(1,3,224,224))
    output = allconv6(inputs)
    print(output.shape)