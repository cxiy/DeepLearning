#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @file name  : Allconv6.py
# @author     : cenzy
# @date       : 2021-05-10
# @brief      : Allconv6网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
 
class Allconv6(nn.Module):
    def __init__(self , num_classes, add_bn = False, add_dropout = False):
        '''
        num_classes : 分类数量
        add_bn : 是否添加BatchNorm2d层
        add_dropout : 是否添加dropout层

        '''
        super(Allconv6,self).__init__()
        self.add_bn = add_bn
        self.add_dropout = add_dropout
        if self.add_bn:
            self.layer1 = self.conv_bn(3, 64, 3, 2, padding=1)
            self.layer2 = self.conv_bn(64, 64, 3, 2, padding=1)
            self.layer3 = self.conv_bn(64, 128, 3, 2, padding=1)
            self.layer4 = self.conv_bn(128, 128, 3, 2, padding=1)
            self.layer5 = self.conv_bn(128, 256, 3, 2, padding=1)    
        else:               
            self.layer1 = nn.Conv2d(3, 64, 3, 2, padding=1)
            self.layer2 = nn.Conv2d(64, 64, 3, 2, padding=1)
            self.layer3 = nn.Conv2d(64, 128, 3, 2, padding=1)
            self.layer4 = nn.Conv2d(128, 128, 3, 2, padding=1)
            self.layer5 = nn.Conv2d(128, 256, 3, 2, padding=1)
        
        if self.add_dropout:
            self.dropout = nn.Dropout(p=0.5)    
        self.pool6 = nn.AvgPool2d(7,stride=1)
        self.fc = nn.Linear(256*1,num_classes)
      
        #init
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0)
             
    def conv_bn(self,in_channels, out_channels, kernel_size,stride, padding):
        layers = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
                    nn.BatchNorm2d(out_channels)
                    )
        return layers
         
    def forward(self , x):      
        x = F.relu(self.layer1(x))     
        x = F.relu(self.layer2(x))    
        x = F.relu(self.layer3(x))     
        x = F.relu(self.layer4(x))     
        x = F.relu(self.layer5(x))       
        if self.add_dropout:
            x = self.dropout(x)           
        x = self.pool6(x)
        x = x.view(-1 , 256*1) 
        out = self.fc(x)
        return out
    
    
if __name__ == "__main__":
    
    from torch.autograd import Variable
    
    allconv6 = Allconv6(20, add_bn=1,add_dropout = True)
    print(allconv6)
    inputs = Variable(torch.randn(1,3,224,224))
    output = allconv6(inputs)
    print(output.shape)