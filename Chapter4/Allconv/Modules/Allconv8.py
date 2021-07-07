#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# @file name  : Allconv8.py
# @author     : cenzy
# @date       : 2021-05-12
# @brief      : Allconv8网络
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Allconv8(nn.Module):
    def __init__(self , num_classes, add_bn = False, add_dropout = False,conv_8_id = 1):
        '''
        num_classes : 分类数量
        add_bn : 是否添加BatchNorm2d层
        add_dropout : 是否添加dropout层
        conv_8_id: 1:allcomv8_1 网络,默认allcomv8_1,
                  2:allcomv8_2 网络
                  8:allcomv8_3 网络
        '''
        super(Allconv8,self).__init__()
        self.add_dropout = add_dropout
        if add_bn:
            self.make_layers_bn()    #allcomv8_1 bn
        elif conv_8_id == 2:
            self.meke_layers_8_2() #allcomv8_2
        elif conv_8_id == 3:
            self.meke_layers_8_3()  #allcomv8_3
        else:         
            self.meke_layers_8_1()  #allcomv8_1
            
        if self.add_dropout:
            self.dropout = nn.Dropout(p=0.5)
            
        self.pool8 = nn.AvgPool2d(7,stride=1)
        self.fc = nn.Linear(256*1,num_classes)
        
        #init
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0)

    def conv_bn(self,in_channels, out_channels, kernel_size,stride, padding):
        layers = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size,stride,padding=padding),
                    nn.BatchNorm2d(out_channels)
                    )
        return layers
    
    def make_layers_bn(self):
        self.layer1 = self.conv_bn(3, 64, 3, 1, padding=1)
        self.layer2 = self.conv_bn(64, 64, 3, 1, padding=1)       
        self.layer3 = self.conv_bn(64, 128, 3, 2, padding=1)
        self.layer4 = self.conv_bn(128, 128, 3, 2, padding=1)
        self.layer5 = self.conv_bn(128, 256, 3, 2, padding=1)
        self.layer6 = self.conv_bn(256, 256, 3, 2, padding=1)
        self.layer7 = self.conv_bn(256, 256, 3, 2, padding=1) 
        
    def meke_layers_8_1(self):
        self.layer1 = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.layer2 = nn.Conv2d(64, 64, 3, 1, padding=1)       
        self.layer3 = nn.Conv2d(64, 128, 3, 2, padding=1)
        self.layer4 = nn.Conv2d(128, 128, 3, 2, padding=1)
        self.layer5 = nn.Conv2d(128, 256, 3, 2, padding=1)
        self.layer6 = nn.Conv2d(256, 256, 3, 2, padding=1)
        self.layer7 = nn.Conv2d(256, 256, 3, 2, padding=1) 
        
    def meke_layers_8_2(self):  
        self.layer1 = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.layer2 = nn.Conv2d(64, 64, 3, 2, padding=1)       
        self.layer3 = nn.Conv2d(64, 128, 3, 2, padding=1)
        self.layer4 = nn.Conv2d(128, 128, 3, 2, padding=1)
        self.layer5 = nn.Conv2d(128, 256, 3, 2, padding=1)
        self.layer6 = nn.Conv2d(256, 256, 3, 2, padding=1)
        self.layer7 = nn.Conv2d(256, 256, 3, 1, padding=1)   
        
    def meke_layers_8_3(self):
        self.layer1 = nn.Conv2d(3, 64, 3, 2, padding=1)
        self.layer2 = nn.Conv2d(64, 64, 3, 2, padding=1)       
        self.layer3 = nn.Conv2d(64, 128, 3, 2, padding=1)
        self.layer4 = nn.Conv2d(128, 128, 3, 2, padding=1)
        self.layer5 = nn.Conv2d(128, 256, 3, 2, padding=1)
        self.layer6 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.layer7 = nn.Conv2d(256, 256, 3, 1, padding=1) 
        
    def forward(self , x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        if self.add_dropout:
            self.dropout(x)
        x = self.pool8(x)
        x = x.view(-1 , 256*1) 
        out = self.fc(x)
        return out
    
    
if __name__ == "__main__":
    
    from torch.autograd import Variable
    
    allconv6 = Allconv8(20,add_bn=(True))
    print(allconv6)
    inputs = Variable(torch.randn(1,3,224,224))
    output = allconv6(inputs)
    print(output.shape)