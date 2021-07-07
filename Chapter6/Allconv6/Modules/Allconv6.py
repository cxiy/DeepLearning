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
    def __init__(self , num_classes ,add_bn = False,conv6_id = 0):
        '''
        num_classes : 分类数量
        add_bn : 是否添加BatchNorm2d层
        conv6_id : 不同通道数Allconv6网络，
                  0：Allconv6,默认， 1：Allconv6_1, 2：Allconv6_2, 3：Allconv6_3,
                  4：Allconv6_4, 5：Allconv6_5, 6：Allconv6_6, 7：Allconv6_7,
                  8：Allconv6_8
        '''
        super(Allconv6,self).__init__()
        self.conv6_id = conv6_id
        self.num_classes = num_classes
        self.channels = self.get_channels_list()
        if add_bn:
            self.layers = self.make_layers_bn()  #allcomv6 bn
        else:
            self.layers = self.meke_layers()   #allcomv6_id
            
        self.fc = nn.Linear(self.channels[4],self.num_classes)

        #init
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0)
            
    
    def make_layers_bn(self):     
        layers = nn.Sequential(
            nn.Conv2d(3, self.channels[0], 3, 2, padding=1),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU(),
            nn.Conv2d(self.channels[0], self.channels[1], 3, 2, padding=1),
            nn.BatchNorm2d(self.channels[1]),
            nn.ReLU(),
            nn.Conv2d(self.channels[1], self.channels[2], 3, 2, padding=1),
            nn.BatchNorm2d(self.channels[2]),
            nn.ReLU(),
            nn.Conv2d(self.channels[2], self.channels[3], 3, 2, padding=1),
            nn.BatchNorm2d(self.channels[3]),
            nn.ReLU(),
            nn.Conv2d(self.channels[3], self.channels[4], 3, 2, padding=1),
            nn.BatchNorm2d(self.channels[4]),
            nn.ReLU(),
            nn.AvgPool2d(7,stride=1)
            )
        
        return layers
        
        
    def meke_layers(self):
        layers = nn.Sequential(
            nn.Conv2d(3, self.channels[0], 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels[0], self.channels[1], 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels[1], self.channels[2], 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels[2], self.channels[3], 3, 2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.channels[3], self.channels[4], 3, 2, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(7,stride=1)
            )
        return layers
        
    def get_channels_list(self):
        channels_list = []
        if self.conv6_id == 1:
            channels_list = [16,16,32,32,64]
        elif self.conv6_id == 2:
            channels_list = [32,32,32,32,64]
        elif self.conv6_id == 3:
            channels_list = [16,16,64,64,64]
        elif self.conv6_id == 4:
            channels_list = [16,16,32,32,128]
        elif self.conv6_id == 5:
            channels_list = [32,32,64,64,128]
        elif self.conv6_id == 6:
            channels_list = [64,64,64,64,128]
        elif self.conv6_id == 7:
            channels_list = [32,32,128,128,128]
        elif self.conv6_id == 8:
            channels_list = [32,32,64,64,256]
        else:
            channels_list = [64,64,128,128,256]
        return channels_list
    
    def forward(self , x):
        out = self.layers(x)
        out = out.view(-1,out.shape[1]) 
        out = self.fc(out)
        return out
    
    
if __name__ == "__main__":
    
    from torch.autograd import Variable
    
    allconv6 = Allconv6(20,add_bn=True,conv6_id=8)
    print(allconv6)
    inputs = Variable(torch.randn(1,3,224,224))
    output = allconv6(inputs)
    print(output.shape)