# -*- coding: utf-8 -*-
"""
# @file name  : config.py
# @author     : cenzy
# @date       : 2021-05-11
# @brief      : 配置文件
"""
import os
import torch.optim as optim
from Modules.Allconv8 import Allconv8
from DataSets.ghim10k_dataset import ghim10k_dataset

#数据路径
data_dir = os.path.join("/media/cenzy/E1AFE5F4AEE8846A/BookCodeCurrence", "data")
train_txt_path = os.path.join(data_dir,"GHIM-20","list_train_shuffle.txt")
val_txt_path = os.path.join(data_dir,"GHIM-20","list_val_shuffle.txt")

#数据集
train_data = ghim10k_dataset(train_txt_path,data_dir)
valid_data = ghim10k_dataset(val_txt_path,data_dir)
    
#分类数量
num_classes = 20
#Epochs 
max_epoch = 60
#Batch size
batch_size = 64
#学习率
learning_rate = 0.01

#模型
model = Allconv8(num_classes,conv_8_id=3)

#优化器
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.005) 

#学习率策略
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)   

 #预训练模型路径
fine_dic_path = ""

#日志文件名
log_file_add_name = "deep_allconv8_3"