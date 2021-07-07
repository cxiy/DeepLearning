# -*- coding: utf-8 -*-
"""
# @file name  : config.py
# @author     : cenzy
# @date       : 2021-05-11
# @brief      : 配置文件
"""
import os
import torch.optim as optim
from Modules.CccpNet import CccpNet
from DataSets.ghim10k_dataset import ghim10k_dataset

#数据路径
data_dir = os.path.join("/media/cenzy/E1AFE5F4AEE8846A/BookCodeCurrence/data/Place20")
train_txt_path = os.path.join(data_dir,"place20trainshuffle.txt")
val_txt_path = os.path.join(data_dir,"place20val.txt")

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
learning_rate = 0.0001

#模型
model = CccpNet(cccp_count=4)

#优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9, 0.999), weight_decay=0.000) 

#学习率策略
scheduler = None  

 #预训练模型路径
fine_dic_path = ""

#日志文件名
log_file_add_name = "cccp_4"