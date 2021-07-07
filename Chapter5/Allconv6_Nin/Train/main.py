# -*- coding: utf-8 -*-
"""
# @file name  : main.py
# @author     : cenzy
# @date       : 2021-05-11
# @brief      : 分类训练
"""

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# import torch.optim as optim
# from Modules.Allconv6 import Allconv6
from Train.common_tools import ModelTrainer, plot_line
# from DataSets.ghim10k_dataset import ghim10k_dataset

import config as config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
  
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    time_str += "_" + config.log_file_add_name
    log_dir = os.path.join(BASE_DIR, "results", time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    #Loas,Acc Log File
    log_trainloss = os.path.join(BASE_DIR, "results", time_str, "train_loss.txt")
    log_trainacc = os.path.join(BASE_DIR, "results", time_str, "train_acc.txt")
    log_valloss = os.path.join(BASE_DIR, "results", time_str, "val_loss.txt")
    log_valacc = os.path.join(BASE_DIR, "results", time_str, "val_acc.txt")
    if not os.path.exists(log_trainloss):
        with open(log_trainloss, 'w')as iter_trainloss_file:
            pass
    if not os.path.exists(log_trainacc):
        with open(log_trainacc, 'w') as iter_trainacc_file:
            pass
    if not os.path.exists(log_valloss):
        with open(log_valloss, 'w') as iter_valloss_file:
            pass
    if not os.path.exists(log_valacc):
        with open(log_valacc, 'w') as iter_valacc_file:
            pass
    
    # label_name = {"0": 0, "1": 1, "2": 2, "3": 3,"4":4,.....,"20":20}
    num_classes = config.num_classes 

    MAX_EPOCH = config.max_epoch     
    BATCH_SIZE = config.batch_size     
    LR = config.learning_rate 
    log_interval = 1
    val_interval = 1
    start_epoch = -1

    # ============================ step 1/5 数据 ============================
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # 构建MyDataset实例
    train_data = config.train_data
    train_data.set_transform(train_transform)
    valid_data = config.valid_data
    valid_data.set_transform(valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, num_workers=2)

    # ============================ step 2/5 模型 ============================
    model = config.model #Allconv6(num_classes)

    model.to(device)
    # ============================ step 3/5 损失函数 ============================
    criterion = nn.CrossEntropyLoss()
    
    # ============================ step 4/5 优化器 ============================
    optimizer = config.optimizer   # 选择优化器
    #学习率策略
    scheduler = config.scheduler 
    
    # 如果有保存的模型，则加载模型，并在其基础上继续训练
    if os.path.exists(config.fine_dic_path):
        checkpoint = torch.load(config.fine_dic_path)
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('无保存模型，从头开始训练！')
        
# ============================ step 5/5 训练 ============================
    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0

    for epoch in range(start_epoch + 1, MAX_EPOCH):

        # 训练(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch)
        loss_train, acc_train, mat_train = ModelTrainer.train(train_loader, model, criterion, optimizer, epoch, device, MAX_EPOCH, num_classes)
        loss_valid, acc_valid, mat_valid = ModelTrainer.valid(valid_loader, model, criterion, device, num_classes)
        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{}".format(
            epoch + 1, MAX_EPOCH, acc_train, acc_valid, loss_train, loss_valid, optimizer.param_groups[0]["lr"]))

        if scheduler is not None:
            scheduler.step()  # 更新学习率

        # 绘图
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)

        #Save Log
        with open(log_trainloss, 'a') as iter_val_file:
            iter_val_file.write('%.8f\n' % (loss_train))
        with open(log_trainacc, 'a') as iter_val_file:
            iter_val_file.write('%.8f\n' % (acc_train))
        with open(log_valloss, 'a') as iter_val_file:
            iter_val_file.write('%.8f\n' % (loss_valid))
        with open(log_valacc, 'a') as iter_val_file:
            iter_val_file.write('%.8f\n' % (acc_valid))
   
        plt_x = np.arange(1, len(loss_rec["train"])+1)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)

        if epoch > 20  and best_acc < acc_valid: #(MAX_EPOCH/2)
            best_acc = acc_valid
            best_epoch = epoch

            checkpoint = {"model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "epoch": epoch,
                      "best_acc": best_acc}

            path_checkpoint = os.path.join(log_dir, "checkpoint_best.pkl")
            torch.save(checkpoint, path_checkpoint)

    print(" done ~~~~ {}, best acc: {} in :{} epochs. ".format(datetime.strftime(datetime.now(), '%m-%d_%H-%M'),
                                                      best_acc, best_epoch))
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)






