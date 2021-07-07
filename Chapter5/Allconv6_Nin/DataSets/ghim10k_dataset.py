# -*- coding: utf-8 -*-
"""
# @file name  : ghim10k_dataset.py
# @author     : cenzy
# @date       : 2021-05-10
# @brief      : 数据集Dataset定义
"""

import os
import torch
import torch.utils.data as data
from PIL import Image

# 数据集
class ghim10k_dataset(data.Dataset):
    # 自定义的参数
    def __init__(self, list_path, data_dir,transforms=None):
        """
        :param list_path, 数据集标记文件路径
        :param data_dir:  数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        self.list_path = list_path
        self.data_dir = data_dir
        img_paths,labels = self._get_img_info()
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms
        
    def set_transform(self, transforms):
        self.transforms = transforms
        
    # 返回图片个数
    def __len__(self):
        return len(self.img_paths)

    # 获取每个图片
    def __getitem__(self, item):
        # path
        img_path =self.img_paths[item]
        #read image
        img = Image.open(img_path).convert('RGB')     # 0~255
        #augmentation
        if self.transforms is not None:
             # 在这里做transform，转为tensor等等
            img = self.transforms(img)
        #read label
        label = self.labels[item]
        return img, int(label)
    
    #获取数据集图片路径，标签
    def _get_img_info(self):
        '''
        return：图片路径列表，标签列表
        '''
        img_paths = []
        label = []
        with open(self.list_path, "r") as lines:
            for line in lines:
                imgpath = os.path.join(self.data_dir,line.split(' ')[0] )
                img_paths.append(imgpath)
                label.append(line.split(' ')[1])
        return img_paths, label



from torchvision import transforms
if __name__ == "__main__":
    
    list_path = '/media/cenzy/E1AFE5F4AEE8846A/BookCodeCurrence/data/GHIM-20/list_train_shuffle.txt'
    data_path ='/media/cenzy/E1AFE5F4AEE8846A/BookCodeCurrence/data'
    transform = transforms.Compose([transforms.Resize((256,256)),
                                    transforms.RandomSizedCrop(224),
                                    transforms.ToTensor(),])
    train_data = ghim10k_dataset(list_path,data_path,transforms = transform)
    print(len(train_data))
    train_dataset = data.DataLoader(train_data, batch_size=2, shuffle=None,num_workers=4)
    for i, (img,lbl) in enumerate(train_dataset):
        print(img.shape)
        print(lbl)
