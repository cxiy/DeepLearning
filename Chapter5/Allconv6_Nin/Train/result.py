# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
import numpy as np
def plot_line(list_x, list_y,  label_name, line_type, out_dir,save_name):
    """
    绘制acc曲线
    """
    for i in range(len(list_x)):
        plt.plot(list_x[i], list_y[i],line_type[i], label=label_name[i])


    plt.ylabel("acc")
    plt.xlabel('Epoch')

    location = 'lower right'
    plt.legend(loc=location)
    plt.ylim((0.3,1))
    y_ticks = np.arange(0.3,1,0.1)
    plt.yticks(y_ticks)

    # plt.title("")
    plt.savefig(os.path.join(out_dir, save_name + '.png'))
    plt.show()
    plt.close()
    
def read_acc_file(filename,path_list, label_name,line_type,save_path,save_name) :
    x = []
    y = []
    for i in range(len(path_list)):
       acc_path = os.path.join("./results",path_list[i] , filename)
       if os.path.exists(acc_path):
           with open(acc_path, 'r') as files:
               list_x = []
               list_y = []
               lines = files.readlines()
               for j,line in enumerate(lines):
                   list_y.append(float(line))
                   list_x.append(j)
               x.append(list_x)
               y.append(list_y)
    plot_line(x,y,label_name, line_type,save_path,save_name)
   

def drawn_conv2_bottleneck():
    filename = 'val_acc.txt'
    path_list = ["05-15_16-53_allconv6_baseline_adm",
                 "05-15_17-48_allconv6_bottleneck_conv2_adm",
                 ]
    label_name = ["Allconv6_baseline","conv2_bottleneck"]
    line_type = ["-",'--']
    save_path = "./draw_result"
    save_name = "5.9_conv2_bottleneck"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)
   
def drawn_conv_bottleneck():
    filename = 'val_acc.txt'
    path_list = ["05-15_16-53_allconv6_baseline_adm",
                 "05-15_17-48_allconv6_bottleneck_conv2_adm",
                 "05-15_18-30_allconv6_bottleneck_conv3_adm",
                 "05-15_21-52_allconv6_bottleneck_conv4_adm",
                 "05-15_22-35_allconv6_bottleneck_conv5_adm"
                 ]
    label_name = ["Allconv6_baseline","conv2_bottleneck","conv3_bottleneck",
                  "conv4_bottleneck","conv5_bottleneck"]
    line_type = ["-",'--','-.',"-",'--']
    save_path = "./draw_result"
    save_name = "5.10_conv_bottleneck"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)
   
def drawn_nin_nocccp():
    filename = 'val_acc.txt'
    path_list = ["05-20_21-18_no_cccp",
                 "05-21_11-27_cccp_7_baseline",
                 ]
    label_name = ["nin_nocccp","nin"]
    line_type = ["-",'--']
    save_path = "./draw_result"
    save_name = "5.13_conv2_bottleneck"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)
   

if __name__ == "__main__":
    
    # drawn_conv2_bottleneck()
    # drawn_conv_bottleneck()
    drawn_nin_nocccp()
    