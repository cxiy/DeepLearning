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
    
def drawn_lr_scheduler():
    filename = 'val_acc.txt'
    path_list = ["05-13_16-52_lr_fix_0.01","05-21_11-50_lr_step","05-13_16-07_lr_multistep",
                 "05-13_19-41_lr_exp","05-13_20-20_lr_Inv","05-13_21-02_lr_ploy"]
    label_name = ["fix","step","multistep","exp","Inv","Poly"]
    line_type = ["-",'--',"-.","-","-","--"]
    save_path = "./draw_result"
    save_name = "4.10_lr_scheduler"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)
    
def draw_lr_fixed_step():
    filename = 'val_acc.txt'
    path_list = ["05-13_16-52_lr_fix_0.01","05-13_17-36_lr_fix_0.001","05-21_11-50_lr_step"]
    label_name = ["fix=0.01","fix=0.001","step"]
    line_type = ["-",'-',"-"]
    save_path = "./draw_result"
    save_name = "4.11_lr_scheduler_fixed_step"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)
    
def drawn_momentum():
    filename = 'val_acc.txt'
    path_list = ["05-16_07-00_momentum=0","05-16_07-42_momentum=0.8","05-16_10-27_momentum=0.9",
                 "05-16_11-19_momentum=0.95","05-16_14-35_momentum=0.99"]
    label_name = ["m=0","m=0.8","m=0.9","m=0.95","m=0.99"]
    line_type = ["-",'--',"-.","-","--"]
    save_path = "./draw_result"
    save_name = "4.13_momentum"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)
    
def drawn_place20():
    filename = 'val_acc.txt'
    path_list = ["05-19_19-24_deep_allconv5_adm_place20",
                 "05-19_20-30_deep_allconv6_adm_place20",
                 "05-19_22-37_deep_allconv7_1_adm_place20"]
    label_name = ["allconv5","allconv6","allconv7_1"]
    line_type = ["-",'--',"-.","-"]
    save_path = "./draw_result"
    save_name = "place20"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)
    
def drawn_bn():
    filename = 'val_acc.txt'
    path_list = ["05-18_17-56_deep_allconv6_bn",
                 "05-18_18-23_deep_allconv7_1_bn",
                 "05-18_19-02_deep_allconv8_1_bn"]
    label_name = ["allconv6_bn","allconv7_1_bn","allconv8_1_bn"]
    line_type = ["-",'--',"-.","-"]
    save_path = "./draw_result"
    save_name = "bn"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)
    
def drawn_dropout():
    filename = 'val_acc.txt'
    path_list = ["05-18_20-07_deep_allconv6_dropout",
                 "05-19_08-01_deep_allconv7_1_dropout",
                 "05-19_11-50_deep_allconv8_1_dropout"]
    label_name = ["allconv6_dropout","allconv7_dropout","allconv8_dropout"]
    line_type = ["-",'--',"-."]
    save_path = "./draw_result"
    save_name = "_dropout"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)
    
def drawn_nag():
    filename = 'val_acc.txt'
    path_list = ["05-16_15-43_optim_nag","05-21_11-50_lr_step"]
    label_name = ["nag","SGD_baseline"]
    line_type = ["-",'--']
    save_path = "./draw_result"
    save_name = "4.14_nsg_step"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)
    
def drawn_adgrad():
    filename = 'val_acc.txt'
    path_list = ["05-14_11-22_optim_adagrads_0.001",
                 "05-13_23-35_optim_adagrads_fixed0.01",
                 "05-14_21-32_optim_adagrads_step",
                 "05-21_11-50_lr_step"]
    label_name = ["AdaGrad(0.001)","AdaGrad(0.01)","AdaGrad(step)","SGD_baseline"]
    line_type = ["-",'--',"-.","-"]
    save_path = "./draw_result"
    save_name = "4.15_adgrad_baseline"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)
    
def drawn_adadelta():
    filename = 'val_acc.txt'
    path_list = ["05-13_22-09_optim_adadelta","05-21_11-50_lr_step"]
    label_name = ["Adadelta(fixed)","SGD_baseline"]
    line_type = ["-",'--',]
    save_path = "./draw_result"
    save_name = "4.16_adadelta_baseline"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)

def drawn_RMSProp():
    filename = 'val_acc.txt'
    path_list = ["05-16_16-32_rms_rmddecay=0.9_lr=0.001",
                 "07-01_20-56_rms_rmddecay=0.99_lr=0.001",
                 "05-21_11-50_lr_step"]
    label_name = ["rms_decay=0.9","rms_decay=0.99","SGD_baseline"]
    line_type = ["-",'--',"-.","-"]
    save_path = "./draw_result"
    save_name = "4.17_rms"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)
    
def drawn_adam():
    filename = 'val_acc.txt'
    path_list = ["05-15_14-00_optim_adm_0.00001_0.9_0.99",
                 "05-15_13-18_optim_adm_0.0001_0.9_0.99",           
                 "05-15_14-37_optim_adm_0.001_0.9_0.99",
                 "05-15_16-29_optim_adm_0.003_0.9_0.99",
                 "05-15_12-37_optim_adm_0.01_0.9_0.99"]
    label_name = ["0.00001","0.0001","0.001","0.003","0.01"]
    line_type = ["-",'--',"-.","-","-."]
    save_path = "./draw_result"
    save_name = "4.18_adam_0.99"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)

def drawn_adam_2():
    filename = 'val_acc.txt'
    path_list = ["05-15_18-31_optim_adm_0.00001_0.9_0.999",
                 "05-15_17-49_optim_adm_0.0001_0.9_0.999",
                 "05-15_19-18_optim_adm_0.001_0.9_0.999",
                 "05-15_22-35_optim_adm_0.003_0.9_0.999"]
    label_name = ["0.00001","0.0001","0.001","0.003"]
    line_type = ["-",'--',"-.","-"]
    save_path = "./draw_result"
    save_name = "4.19_adam_0.999"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)

def drawn_optim():
    filename = 'val_acc.txt'
    path_list = ["05-16_10-27_momentum=0.9",
                 "05-16_15-43_optim_nag",
                 "05-13_23-35_optim_adagrads_fixed0.01",
                 "05-13_22-09_optim_adadelta",
                 "05-16_16-32_rms_rmddecay=0.9_lr=0.001",
                 "05-15_14-37_optim_adm_0.001_0.9_0.99"]
    label_name = ["momentum","Nesterov","AdaGrad","Adadelta","RMSProp","Adam"]
    line_type = ["-",'--',"-.","-","--","-."]
    save_path = "./draw_result"
    save_name = "4.20_optimizer"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)
    
def drawn_model():
    filename = 'val_acc.txt'
    path_list = ["05-16_17-10_deep_allconv5","05-16_18-41_deep_allconv6","05-17_18-28_deep_allconv7_1",
                 "05-18_15-27_deep_allconv8_1"]
    label_name = ["allconv5","allconv6","allconv7_1","allconv8_1"]
    line_type = ["-",'--',"-.","-"]
    save_path = "./draw_result"
    save_name = "4.21_model"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)
    
def drawn_allconv7():
    filename = 'val_acc.txt'
    path_list = ["05-17_18-28_deep_allconv7_1",
                 "05-17_20-32_deep_allconv7_2"]
    label_name = ["Allconv7_1","Allconv7_2"]
    line_type = ["-",'--']
    save_path = "./draw_result"
    save_name = "4.22_Allconv7"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)

def drawn_allconv8():
    filename = 'val_acc.txt'
    path_list = ["05-18_15-27_deep_allconv8_1",
                 "05-18_16-38_deep_allconv8_2",
                 "05-18_17-16_deep_allconv8_3"]
    label_name = ["Allconv8_1","Allconv8_2","Allconv8_3"]
    line_type = ["-",'--','-.']
    save_path = "./draw_result"
    save_name = "4.23_Allconv8"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)
    
def drawn_allconv_bn():
    filename = 'val_acc.txt'
    path_list = ["05-18_17-56_deep_allconv6_bn",
                 "05-18_18-23_deep_allconv7_1_bn",
                 "05-18_19-02_deep_allconv8_1_bn"]
    label_name = ["Allconv6_bn","Allconv7_bn","Allconv8_bn"]
    line_type = ["-",'--','-.']
    save_path = "./draw_result"
    save_name = "4.24_Allconv_bn"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)

def drawn_allconv_dropout():
    filename = 'val_acc.txt'
    path_list = ["05-18_20-07_deep_allconv6_dropout",
                 "05-19_08-01_deep_allconv7_1_dropout",
                 "05-19_11-50_deep_allconv8_1_dropout"]
    label_name = ["Allconv6_dropout","Allconv7_dropout","Allconv8_dropout"]
    line_type = ["-",'--','-.']
    save_path = "./draw_result"
    save_name = "4.26_Allconv_dropout"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)

def drawn_mobilenet():
    filename = 'val_acc.txt'
    path_list = ["05-19_14-01_MobileNet_10",
                 "05-19_14-26_MobileNet_12",
                 "05-19_14-51_MobileNet_14",
                 "05-19_15-18_MobileNet_16",
                 "05-19_16-22_MobileNet_20",
                 "05-19_16-48_MobileNet_24",
                 "05-19_17-20_MobileNet_26"]
    label_name = ["V10","V12","V14","V16","V20","V24","V26"]
    line_type = ["-",'--','-.',"-",'--','-.','-']
    save_path = "./draw_result"
    save_name = "4.32_mobile"
    read_acc_file(filename,path_list,label_name,line_type,save_path,save_name)
   

if __name__ == "__main__":
    
    # drawn_lr_scheduler()
    # draw_lr_fixed_step()
    # drawn_momentum()
    # drawn_bn()
    # drawn_dropout()
    # drawn_nag()
    # drawn_adgrad()
    # drawn_adadelta()
    # drawn_RMSProp() 
    # drawn_adam()
    # drawn_adam_2()
    # drawn_optim()
    # drawn_model()
    # drawn_allconv7()
    # drawn_allconv8()
    # drawn_allconv_bn()
    # drawn_allconv_dropout()
    drawn_mobilenet()
    