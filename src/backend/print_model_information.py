# -*- coding = utf-8 -*-
# @Time : 2024/10/12 下午6:39
# @Author : 李兆堃
# @File : print_model_information.py
# @Software : PyCharm
import torch

if __name__ == '__main__':
    check_point = torch.load(r'D:\Git Hub Repositories\Oil Prediction/model/pres_liqu/min_loss_model.pth')
    cfg = check_point['config']
    print(cfg.num_layers)
    print(cfg.hidden_size)
    print(cfg.batch_size)

