# -*- coding = utf-8 -*-
# @Time : 2024/10/12 下午6:39
# @Author : 李兆堃
# @File : print_model_information.py
# @Software : PyCharm
import torch

if __name__ == '__main__':
    for mode in ['liqu_oil', 'liqu_pres', 'pres_oil', 'pres_liqu']:
        check_point = torch.load(rf'D:\Git Hub Repositories\Oil Prediction/model/{mode}/min_loss_model.pth')
        cfg = check_point['config']
        print(check_point['loss'])

