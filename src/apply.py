# -*- coding = utf-8 -*-
# @Time : 2024/10/10 下午6:54
# @Author : 李兆堃
# @File : apply.py
# @Software : PyCharm
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from backend.data import Data


def prediction_init(mode):
    model_path = fr'D:/Git Hub Repositories/Oil Prediction/model/{mode}/end_model.pth'

    if os.path.exists(fr'D:/Git Hub Repositories/Oil Prediction/model/{mode}/min_loss_model.pth'):
        model_path = fr'D:/Git Hub Repositories/Oil Prediction/model/{mode}/min_loss_model.pth'
    checkpoint = torch.load(model_path)

    # 获取config
    cfg = checkpoint['config']
    cfg.is_shuffle = False
    # cfg.batch_size = 1
    net = checkpoint['model']
    net.to(cfg.devices)
    return net, cfg
    pass


def pred_data_init(cfg):
    data = Data(cfg, True, train_test_ratio=0.2)
    test_x, test_y = data.get_test_data()
    test_liqu_dataloader = data.get_data_loader(test_x, test_y, cfg)

    return test_liqu_dataloader
    pass


def prediction(mode):
    net, cfg = prediction_init(mode)
    test_liqu_dataloader = pred_data_init(cfg)

    net.eval()
    pred_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(test_liqu_dataloader):
            x_batch, y_batch = batch
            x_batch = torch.as_tensor(x_batch).to(cfg.devices)
            y_batch = torch.as_tensor(y_batch).to(cfg.devices)

            pred = net.forward(x_batch)  # 对data进行前向推理，得到预测
            pred_list.append(pred.cpu().numpy())
            label_list.append(y_batch[0].cpu().numpy())
        return np.array(pred_list), np.array(label_list)
    pass


def draw(pred, label):
    x = [i for i in range(len(pred))]

    plt.plot(x, pred[:, 0], label='pres_pred', color='black')
    plt.plot(x, label, label='pres_label', color='red')

    plt.show()
    pass


if __name__ == '__main__':
    mode = 'pres_oil'
    #
    pred, label = prediction(mode)
    draw(pred, label)
