# -*- coding = utf-8 -*-
# @Time : 2024/10/10 下午6:54
# @Author : 李兆堃
# @File : test.py
# @Software : PyCharm
import os

import joblib
import numpy as np
import pandas as pd
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


def test_data_init(cfg):
    data = Data(cfg, True, train_test_ratio=cfg.train_test_ratio)
    test_x, test_y = data.get_test_data()
    test_liqu_dataloader = data.get_data_loader(test_x, test_y, cfg)

    return test_liqu_dataloader
    pass


def prediction(mode, train_test_ratio=None):
    net, cfg = prediction_init(mode)
    if train_test_ratio is not None:
        cfg.train_test_ratio = train_test_ratio

    prediction_dataloader = test_data_init(cfg)

    net.eval()
    pred_list = []
    label_list = []
    with torch.no_grad():
        for i, batch in enumerate(prediction_dataloader):
            x_batch, y_batch = batch
            x_batch = torch.as_tensor(x_batch).to(cfg.devices)
            y_batch = torch.as_tensor(y_batch).to(cfg.devices)

            pred = net.forward(x_batch)  # 对data进行前向推理，得到预测
            pred_list.append(pred.cpu().numpy())
            label_list.append(y_batch[0].cpu().numpy())
        return np.array(pred_list), np.array(label_list), cfg
    pass

def process(data, cfg):
    cons_liqu = cfg.cons_liqu
    cons_pres = cfg.cons_pres

    pred_oil = np.concatenate((data[0, 0].flatten(), data[2, 0].flatten()), axis=0)
    pred_liqu = np.concatenate((np.array([cons_liqu for i in range(len(pred_oil) - len(data[3, 0]))]), data[3, 0].flatten()), axis=0)
    pred_pres = np.concatenate((data[1, 0].flatten(), np.array([cons_pres for i in range(len(pred_oil) - len(data[1, 0]))])), axis=0).flatten()
    # label = pd.read_excel(r'D:/Git Hub Repositories/Oil Prediction/data/Y3557井生产特征10.3.xlsx').iloc[:, 1:].to_numpy()    # 油， 液， 压
    label_oil = np.concatenate((data[0, 1].flatten(), data[2, 1].flatten()), axis=0)
    label_liqu = np.concatenate((np.array([cons_liqu for i in range(len(pred_oil) - len(data[3, 1]))]), data[3, 1].flatten()), axis=0)
    label_pres = np.concatenate((data[1, 1].flatten(), np.array([cons_pres for i in range(len(pred_oil) - len(data[1, 1]))])), axis=0).flatten()
    pred = [pred_oil, pred_liqu, pred_pres]
    label = [label_oil, label_liqu, label_pres]
    return pred, label
    pass

def draw(pred, label):
    x = [i for i in range(len(pred[0]))]

    plt.plot(x, pred[0], label='pred_oil', color='black')
    plt.plot(x, pred[1], label='pred_liqu', color='black')
    plt.plot(x, pred[2], label='pred_pres', color='black')

    plt.plot(x, label[0], label='label_oil', color='red')
    plt.plot(x, label[1], label='label_liqu', color='red')
    plt.plot(x, label[2], label='label_pres', color='red')

    plt.legend()
    plt.show()
    pass


def test():
    mode_list = ['liqu_oil', 'liqu_pres', 'pres_oil', 'pres_liqu']
    result = []
    config = None

    for mode in mode_list:
        pred, label, cfg = prediction(mode, 0)      # 全量预测测试
        result.append([pred, label])
        config = cfg

    pred, label = process(np.array(result), config)
    draw(pred, label)


if __name__ == '__main__':
    test()


