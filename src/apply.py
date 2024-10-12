# -*- coding = utf-8 -*-
# @Time : 2024/10/10 下午6:54
# @Author : 李兆堃
# @File : apply.py
# @Software : PyCharm
import numpy as np
import torch
from matplotlib import pyplot as plt

from backend.data import Data


def prediction_init(mode):
    checkpoint = torch.load(fr'model/{mode}/best_model.pth')
    cfg = checkpoint['config']
    cfg.is_shuffle = False
    cfg.batch_size = 1
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


def prediction():
    net, cfg = prediction_init()
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
    oil_pred = pred[:, 0]
    pres_pred = pred[:, 0]
    oil_label = label[:, 0]
    pres_label = label[:, 0]

    # plt.plot(x, oil_pred, label='oil_pred', color='black')
    # plt.plot(x, oil_label, label='oil_label', color='red')
    plt.plot(x, pres_pred, label='pres_pred', color='black')
    plt.plot(x, pres_label, label='pres_label', color='red')

    plt.show()
    pass


if __name__ == '__main__':

    #
    pred, label = prediction()
    draw(pred, label)