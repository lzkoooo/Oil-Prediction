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
from matplotlib import pyplot as plt, rcParams

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


def draw_item(pred, label):
    plt.rcParams['font.size'] = 13
    rcParams['font.sans-serif'] = ['SimHei']
    x = [i for i in range(len(pred[0]))]

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axs[0].set_title('产油预测')
    axs[0].plot(x, label[0], label='真实值', color='r')
    axs[0].plot(x, pred[0], label='预测值', color='g')
    axs[0].set_xlabel('天数/d')
    axs[0].set_ylabel('日产油量/m^3')

    axs[1].set_title('产液预测')
    axs[1].plot(x, label[1], label='真实值', color='r')
    axs[1].plot(x, pred[1], label='预测值', color='g')
    axs[1].set_xlabel('天数/d')
    axs[1].set_ylabel('日产液量/m^3')

    axs[2].set_title('井底压力预测')
    axs[2].plot(x, label[2], label='真实值', color='r')
    axs[2].plot(x, pred[2], label='预测值', color='g')
    axs[2].set_xlabel('天数/d')
    axs[2].set_ylabel('井底压力/bar')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=14)
    fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.3, hspace=0.4, wspace=0.4)
    fig.suptitle('单井预测对比')

    plt.tight_layout()
    plt.show()
    plt.savefig(r'../result/item.png', dpi=300, bbox_inches='tight')
    pass


def draw_all(pred, label):
    plt.rcParams['font.size'] = 13
    rcParams['font.sans-serif'] = ['SimHei']
    x = [i for i in range(len(pred[0]))]

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axs[0].set_title('真实值')
    axs[0].plot(x, label[0], label='真实值', color='r')
    axs[0].plot(x, label[1], label='真实值', color='r')
    axs[0].plot(x, label[2], label='真实值', color='r')
    axs[0].set_xlabel('天数/d')

    axs[1].set_title('预测值')
    axs[1].plot(x, pred[0], label='预测值', color='g')
    axs[1].plot(x, pred[1], label='预测值', color='g')
    axs[1].plot(x, pred[2], label='预测值', color='g')
    axs[1].set_xlabel('天数/d')

    fig.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.3, hspace=0.4, wspace=0.4)
    fig.suptitle('单井预测对比')

    plt.tight_layout()
    plt.show()
    plt.savefig(r'../result/all.png', dpi=300, bbox_inches='tight')
    pass


def test():
    mode_list = ['liqu_oil', 'liqu_pres', 'pres_oil', 'pres_liqu']
    result = []
    config = None

    for mode in mode_list:
        pred, label, cfg = prediction(mode, 0)      # 全量预测测试
        result.append([np.array(pred), np.array(label)])
        config = cfg

    pred, label = process(np.array(result), config)
    draw_item(pred, label)
    draw_all(pred, label)


if __name__ == '__main__':
    test()


