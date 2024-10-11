# -*- coding = utf-8 -*-
# @Time : 2024/10/10 下午5:12
# @Author : 李兆堃
# @File : main.py
# @Software : PyCharm
import matplotlib.pyplot as plt

from apply import prediction
from configs import Config
from train import train


def draw(pred, label):
    x = [i for i in range(len(pred))]
    # oil_pred = pred[:, 0]
    pres_pred = pred[:, 0]
    # oil_label = label[:, 0]
    pres_label = label[:, 0]

    # plt.plot(x, oil_pred, label='oil_pred', color='black')
    # plt.plot(x, oil_label, label='oil_label', color='red')
    plt.plot(x, pres_pred, label='pres_pred', color='black')
    plt.plot(x, pres_label, label='pres_label', color='red')

    plt.show()
    pass


if __name__ == '__main__':
    # cfg = Config(3, 2, num_epoch=30, batch_size=1, num_layers=1, hidden_size=8, learn_rate=0.2)
    # train(cfg)
    #
    pred, label = prediction()
    draw(pred, label)
