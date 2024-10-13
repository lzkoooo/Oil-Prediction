# -*- coding = utf-8 -*-
# @Time : 2024/10/13 上午10:25
# @Author : 李兆堃
# @File : apply.py
# @Software : PyCharm
import os

import joblib
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt


def prediction_init():
    mode_list = ['liqu_oil', 'liqu_pres', 'pres_oil', 'pres_liqu']
    net_list = []

    for mode in mode_list:
        model_path = fr'D:/Git Hub Repositories/Oil Prediction/model/{mode}/end_model.pth'

        if os.path.exists(fr'D:/Git Hub Repositories/Oil Prediction/model/{mode}/min_loss_model.pth'):
            model_path = fr'D:/Git Hub Repositories/Oil Prediction/model/{mode}/min_loss_model.pth'
        checkpoint = torch.load(model_path)

        net = checkpoint['model']
        devices = checkpoint['config'].devices
        net.eval()

        net.to(devices)
        net_list.append(net)

    return net_list, devices
    pass


def build_norm_data(data, mode):
    norm_data = None

    if mode == 'cons_liqu':
        scaler = joblib.load('D:/Git Hub Repositories/Oil Prediction/tool/liqu_scaler.pkl')

        oil = scaler.transform(data)[:, 0]
        pres = scaler.transform(data)[:, 1]
        norm_data = np.column_stack((oil, pres))

    elif mode == 'cons_pres':
        scaler = joblib.load('D:/Git Hub Repositories/Oil Prediction/tool/pres_scaler.pkl')

        oil = scaler.transform(data)[:, 0]
        liqu = scaler.transform(data)[:, 1]
        norm_data = np.column_stack((oil, liqu))

    return norm_data
    pass


def continuous_prediction(init_data, pre_days):
    cons_liqu = 40
    cons_pres = 85
    mode_switch = pre_days - 814
    current_data = init_data
    result = []
    net_list, devices = prediction_init()

    with torch.no_grad():
        while True:
            if pre_days <= mode_switch or pre_days == 0:  # 切换模式
                break

            norm_data = build_norm_data(current_data[:, [0, 2]], 'cons_liqu')
            pred_oil = net_list[0].forward(torch.as_tensor(norm_data).to(devices)).cpu().numpy()
            pred_pres = net_list[1].forward(torch.as_tensor(norm_data).to(devices)).cpu().numpy()
            new_data = np.array([float(pred_oil), float(cons_liqu), float(pred_pres)])
            current_data = np.concatenate((current_data[1:, :], [new_data]), axis=0)
            result.append(new_data)

            pre_days -= 1

            # if pred_pres < cons_pres:  # 切换模式
            #     break

        while True:
            if pre_days == 0:  # 结束
                break

            norm_data = build_norm_data(current_data[:, [0, 1]], 'cons_pres')
            pred_oil = net_list[2].forward(torch.as_tensor(norm_data).to(devices)).cpu().numpy()
            pred_liqu = net_list[3].forward(torch.as_tensor(norm_data).to(devices)).cpu().numpy()
            new_data = np.array([float(pred_oil), float(pred_liqu), float(cons_pres)])
            current_data = np.concatenate((current_data[1:, :], [new_data]), axis=0)
            result.append(new_data)

            pre_days -= 1

    return np.array(result)
    pass


if __name__ == '__main__':
    prediction_days = 2000

    init_data = np.array([[12.03142548, 40, 99.34012604], [12.04320621, 40.00000095, 98.45579529],
                          [12.04172993, 39.99999905, 98.22970581]])
    pred = continuous_prediction(init_data, prediction_days)
    label = pd.read_excel(r'D:/Git Hub Repositories/Oil Prediction/data/Y3557井生产特征10.3.xlsx').iloc[:, 1:].to_numpy()[1+3+1: prediction_days + 1+3+1]
    x = [i for i in range(prediction_days)]

    plt.plot(x, pred[:, 0], label='pred_oil', color='black')
    plt.plot(x, pred[:, 1], label='pred_liqu', color='black')
    plt.plot(x, pred[:, 2], label='pred_pres', color='black')

    plt.plot(x, label[:, 0], label='label_oil', color='red')
    plt.plot(x, label[:, 1], label='label_liqu', color='red')
    plt.plot(x, label[:, 2], label='label_pres', color='red')

    plt.legend()
    plt.show()

