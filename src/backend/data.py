# -*- coding = utf-8 -*-
# @Time : 2024/10/10 下午3:50
# @Author : 李兆堃
# @File : data.py
# @Software : PyCharm
from collections import deque

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from backend.configs import Config


class Data:
    def __init__(self, cfg, norm_open=False, train_test_ratio=0.7):
        self.cfg = cfg

        self.ori_data = pd.read_excel(r'D:/Git Hub Repositories/Oil Prediction/data/Y3557井生产特征10.3.xlsx')
        self.norm_open = norm_open
        self.train_test_ratio = train_test_ratio

        self.liqu_data = None
        self.pres_data = None

        self.process()

        pass

    def save_csv_file(self, data, path):
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    def read_csv_file(self, path):
        df = pd.read_csv(path)
        return df.to_numpy()
        pass

    def select_liqu_pres_point(self, data):
        liqu_pres_point = None
        for i in range(len(data)):
            if data[i][1] < (self.cfg.cons_liqu - 0.001) and data[i][2] == self.cfg.cons_pres:  # 选出来是cons_pres数据开头第一个
                liqu_pres_point = i
                break
        return liqu_pres_point

    def process(self):
        # 去除时间列
        all_data = self.ori_data.iloc[:, 1:]  # 此时为(油, 液, 压)
        all_data = all_data.to_numpy()[1:]  # 转为numpy数组，并去除第一行

        # 分割liqu和pres数据并保存
        point = self.select_liqu_pres_point(all_data)
        self.liqu_data = all_data[:point, :]
        self.pres_data = all_data[point:, :]

        # self.save_csv_file(all_data, 'data/all_data.csv')
        # self.save_csv_file(self.liqu_data, 'data/liqu_data.csv')
        # self.save_csv_file(self.pres_data, 'data/pres_data.csv')
        pass

    def build_sample_by_mode(self):  # 已分割过liqu和pres数据
        x = None
        y = None
        if self.cfg.mode == 'liqu_oil' or self.cfg.mode == 'liqu_pres':
            x = self.liqu_data[:, [0, 2]]
            if self.cfg.mode == 'liqu_oil':
                y = self.liqu_data[:, 0]  # 定液  选油为label
            elif self.cfg.mode == 'liqu_pres':
                y = self.liqu_data[:, 2]  # 定液  选压为label
            # 返回liqu数据
        elif self.cfg.mode == 'pres_oil' or self.cfg.mode == 'pres_liqu':
            x = self.pres_data[:, [0, 1]]
            if self.cfg.mode == 'pres_oil':
                y = self.pres_data[:, 0]  # 定压  选油为label
            elif self.cfg.mode == 'pres_liqu':
                y = self.pres_data[:, 1]  # 定压  选液为label
            # 返回pres数据
        return x, y

    def norm(self, x):
        scaler = MinMaxScaler((0, 1))
        sca_x = scaler.fit_transform(x)
        if self.cfg.mode == 'liqu_oil' or self.cfg.mode == 'liqu_pres':
            joblib.dump(scaler, 'D:/Git Hub Repositories/Oil Prediction/tool/liqu_scaler.pkl')
        elif self.cfg.mode == 'pres_oil' or self.cfg.mode == 'pres_liqu':
            joblib.dump(scaler, 'D:/Git Hub Repositories/Oil Prediction/tool/pres_scaler.pkl')

        return sca_x
        pass

    def split_train_test(self):
        x, y = self.build_sample_by_mode()
        if self.norm_open:
            x = self.norm(x)

        train_test_point = int(len(x) * self.train_test_ratio)
        train_x = x[: train_test_point]
        train_y = y[: train_test_point]
        test_x = x[train_test_point:]
        test_y = y[train_test_point:]
        return train_x, train_y, test_x, test_y

    def build_deq(self, X, Y):
        x_deq = []
        # deq = deque(maxlen=self.cfg.mem_days)
        deq = []
        for i in range(len(X)):
            if len(deq) == self.cfg.mem_days:
                deq = deq[1:]
            deq.append(X[i])
            temp = deq
            if len(deq) == self.cfg.mem_days:
                x_deq.append(temp)
        x_deq = x_deq[:-self.cfg.pre_days]  # 去掉最后几个，因为没有对应的label
        y_deq = Y[self.cfg.mem_days + self.cfg.pre_days - 1:]
        return np.array(x_deq), np.array(y_deq)

    def get_train_data(self):
        train_x, train_y, _, _ = self.split_train_test()
        train_x_deq, train_y_deq = self.build_deq(train_x, train_y)
        return train_x_deq, train_y_deq

    def get_test_data(self):
        _, _, test_x, test_y, = self.split_train_test()
        test_x_deq, test_y_deq = self.build_deq(test_x, test_y)
        return test_x_deq, test_y_deq

    def get_data_loader(self, x, y, config):
        dataset = TensorDatasets(x, y)
        # print(dataset[0:3])
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=config.is_shuffle, drop_last=True)


class TensorDatasets(Dataset):  # 继承Dataset父类
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


if __name__ == '__main__':
    mode = 'liqu_pres'
    cons_liqu = 40
    cons_pres = 85

    cfg = Config(mode, cons_liqu, cons_pres)
    data = Data(cfg, norm_open=True)
    train_x, train_y = data.get_train_data()
    test_x, test_y = data.get_test_data()
    print(train_x.shape, train_y.shape)


