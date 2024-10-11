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


class Data:
    def __init__(self, cfg, constant_liqu, constant_pres, norm_open=False, train_test_ratio=0.7):
        self.cfg = cfg

        self.ori_data = pd.read_excel(r'Y3557井生产特征10.3.xlsx')
        self.constant_liqu = constant_liqu
        self.constant_pres = constant_pres
        self.norm_open = norm_open
        self.train_test_ratio = train_test_ratio

        self.liqu_data = None
        self.pres_data = None

        self.x_liqu = None
        self.y_liqu = None
        self.x_pres = None
        self.y_pres = None
        self.train_x_liqu = None
        self.train_y_liqu = None
        self.test_x_liqu = None
        self.test_y_liqu = None
        self.train_x_pres = None
        self.train_y_pres = None
        self.test_x_pres = None
        self.test_y_pres = None

        self.process()
        self.build_sample()
        if self.norm_open:
            self.norm()
        self.split_train_test()
        self.train_x_liqu, self.train_y_liqu = self.build_deq(self.train_x_liqu, self.train_y_liqu)
        self.test_x_liqu, self.test_y_liqu = self.build_deq(self.test_x_liqu, self.test_y_liqu)
        self.train_x_pres, self.train_y_pres = self.build_deq(self.train_x_pres, self.train_y_pres)
        self.test_x_pres, self.test_y_pres = self.build_deq(self.test_x_pres, self.test_y_pres)
        pass

    def save_csv_file(self, data, path):
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)

    def read_csv_file(self, path):
        df = pd.read_csv(path)
        return df.to_numpy()
        pass

    def select_liqu_pres_point(self, data):
        point = None
        for i in range(len(data)):
            if data[i][1] < (self.constant_liqu - 0.001) and data[i][2] == self.constant_pres:  # 选出来是cons_pres数据开头第一个
                point = i
                break
        return point

    def process(self):
        # 去除时间列
        all_data = self.ori_data.iloc[:, 1:]  # 此时为(油, 液, 压)
        all_data = all_data.to_numpy()[1:]  # 转为numpy数组，并去除第一行
        # print(all_data.dtype)
        self.save_csv_file(all_data, 'data/all_data.csv')
        # 分割liqu和pres数据并保存
        point = self.select_liqu_pres_point(all_data)
        self.liqu_data = all_data[:point, :]
        self.pres_data = all_data[point:, :]
        self.save_csv_file(self.liqu_data, 'data/liqu_data.csv')
        self.save_csv_file(self.pres_data, 'data/pres_data.csv')
        pass

    def build_sample(self):
        self.x_liqu = self.liqu_data[:, [0, 2]]
        self.y_liqu = self.liqu_data[:, [2]]  # 定液选压为label
        self.x_pres = self.pres_data[:, [0, 1]]
        self.y_pres = self.pres_data[:, [0, 1]]  # 定压选油和液为label

    def norm(self):

        # print(np.max(self.x_liqu))
        # print(np.min(self.x_liqu))
        liqu_scaler = MinMaxScaler((0, 1))
        pres_scaler = MinMaxScaler((0, 1))
        self.x_liqu = liqu_scaler.fit_transform(self.x_liqu)
        self.x_pres = pres_scaler.fit_transform(self.x_pres)
        joblib.dump(liqu_scaler, 'tool/liqu_scaler.pkl')
        joblib.dump(pres_scaler, 'tool/pres_scaler.pkl')
        pass

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


    def split_train_test(self):
        self.train_x_liqu = self.x_liqu[:int(len(self.x_liqu) * self.train_test_ratio)]
        self.train_y_liqu = self.y_liqu[:int(len(self.x_liqu) * self.train_test_ratio)]
        self.test_x_liqu = self.x_liqu[int(len(self.x_liqu) * self.train_test_ratio):]
        self.test_y_liqu = self.y_liqu[int(len(self.x_liqu) * self.train_test_ratio):]

        self.train_x_pres = self.x_pres[:int(len(self.x_pres) * self.train_test_ratio)]
        self.train_y_pres = self.y_pres[:int(len(self.x_pres) * self.train_test_ratio)]
        self.test_x_pres = self.x_pres[int(len(self.x_pres) * self.train_test_ratio):]
        self.test_y_pres = self.y_pres[int(len(self.x_pres) * self.train_test_ratio):]

    def get_data_loader(self, x, y, config):
        dataset = TensorDatasets(x, y)
        # print(dataset[0:3])
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=config.is_shuffle, drop_last=True)


class TensorDatasets(Dataset):        # 继承Dataset父类
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # np.random.shuffle(self.data)        # 按第一维度打乱数据

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):    # 按照索引获取值，重写为对单个句子的数据进行格式化处理
        # self.x = torch.from_numpy(self.x).to('cuda')
        # self.y = torch.from_numpy(self.y).to('cuda')
        return self.x[index], self.y[index]



if __name__ == '__main__':
    data = Data(40, 85)
    train_x_liqu = data.train_x_liqu
    train_y_liqu = data.train_y_liqu
    test_x_liqu = data.test_x_liqu
    test_y_liqu = data.test_y_liqu

    print(len(train_x_liqu))
    print(len(test_x_liqu))
    print(train_x_liqu[-1])
