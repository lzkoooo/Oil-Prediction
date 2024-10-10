import joblib
import numpy as np
import pandas as pd
from collections import deque

from sklearn.preprocessing import StandardScaler


class Data:
    def __init__(self, df, mem_days, pre_days, constant_liquid, constant_pressure, train_test_ratio=0.8):
        # self.path = path
        self.df = df
        self.mem_days = mem_days
        self.pre_days = pre_days
        self.data = None
        self.cons_liqu = constant_liquid
        self.cons_pres = constant_pressure
        self.train_test_ratio = train_test_ratio

        self.condition_liqu = (self.df['日产液 [sm3/d]'] > (self.cons_liqu - 0.001))
        self.condition_pres = (self.df['井底压力[bar]'] == self.cons_pres)

    @property
    def deq_data(self):
        self._preprocess()
        self._build_data()
        # self._normalization()
        deq_liqu, y_liqu = self._build_deq(self.df_cons_liqu)
        deq_pres, y_pres = self._build_deq(self.df_cons_pres)
        # 分割训练集和测试集
        x_liqu_train, x_liqu_test = self._split_data(deq_liqu)
        y_liqu_train, y_liqu_test = self._split_data(y_liqu)
        x_pres_train, x_pres_test = self._split_data(deq_pres)
        y_pres_train, y_pres_test = self._split_data(y_pres)
        return x_liqu_train, y_liqu_train, x_liqu_test, y_liqu_test, x_pres_train, y_pres_train, x_pres_test, y_pres_test
        pass

    def _preprocess(self):
        # 删除时间列
        self.df.drop(columns=['时间'], inplace=True)
        self.df = self.df.apply(pd.to_numeric, errors='coerce')
        # self.df['时间'] = pd.to_datetime(self.df['时间'])
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)
        # 删除第一行
        self.df.shift(-1)
        # 重置索引
        self.df.reset_index(drop=True, inplace=True)

    def _build_data(self):
        # 构建label列
        self.df['label_oil'] = self.df['日产油 [sm3/d]'].shift(-(self.mem_days + self.pre_days - 1))
        self.df['label_liquid'] = self.df['日产液 [sm3/d]'].shift(-(self.mem_days + self.pre_days - 1))
        self.df['label_pressure'] = self.df['井底压力[bar]'].shift(-(self.mem_days + self.pre_days - 1))

        # 分割数据
        self.df_cons_liqu = self.df[self.condition_liqu].copy()
        self.df_cons_pres = self.df[self.condition_pres].copy()
        # 删除各自不需要的label
        self.df_cons_liqu.drop(columns=['label_liquid'], inplace=True)
        self.df_cons_pres.drop(columns=['label_pressure'], inplace=True)

    def _normalization(self):
        # 归一化
        scaler_liqu = StandardScaler()
        self.df_cons_liqu = self.df_cons_liqu.to_numpy()
        x_liqu = self.df_cons_liqu[:, :3]
        label_liqu = self.df_cons_liqu[:, 3:]
        x_liqu = scaler_liqu.fit_transform(x_liqu)     # 只对x归一化
        joblib.dump(scaler_liqu, 'scaler_liqu.pkl')

        self.df_cons_liqu = np.concatenate((x_liqu, label_liqu), axis=1)

        scaler_pres = StandardScaler()
        self.df_cons_pres = self.df_cons_pres.to_numpy()
        x_pres = self.df_cons_pres[:, :3]
        label_pres = self.df_cons_pres[:, 3:]
        x_pres = scaler_liqu.fit_transform(x_pres)  # 只对x归一化
        joblib.dump(scaler_pres, 'scaler_pres.pkl')

        self.df_cons_liqu = np.concatenate((x_pres, label_pres), axis=1)
        # # 处理定液数据
        # self.df_cons_liqu['日产液 [sm3/d]'] = self.df_cons_liqu['日产液 [sm3/d]'] / self.cons_liqu  # 恒为1
        # self.df_cons_liqu['日产油 [sm3/d]'] = self.df_cons_liqu['日产油 [sm3/d]'] / self.cons_liqu  # 趋向于0
        # self.df_cons_liqu['井底压力[bar]'] = self.cons_pres / self.df_cons_liqu['井底压力[bar]']  # 从0趋向于1,最后达到1
        # self.df_cons_liqu['label_oil'] = self.df_cons_liqu['label_oil'] / self.cons_liqu  # 趋向于0
        # self.df_cons_liqu['label_pressure'] = self.cons_pres / self.df_cons_liqu['label_pressure']  # 从0趋向于1,最后达到1
        # # 处理定压数据
        # self.df_cons_pres['日产液 [sm3/d]'] = self.df_cons_pres['日产液 [sm3/d]'] / self.cons_liqu  # 从1趋向于0
        # self.df_cons_pres['日产油 [sm3/d]'] = self.df_cons_pres['日产油 [sm3/d]'] / self.cons_liqu  # 趋向于0
        # self.df_cons_pres['井底压力[bar]'] = self.cons_pres / self.df_cons_pres['井底压力[bar]']  # 恒为1
        # self.df_cons_pres['label_oil'] = self.df_cons_pres['label_oil'] / self.cons_liqu  # 趋向于0
        # self.df_cons_pres['label_liquid'] = self.df_cons_pres['label_liquid'] / self.cons_liqu  # 恒为1
        pass

    def _build_deq(self, df):
        x_deq = []
        y_deq = []
        deq = deque(maxlen=self.mem_days)
        for item in range(len(df)):
            item = df.iloc[item].to_numpy()
            # item = df[item]
            deq.append(item[:3])  # 前3列为x
            y_deq.append(item[3:])  # 后2列为y
            if len(deq) == self.mem_days:
                x_deq.append(list(deq))
        if self.mem_days == 1:
            return np.array(x_deq), np.array(y_deq)
        else:
            return np.array(x_deq), np.array(y_deq[:-(self.mem_days - 1)])  # y去掉最后几组数据，因为最后一组数据对应的是倒数第5天的数据，且pre_days移动过了，所以去掉最后mem_days - 1组数据

    def _split_data(self, data):
        point = int(len(data) * self.train_test_ratio)
        train = data[:point]
        test = data[point:]

        return train, test

if __name__ == '__main__':
    df = pd.read_excel(r'D:\Git Hub Repositories\Oil Prediction\Y3557井生产特征10.3.xlsx')
    mem_days = 5
    pre_days = 1
    constant_liquid = 40
    constant_pressure = 85
    data = Data(df, mem_days, pre_days, constant_liquid, constant_pressure)
    x_liqu, y_liqu, x_pres, y_pres = data.deq_data
