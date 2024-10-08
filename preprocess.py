import pandas as pd
from collections import deque


class Data:
    def __init__(self, df, mem_days, pre_days, constant_liquid, constant_pressure):
        # self.path = path
        self.df = df
        self.mem_days = mem_days
        self.pre_days = pre_days
        self.data = None
        self.cons_liqu = constant_liquid
        self.cons_pres = constant_pressure

        self.condition_liqu = (self.df['日产液 [sm3/d]'] > (self.cons_liqu - 0.001))
        self.condition_pres = (self.df['井底压力[bar]'] == self.cons_pres)

    # def _read_file(self):
    #     self.df = pd.read_excel(self.path)
    @property
    def deq_data(self):
        self._preprocess()
        self._build_data()
        # self._normalization()
        deq_liqu = self._build_deq(self.df_cons_liqu)
        deq_pres = self._build_deq(self.df_cons_pres)
        return deq_liqu, deq_pres
        pass

    def _preprocess(self):
        # 删除时间列
        self.df.drop(columns=['时间'], inplace=True)
        self.df = self.df.apply(pd.to_numeric, errors='coerce')
        # self.df['时间'] = pd.to_datetime(self.df['时间'])
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)
        # 删除第一行
        # self.df.drop(index=0, inplace=True)
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
        # 处理定液数据
        self.df_cons_liqu['日产液 [sm3/d]'] = self.df_cons_liqu['日产液 [sm3/d]'] / self.cons_liqu  # 恒为1
        self.df_cons_liqu['日产油 [sm3/d]'] = self.df_cons_liqu['日产油 [sm3/d]'] / self.cons_liqu  # 趋向于0
        self.df_cons_liqu['井底压力[bar]'] = self.cons_pres / self.df_cons_liqu['井底压力[bar]']  # 从0趋向于1,最后达到1
        self.df_cons_liqu['label_oil'] = self.df_cons_liqu['label_oil'] / self.cons_liqu  # 趋向于0
        self.df_cons_liqu['label_pressure'] = self.cons_pres / self.df_cons_liqu['label_pressure']  # 从0趋向于1,最后达到1
        # 处理定压数据
        self.df_cons_pres['日产液 [sm3/d]'] = self.df_cons_pres['日产液 [sm3/d]'] / self.cons_liqu  # 从1趋向于0
        self.df_cons_pres['日产油 [sm3/d]'] = self.df_cons_pres['日产油 [sm3/d]'] / self.cons_liqu  # 趋向于0
        self.df_cons_pres['井底压力[bar]'] = self.cons_pres / self.df_cons_pres['井底压力[bar]']  # 恒为1
        self.df_cons_pres['label_oil'] = self.df_cons_pres['label_oil'] / self.cons_liqu  # 趋向于0
        self.df_cons_pres['label_liquid'] = self.df_cons_pres['label_liquid'] / self.cons_liqu  # 恒为1
        pass

    def _build_deq(self, df):
        x_deq = []
        y_deq = []
        deq = deque(maxlen=self.mem_days)
        for i in range(len(df)):
            deq.append(df.iloc[i].to_numpy())
            if len(deq) == self.mem_days:
                y_deq.append(deq[0][3:])  # label
                x_deq.append(list(deq))
        return x_deq[:-(self.mem_days + self.pre_days - 1)]  # 去掉最后几组数据，因为最后几组数据没有label


if __name__ == '__main__':
    df = pd.read_excel(r'D:\Git Hub Repositories\Oil Prediction\Y3557井生产特征10.3.xlsx')
    mem_days = 5
    pre_days = 1
    constant_liquid = 40
    constant_pressure = 85
    data = Data(df, mem_days, pre_days, constant_liquid, constant_pressure)
    x_liqu, x_pres = data.deq_data
