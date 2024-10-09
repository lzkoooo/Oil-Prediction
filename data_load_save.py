import numpy as np
from torch.utils.data import Dataset, DataLoader


class Datasets(Dataset):  # 继承Dataset父类
    def __init__(self, X, Y):
        self.data = X
        self.label = Y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):  # 按照索引获取值
        return self.data[index], self.label[index]


def data_loader(X, Y, config):
    dataset = Datasets(X, Y)  # 实例化数据集
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=config.is_shuffle)


def save_all_data(data):
    x_liqu_train, y_liqu_train, x_liqu_test, y_liqu_test, x_pres_train, y_pres_train, x_pres_test, y_pres_test = data.deq_data
    np.savez('data/data.npz', x_liqu_train=x_liqu_train, y_liqu_train=y_liqu_train, x_liqu_test=x_liqu_test, y_liqu_test=y_liqu_test, x_pres_train=x_pres_train, y_pres_train=y_pres_train, x_pres_test=x_pres_test, y_pres_test=y_pres_test)
    pass


def load_all_data():
    data = np.load('data/data.npz')
    x_liqu_train = data['x_liqu_train']
    y_liqu_train = data['y_liqu_train']
    x_liqu_test = data['x_liqu_test']
    y_liqu_test = data['y_liqu_test']
    x_pres_train = data['x_pres_train']
    y_pres_train = data['y_pres_train']
    x_pres_test = data['x_pres_test']
    y_pres_test = data['y_pres_test']
    return x_liqu_train, y_liqu_train, x_liqu_test, y_liqu_test, x_pres_train, y_pres_train, x_pres_test, y_pres_test
