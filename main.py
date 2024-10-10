import joblib
import pandas as pd
import torch
from matplotlib import pyplot as plt

from configs import Config
from data_load_save import save_all_data, load_all_data
from preprocess import Data
from train import start_train, prediction


def search_model():
    '''
        初步搜索最佳超参
    '''
    constant_liquid = 40
    constant_pressure = 85
    mem_days = [5]
    pre_days = 1
    hidden_size = [16]
    num_layers = [2]
    dropout = 0.3
    num_epoches = 100

    for mem in mem_days:
        for hid in hidden_size:
            for lay in num_layers:
                df = pd.read_excel(r'D:\Git Hub Repositories\Oil Prediction\Y3557井生产特征10.3.xlsx')
                cfg = Config(constant_liquid, constant_pressure, mem, pre_days, hid, lay, dropout, num_epoches=num_epoches)
                data = Data(df, cfg.mem_days, cfg.pre_days, cfg.constant_liquid, cfg.constant_pressure)
                save_all_data(data)
                start_train(cfg, r'models/liqu_model/')


def train_best_model():
    '''
        正式模型训练
    '''
    check_point = torch.load(r'models/best_liqu_model.pth')

    cfg = check_point["config"]
    num_epoches = 1000
    df = pd.read_excel(r'D:\Git Hub Repositories\Oil Prediction\Y3557井生产特征10.3.xlsx')

    data = Data(df, cfg.mem_days, cfg.pre_days, cfg.constant_liquid, cfg.constant_pressure)
    save_all_data(data)
    start_train(cfg, r'models/best_liqu_model.pth')


def apply_model():

    df = pd.read_excel(r'D:\Git Hub Repositories\Oil Prediction\Y3557井生产特征10.3.xlsx')
    data = Data(df, 4, 1, 40, 85, 0)
    save_all_data(data)
    _, _, x_liqu_test, y_liqu_test, _, _, _, _ = load_all_data()

    scaler = joblib.load('scaler_liqu.pkl')
    x_liqu_test = scaler.transform(x_liqu_test)
    pred_oil, pred_pres, label_oil, label_pres = prediction(x_liqu_test, y_liqu_test)
    lenth = [i for i in range(len(pred_oil))]

    plt.subplot(2, 1, 1)
    plt.plot(lenth, pred_oil, label='pred_oil', color='black')
    plt.plot(lenth, label_oil, label='label_oil', color='red')
    plt.title('Oil Predction')
    plt.xlabel('data')
    plt.ylabel('oil')

    plt.subplot(2, 1, 2)
    plt.plot(lenth, pred_pres, label='pred_pres', color='black')
    plt.plot(lenth, label_pres, label='label_oil', color='red')
    plt.title('Pressure Predction')
    plt.xlabel('data')
    plt.ylabel('pressure')

    # plt.legend()  # 显示图例


    # 显示图形
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    search_model()
    # train_best_model()

    apply_model()
    pass

