# -*- coding = utf-8 -*-
# @Time : 2024/10/10 下午4:49
# @Author : 李兆堃
# @File : train.py
# @Software : PyCharm
import os

import numpy as np
import torch
from torch import nn, optim

from backend.configs import Config
from backend.data import Data
from backend.model import Model


def data_init(cfg):
    np.random.seed(2)
    torch.manual_seed(5)
    torch.cuda.manual_seed_all(8)

    data = Data(cfg, norm_open=True)
    train_x, train_y = data.get_train_data()
    train_liqu_dataloader = data.get_data_loader(train_x, train_y, cfg)

    return train_liqu_dataloader
    pass


def train_init(cfg):
    net = Model(cfg)
    net.to(cfg.devices)

    loss_func = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=cfg.learn_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)  # 学习率调整
    return net, optimizer, scheduler, loss_func
    pass


def save_model(config, net, epoch=None, loss=None, model_path=None):
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model': net,
        'config': config
    }
    torch.save(checkpoint, model_path)


def train(cfg):
    train_dataloader = data_init(cfg)
    net, optimizer, scheduler, loss_func = train_init(cfg)

    for epoch in range(cfg.num_epoch):
        ave_loss_epoch = 0.0
        iter_num = 0
        net.train()

        for i, batch in enumerate(train_dataloader):
            x_batch, y_batch = batch
            x_batch = torch.as_tensor(x_batch).to(cfg.devices)
            y_batch = torch.as_tensor(y_batch).to(cfg.devices)

            optimizer.zero_grad()  # 梯度置为0，不置为0的话会与上一次batch的数据相关
            pred = net.forward(x_batch)  # 对data进行前向推理，得到预测

            loss = loss_func(pred, y_batch)  # 计算loss
            # print(f'pred:{pred.item()}   label:{y_batch.item()}')
            loss.backward()  # 进行反向传播得到每个参数的梯度值
            optimizer.step()  # 通过梯度下降对参数进行更新

            ave_loss_epoch += loss.item()
            iter_num += 1

        ave_loss_epoch /= iter_num
        print(f"\nEpoch: {epoch}, Loss: {ave_loss_epoch}")
        scheduler.step()

        if os.path.exists(fr'D:/Git Hub Repositories/Oil Prediction/model/{mode}/min_loss_model.pth'):
            model_path = fr'D:/Git Hub Repositories/Oil Prediction/model/{mode}/min_loss_model.pth'
            current_min_loss_model_loss = torch.load(model_path)['loss']
        else:
            current_min_loss_model_loss = 9999

        # 保存模型
        if ave_loss_epoch < current_min_loss_model_loss:  # 与当前最小loss模型对比
            save_model(cfg, net, epoch, ave_loss_epoch, fr'D:/Git Hub Repositories/Oil Prediction/model/{cfg.mode}/min_loss_model.pth')
    save_model(cfg, net, model_path=fr'D:/Git Hub Repositories/Oil Prediction/model/{cfg.mode}/end_model.pth')    # 最后保存一次

    pass


if __name__ == '__main__':
    mode_list = ['liqu_pres']
    cons_liqu = 40
    cons_pres = 85

    num_epoch = 40
    # mem_days_list = [3, 5, 10]
    # batch_size_list = [1, 2, 4]
    # hidden_size_list = [16, 32]
    # num_layers_list = [1, 2]
    # learn_rate_list = [0.01, 0.03, 0.06]

    mem_days_list = [3]
    batch_size_list = [1]
    hidden_size_list = [16, 32]
    num_layers_list = [1, 2, 4]
    learn_rate_list = [0.01, 0.03, 0.06]

    for mem_days in mem_days_list:
        for mode in mode_list:
            for batch_size in batch_size_list:
                for hidden_size in hidden_size_list:
                    for num_layers in num_layers_list:
                        for learn_rate in learn_rate_list:
                            print(f'mode:{mode}  batch_size:{batch_size}  hidden_size:{hidden_size}  num_layers:{num_layers}  learn_rate:{learn_rate}')
                            cfg = Config(mode, cons_liqu, cons_pres, mem_days, 1, num_epoch=num_epoch, batch_size=batch_size, num_layers=num_layers, hidden_size=hidden_size, learn_rate=learn_rate)
                            train(cfg)
