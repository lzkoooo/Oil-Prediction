# -*- coding = utf-8 -*-
# @Time : 2024/10/10 下午4:49
# @Author : 李兆堃
# @File : train.py
# @Software : PyCharm
import numpy as np
import torch
from torch import nn, optim

from configs import Config
from data import Data
from model import Model


def data_init(cfg):
    np.random.seed(2)
    torch.manual_seed(5)
    torch.cuda.manual_seed_all(8)

    cons_liqu = 40
    cons_pres = 85

    data = Data(cfg, cons_liqu, cons_pres, False)
    train_x_liqu = data.train_x_liqu
    train_y_liqu = data.train_y_liqu
    train_liqu_dataloader = data.get_data_loader(train_x_liqu, train_y_liqu, cfg)

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


def save_model(cfg, net, epoch=None, loss=None, model_path='./model/last_model.pth'):
    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model': net,
        'config': cfg
    }
    torch.save(checkpoint, model_path)


def train(cfg):
    train_liqu_dataloader = data_init(cfg)
    net, optimizer, scheduler, loss_func = train_init(cfg)

    for epoch in range(cfg.num_epoch):
        ave_loss_epoch = 0.0
        iter_num = 0
        net.train()

        for i, batch in enumerate(train_liqu_dataloader):
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
        # 保存模型
        if ave_loss_epoch < 1:
            save_model(cfg, net, epoch, ave_loss_epoch, fr'model/model_{ave_loss_epoch}.pth')
    save_model(cfg, net)
    pass
