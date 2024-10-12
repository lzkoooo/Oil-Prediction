# -*- coding = utf-8 -*-
# @Time : 2024/10/10 下午4:49
# @Author : 李兆堃
# @File : train.py
# @Software : PyCharm
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
    save_min_loss = 9999

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
        # 保存模型
        if ave_loss_epoch < save_min_loss:  # 只保存最小loss模型
            save_model(cfg, net, epoch, ave_loss_epoch, fr'D:/Git Hub Repositories/Oil Prediction/model/{cfg.mode}/min_loss_model.pth')
            save_min_loss = ave_loss_epoch
    save_model(cfg, net, model_path=fr'D:/Git Hub Repositories/Oil Prediction/model/{cfg.mode}/end_model.pth')    # 最后保存一次
    pass


if __name__ == '__main__':
    mode = 'pres_oil'
    cons_liqu = 40
    cons_pres = 85

    search_min_loss = 9999

    num_epoch = 30
    batch_size = [1, 2, 4]
    hidden_size = [16, 32]
    learn_rate = []

    cfg = Config(mode, cons_liqu, cons_pres, 3, 1, num_epoch=num_epoch, batch_size=batch_size, hidden_size=hidden_size, learn_rate=learn_rate)
    search_min_loss = train(cfg, search_min_loss)
