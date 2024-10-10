import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from data_load_save import data_loader, load_all_data
from model import Model
from preprocess import Data


def train_init(cfg):
    # 设置随机种子
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    x_liqu_train, y_liqu_train, _, _, _, _, _, _ = load_all_data()
    liqu_train_dataloader = data_loader(x_liqu_train, y_liqu_train, cfg)

    net = Model(cfg)
    net.to(cfg.devices)

    loss_func = nn.MSELoss()  # 损失函数
    optimizer = optim.Adam(net.parameters(), lr=cfg.learn_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)  # 学习率调整

    return liqu_train_dataloader, net, loss_func, optimizer, scheduler


def start_train(cfg, save_path):
    min_loss = 99999

    liqu_train_dataloader, net, loss_func, optimizer, scheduler = train_init(cfg)

    # 开始训练
    for epoch in range(cfg.num_epoches):
        net.train()
        loss_epoch = 0.0
        iter_num = 0
        for i, batch in enumerate(liqu_train_dataloader):  # 取出每一个batch来训练


            data, label = batch
            # print("Batch data shape:", data.shape)
            label = torch.as_tensor(label, dtype=torch.float64).to(cfg.devices)
            data = torch.as_tensor(data, dtype=torch.float64).to(cfg.devices)

            optimizer.zero_grad()  # 梯度置为0，不置为0的话会与上一次batch的数据相关
            pred = net.forward(data)  # 对data进行前向推理，得到预测
            loss = loss_func(pred, label)  # 计算loss

            loss.backward()  # 进行反向传播得到每个参数的梯度值
            optimizer.step()  # 通过梯度下降对参数进行更新

            loss_epoch += loss.item()
            iter_num += 1

            print(f"Epoch: {epoch}, Batch: {i}, loss: {loss.item()}")
        average_loss = loss_epoch / iter_num
        print(f"average_loss{average_loss}")
        # 出现更低的loss且低于1则保存
        if average_loss < min_loss and average_loss <= 5:
            min_loss = average_loss
            print(f"current_min_loss{min_loss}")
            # print(f'pred: {pred}')
            # print(f'label: {label}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': average_loss,
                'config': cfg,
                'model': net
            }, save_path + f'checkpoint_{average_loss}.pth')
        # 最后保存一次
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_loss,
            'config': cfg,
            'model': net
        }, save_path + f'model.pth')
        scheduler.step()  # 每个batch调整下学习率


def prediction(x_liqu_test, y_liqu_test):
    checkpoint = torch.load('models/best_liqu_model.pth')
    checkpoint = torch.load('models/liqu_model/model.pth')
    cfg = checkpoint['config']
    cfg.batch_size = 1
    cfg.is_shuffle = False
    net = checkpoint['model']
    net.to(cfg.devices)
    liqu_test_dataloader = data_loader(x_liqu_test, y_liqu_test, cfg)

    pred_oil_list = []
    pred_pres_list = []
    label_oil_list = []
    label_pres_list = []
    net.eval()
    with torch.no_grad():
        for i, batch in enumerate(liqu_test_dataloader):
            data, label = batch
            data = torch.as_tensor(data, dtype=torch.float64).to(cfg.devices)
            pred = net.forward(data).cpu().numpy()[0]

            label = label.cpu().numpy()[0]
            # 反归一化
            # pred_oil_list.append(pred[0] * cfg.constant_liquid)
            # pred_pres_list.append(cfg.constant_pressure / pred[1])
            # label_oil_list.append(label[0] * cfg.constant_liquid)
            # label_pres_list.append(cfg.constant_pressure / label[1])
            pred_oil_list.append(pred[0])
            pred_pres_list.append(pred[1])
            label_oil_list.append(label[0])
            label_pres_list.append(label[1])

    return pred_oil_list, pred_pres_list, label_oil_list, label_pres_list

    pass
