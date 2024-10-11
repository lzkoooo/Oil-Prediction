# -*- coding = utf-8 -*-
# @Time : 2024/10/10 下午8:15
# @Author : 李兆堃
# @File : configs.py
# @Software : PyCharm

import torch


class Config:
    def __init__(self, mode, mem_days=5, pre_days=2, num_epoch=50, batch_size=32, is_shuffle=True, num_layers=2, hidden_size=32, learn_rate=0.005, dropout=0.3, step_size=1, gamma=0.99):
        self.mem_days = mem_days
        self.pre_days = pre_days
        self.mode = mode    # 分为'liqu_oil', 'liqu_pres', 'pres_oil', 'pres_liqu'四种模式

        self.input_size = 2
        self.output_size = 1

        self.num_epoch = num_epoch      # 每一次epoch代表遍历一遍数据集
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle

        self.num_layers = num_layers     # 网络层数
        self.hidden_size = hidden_size      # 隐藏层数量节点
        self.learn_rate = learn_rate     # 学习率,0.01~0.001为宜
        self.dropout = dropout      # 在训练过程的前向传播中，让每个神经元以一定概率p处于不激活的状态。以达到减少过拟合的效果

        self.step_size = step_size
        self.gamma = gamma

        self.devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


