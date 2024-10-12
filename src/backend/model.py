# -*- coding = utf-8 -*-
# @Time : 2024/10/10 下午4:20
# @Author : 李兆堃
# @File : model.py
# @Software : PyCharm
from torch import nn


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(cfg.input_size * cfg.mem_days * cfg.batch_size, cfg.hidden_size).double()
        self.fc2 = nn.Linear(cfg.hidden_size, cfg.hidden_size).double()
        self.fc3 = nn.Linear(cfg.hidden_size, cfg.output_size).double()
        self.relu = nn.ReLU().double()

    def forward(self, x):
        x = x.flatten()
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
