# -*- coding = utf-8 -*-
# @Time : 2024/10/10 下午4:20
# @Author : 李兆堃
# @File : model.py
# @Software : PyCharm
from torch import nn


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=cfg.input_size, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
                            dropout=cfg.dropout, batch_first=True).double()
        self.relu = nn.ReLU().double()
        self.fc = nn.Linear(cfg.hidden_size, cfg.output_size).double()

    def forward(self, x):
        # print(x)
        out, _ = self.lstm(x)
        out = self.relu(out)
        out = self.fc(out)
        return out[:, -1, :]
