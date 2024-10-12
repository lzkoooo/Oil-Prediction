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
        if cfg.num_layers == 2 or cfg.num_layers == 3:
            self.fc2 = nn.Linear(cfg.hidden_size, cfg.hidden_size).double()
        else:
            self.fc2 = None
        if cfg.num_layers == 3:
            self.fc3 = nn.Linear(cfg.hidden_size, cfg.hidden_size).double()
        else:
            self.fc3 = None
        self.relu = nn.ReLU().double()
        self.fc_out = nn.Linear(cfg.hidden_size, cfg.output_size).double()

    def forward(self, x):
        x = x.flatten()
        out = self.fc1(x)

        if self.fc2 is not None:
            out = self.fc2(out)
        if self.fc3 is not None:
            out = self.fc3(out)

        out = self.relu(out)
        out = self.fc_out(out)
        return out
