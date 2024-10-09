import torch.nn as nn


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=cfg.input_size, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
                            batch_first=True, dropout=cfg.dropout).double()
        self.linear_layer = nn.Linear(cfg.hidden_size, cfg.output_size).double()  # 构建全连接层
        self.relu = nn.ReLU().double()

    def forward(self, x):  # x为输入数据
        out, _ = self.lstm(x)
        out = out[:, -1, :]     # 即每个batch都选择最后一个时间步的全部节点输出
        out = self.relu(out)  # 激活函数
        out = self.linear_layer(out)
        return out
