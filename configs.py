
import torch


class Config:
    def __init__(self, constant_liquid, constant_pressure, mem_days, pre_days, hidden_size, num_layers, dropout, num_epoches=50):
        self.mem_days = mem_days
        self.pre_days = pre_days
        self.constant_liquid = constant_liquid
        self.constant_pressure = constant_pressure
        self.input_size = 3  # 输入维度
        self.output_size = 2

        self.hidden_size = hidden_size  # 隐藏层数量节点
        self.num_layers = num_layers  # 网络层数
        self.dropout = dropout  # 在训练过程的前向传播中，让每个神经元以一定概率p处于不激活的状态。以达到减少过拟合的效果
        self.batch_size = 32
        self.is_shuffle = False
        self.learn_rate = 0.01  # 学习率,0.01~0.001为宜
        self.num_epoches = num_epoches  # 每一次epoch代表遍历一遍数据集
        self.step_size = 1
        self.gamma = 0.99
        self.weight_decay = 0.00000001
        self.devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
