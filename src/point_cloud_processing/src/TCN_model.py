import sys
sys.path.append("E:/learngit/attnslam/src") #添加路径，使得后面能够引用该目录下的模块
from loss import AttnLoss  #导入AttnLoss类，用于定义损失函数
from config import *       #导入配置文件，包含了训练过程中的各种参数
import torch               #用于构建神经网络，torch是PyTorch的核心库，提供了张量操作、自动求导等功能
import torch.nn as nn      #torch.mm是PyTorch的神经网络模块，提供了各种神经网络层和损失函数的实现
import torch.nn.functional as F
import json                #用于处理json格式的数据


class TCN1(nn.Module):
    def __init__(self, hidden_size, kernel_size=3):
        super(TCN1, self).__init__()
        # 卷积1
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        # 卷积2
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        # 卷积3
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        # 卷积4
        self.conv4 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, kernel_size=kernel_size, padding=kernel_size//2)
        )
        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = X.reshape((X.shape[0], 1, X.shape[1]))           #reshape输入数据的格式，使其符合卷积层的要求
        output = self.conv1(X)                                      # 第一层卷积
        output = self.conv2(output)                                 # 第二层卷积
        output = self.conv3(output)                                 # 第三层卷积
        output = self.conv4(output)                                 # 第四层卷积
        output = self.sigmoid(output)                               #sigmoid函数
        output = output.reshape((X.shape[0], -1))                   # 将输出的格式再次reshape，使其符合最终输出的要求
        return output


class TCN2(nn.Module):
    def __init__(self, hidden_size, kernel_size=3):
        super(TCN2, self).__init__()
        # 卷积1
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        # 卷积2
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=2 * (kernel_size//2), dilation=2),
            nn.ReLU()
        )
        # 卷积3
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=4 * (kernel_size//2), dilation=4),
            nn.ReLU()
        )
        # 卷积4
        self.conv4 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, kernel_size=kernel_size, padding=8 * (kernel_size//2), dilation=8)
        )
        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        output = self.conv1(X)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.sigmoid(output)
        output = output.reshape((X.shape[0], -1))
        return output


class TCN3(nn.Module):
    def __init__(self, hidden_size, kernel_size=3):
        super(TCN3, self).__init__()
        # 卷积+下采样
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # 卷积+下采样
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # 反卷积+上采样
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=kernel_size+1, padding=kernel_size//2, stride=2),
            nn.ReLU()
        )
        # 反卷积+上采样
        self.conv4 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ConvTranspose1d(hidden_size, 1, kernel_size=kernel_size+1, padding=kernel_size//2, stride=2)
        )
        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        output = self.conv1(X)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.sigmoid(output)
        output = output.reshape((X.shape[0], -1))
        return output
