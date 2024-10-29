import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

class EncoderOnly(nn.Module):
    def __init__(self, hidden_size, kernel_size=3):
        super(EncoderOnly, self).__init__()
        
        # 使用原始编码器部分的结构
        self.conv_encoder1 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.t_encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=1
        )

        self.conv_encoder2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.t_encoder2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=1
        )

        self.conv_encoder3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.t_encoder3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=1
        )
        
        self.T_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=3
        )
    
    def forward(self, x):
        # 编码器部分的前向传播与原始模型相同
        B, T, N, _ = x.size()
        
        x = x.view(B * T, 1, N)
        x1 = self.conv_encoder1(x)
        x1 = x1.transpose(1, 2)
        x1 = self.t_encoder1(x1)
        x1 = x1.transpose(1, 2)
        
        x2 = self.conv_encoder2(x1)
        x2 = x2.transpose(1, 2)
        x2 = self.t_encoder2(x2)
        x2 = x2.transpose(1, 2)
        
        x3 = self.conv_encoder3(x2)
        x3 = x3.transpose(1, 2)
        x3 = self.t_encoder3(x3)
        x3 = x3.transpose(1, 2)
        
        x = x3.view(B, T, -1, x3.size(-1))
        x = x.permute(0, 2, 1, 3).contiguous().view(B * N // 8, T, -1)
        x_encoder = self.T_encoder(x)
        
        return x_encoder

