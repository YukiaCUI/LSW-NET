import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import os
import time
import math
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
class PositionalEncodingSine(nn.Module):
    def __init__(self, d_model, max_len=511):  # 默认 max_len 设为 511
        super(PositionalEncodingSine, self).__init__()

        self.max_len = max_len  # max_len 初始化为 511

        # 初始化位置编码矩阵
        pe = torch.zeros(self.max_len, d_model)

        # 生成位置索引：[0, 1, 2, ..., max_len-1]
        position = torch.arange(0, self.max_len).unsqueeze(1)

        # 计算每个位置的 div_term (与 d_model 相关)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # 奇数维度使用正弦，偶数维度使用余弦
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 扩展维度，变为 (1, max_len, d_model)
        pe = pe.unsqueeze(0)    # [1, max_len, d_model]

        # 注册为 buffer，这样它就不会被视为需要训练的参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)

        # 补充最后一个位置的编码，以保证位置编码长度与输入序列长度一致
        if seq_len > self.max_len:
            # 如果输入的序列长度超过最大位置编码长度，手动添加额外的编码
            extra_pos_encoding = self.pe[:, -1:, :]  # 取最后一个位置的编码
            pos_encoding = torch.cat([self.pe[:, :self.max_len, :], extra_pos_encoding.expand(1, seq_len - self.max_len, -1)], dim=1)
        else:
            pos_encoding = self.pe[:, :seq_len, :]

        return x + pos_encoding

class EncoderOnly(nn.Module):
    def __init__(self, hidden_size, kernel_size=3, seq_len=1024):
        super(EncoderOnly, self).__init__()
        
        # Encoder Layers
        self.conv_encoder1 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
                # position encoding
        self.pe_sine1 = PositionalEncodingSine(d_model=hidden_size, max_len=seq_len//2)

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

        self.pe_sine2 = PositionalEncodingSine(d_model=hidden_size, max_len=seq_len//4)

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

        self.pe_sine3 = PositionalEncodingSine(d_model=hidden_size, max_len=seq_len//8)

        self.t_encoder3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=1
        )

        self.pe_sineT = PositionalEncodingSine(d_model=hidden_size, max_len=5)
        
        self.T_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=3
        )

        # Decoder Layers
        self.T_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=3
        )

        self.conv_decoder3 = nn.Sequential(
            nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.t_decoder3 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=1
        )

        self.conv_decoder2 = nn.Sequential(
            nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.t_decoder2 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=1
        )

        self.conv_decoder1 = nn.Sequential(
            nn.ConvTranspose1d(hidden_size, 1, kernel_size=2, stride=2),
            nn.LeakyReLU()
        )
        self.t_decoder1 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=1
        )

        
    
    def forward(self, x):
        # Input (B, T, N, 1)
        B, T, N, _ = x.size()
        mask = torch.rand(B, T, N, 1) < 0.3  # 创建掩码，30% 的数据被选中
        mask = mask.to(x.device)
        x = x * mask.float()  # 应用掩码，设置为0
        # Encoder
        x = x.view(B * T, 1, N)  # Reshape for Conv1d
        
        x1 = self.conv_encoder1(x)
        x1 = x1.transpose(1, 2)
        # print("x1.size",x1.size())
        # x1 shape: B*T, N/2, 128
        x1 = self.pe_sine1(x1)

        x1 = self.t_encoder1(x1)
        x1 = x1.transpose(1, 2)

        x2 = self.conv_encoder2(x1)
        x2 = x2.transpose(1, 2)
        x2 = self.pe_sine2(x2)
        x2 = self.t_encoder2(x2)
        x2 = x2.transpose(1, 2)

        x3 = self.conv_encoder3(x2)
        x3 = x3.transpose(1, 2)

        x3 = self.pe_sine3(x3)

        x3 = self.t_encoder3(x3)
        x3 = x3.transpose(1, 2)

        x = x3.view(B, T, -1, x3.size(-1))
        x = x.permute(0, 2, 1, 3).contiguous().view(B * N // 8, T, -1)
        x = self.pe_sineT(x)
        x_encoder = self.T_encoder(x)
        return x_encoder

