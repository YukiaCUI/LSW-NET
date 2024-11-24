# coding=utf-8 
import sys
sys.path.append("/share/home/tj90055/dhj/Self_Feature_LO/src/point_cloud_processing/src")
from data import ScanData
from loss import AttnLoss
from config import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm

class LSWNet(nn.Module):
    def __init__(self, hidden_size, kernel_size=3):
        super(LSWNet, self).__init__()
        
        # Pool first
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        # # Encoder Layers
        # self.conv_encoder1 = nn.Sequential(
        #     nn.BatchNorm1d(1),
        #     nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
        #     nn.LeakyReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2)
        # )

        self.conv1= nn.Sequential(
            nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.conv2= nn.Sequential(
            nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.ConvTranspose1d(hidden_size, 1, kernel_size=2, stride=2),
            nn.LeakyReLU()
        )
        self.linear_layer = nn.Linear(in_features=1, out_features=1, bias=True)

        self.sigmoid = nn.Sigmoid()

        self.t_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=1, nhead=1, batch_first=True),
            num_layers=1
        )

        

    
    def forward(self, x):

        # Input (B, T, N, C)
        # B, T, N, _ = x.size()
        # x = x.permute(0, 2, 3, 1).contiguous()  # 将形状从 (B, T, N, C) 改为 (B, N, C, T)
        # x = x.view(B * N, -1, T)  # 将形状从 (B, N, C, T) 改为 (B * N, C, T)
        B, T, N, C = x.size()
        x = x.view(B * T, C, N)  # 合并 B 和 T
        # x = self.avg_pool(x).view(B, N, -1).permute(0, 2, 1)
        # x = x.squeeze() # 将形状从 (B, T, N, 1) 改为 (B, T, N)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print("x0 shape:",x.shape)
        #x格式是(B,T,8N)
        x = x.view(B, T, -1, 1)
        x = self.sigmoid(x)
        feature = x.squeeze()
        x = self.linear_layer(x)
        x = self.sigmoid(x)
        weight = x.squeeze()

        return feature, weight