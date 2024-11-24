import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter
import random
from tqdm import tqdm

def generate_random_indices_v2(tensor):
    B, T, N = tensor.shape
    device = tensor.device

    # 原始索引矩阵
    original_indices = torch.arange(N, device=device).unsqueeze(0).expand(B * T, N)

    # 生成随机偏移量
    offset_min = N // 4
    offset_max = 3 * N // 4
    random_offsets = torch.randint(offset_min, offset_max, (1, N), device=device)  # 只沿 N 维生成随机偏移

    # 扩展到 (B * T, N)，确保 B 和 T 不变
    random_offsets = random_offsets.expand(B * T, N)

    # 计算新索引，使用模运算
    new_indices = (original_indices + random_offsets) % N

    # 恢复形状为 (B, T, N)
    new_indices = new_indices.view(B, T, N)

    # 打印调试信息
    # print("new_indices.shape:", new_indices.shape)
    # print("new_indices (示例):", new_indices[0, 0])  # 示例打印首批次数据
    return new_indices
    
class MultiLoss(nn.Module):
    def __init__(self, alpha = 1, beta = 1, lambda_reg=0.001):
        super(MultiLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_reg = lambda_reg
        self.loss_spatem = 0
        self.loss_consist = 0
        self.l2_reg_loss = 0
        self.loss = 0
    
    def get_score(self, curve, tmse):
        # curve shape: (B, N)
        # tmse shape: (B, N)
        curve_score = torch.zeros_like(curve)
        curve_score[curve >= 1] = 1.0
        mask = curve < 1  # 小于 1 的位置
        curve_clipped = curve * mask  # 仅保留小于 1 的值，其他值为 0
        curve_max = torch.max(curve_clipped, dim=1, keepdim=True)[0]
        curve_max[curve_max == 0] = 1.0  # 避免某些行全为大于等于 1 的情况
        curve_max = curve_max.expand_as(curve)  # 广播 curve_max 为 (B, N)
        curve_score[mask] = curve[mask] / curve_max[mask]

        tmse = torch.sqrt(tmse)
        tmse_score = torch.zeros_like(tmse)
        tmse_score[tmse >= 1] = 1.0
        mask = tmse < 1  # 小于 1 的位置
        tmse_clipped = tmse * mask  # 仅保留小于 1 的值，其他值为 0
        tmse_max = torch.max(tmse_clipped, dim=1, keepdim=True)[0]
        tmse_max[tmse_max == 0] = 1.0  # 避免某些行全为大于等于 1 的情况
        tmse_max = tmse_max.expand_as(tmse)  # 广播 tmse_max 为 (B, N)
        tmse_score[mask] = tmse[mask] / tmse_max[mask]

        # 计算每个批次中的 N 个值的和
        curve_sum = curve_score.sum(dim=1, keepdim=True)  # Shape: (B, 1)
        tmse_sum = tmse_score.sum(dim=1, keepdim=True)    # Shape: (B, 1)

        # 动态调整 alpha，使得 curve 和 tmse 的和在每个批次上相等
        alpha = tmse_sum / (curve_sum + tmse_sum + 1e-8)  # 避免除零

        # 最终得分
        score = curve_score * alpha + tmse_score * (1 - alpha)
        
        return score

    def forward(self, features, weights, curve, tmse):
        scores = self.get_score(curve, tmse) 
        # print("scores.shape: ", scores.shape)
        weight = weights[:, 2, :].squeeze()
        
        # 加上 features1 和 features2 

        B, T, N = features.shape
        Ai = features[:, 2, :].squeeze()
        Bi = (torch.sum(features, axis=1)-Ai) / (T-1)
        # print(f"Bi.shape: {Bi.shape}")
        indices = generate_random_indices_v2(features)
        # 使用 gather 获取新的特征
        new_features = torch.gather(features, dim=2, index=indices) #dim=2  N维度
        Aj = new_features[:, 2, :].squeeze()
        # shuffled_features = shuffle_n_dim_with_min_distance(features)
     
        # Aj = shuffled_features[:, 2, :]
        # if torch.isnan(Ai).any() or torch.isnan(Bi).any():
        #   print("NaN detected in Ai or Bi")
        # if torch.isinf(Ai).any() or torch.isinf(Bi).any():
        #     print("Inf detected in Ai or Bi")
        dpos = torch.abs(Ai - Bi)
        dneg = torch.abs(Ai - Aj)

        mse_loss = nn.MSELoss()
        self.loss_spatem = mse_loss(weight, scores) * self.beta
        # print("loss_spatem: ",self.loss_spatem.shape)
        self.loss_consist = ((dpos - dneg) * weight).mean() * self.alpha
        # print("loss_consist: ",self.loss_consist.shape)
        self.loss = self.loss_spatem  + self.loss_consist
        
        # L2 Regularization on features
        # Compute the L2 norm of the features tensor (excluding batch dimension)
        # features_flattened = features.view(-1, N)  # Flatten the tensor (B*T, N) for norm calculation
        # self.l2_reg_loss = torch.norm(features_flattened, p=2)  # L2 regularization (sum of squares)
        # print("l2_reg_loss: ",self.l2_reg_loss.shape)
        # Apply regularization
        # self.loss += self.lambda_reg * self.l2_reg_loss # Add L2 regularization
        
        return self.loss
         


