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

class ContrastLoss(nn.Module):
    def __init__(self, pos_margin=0.1, neg_margin=1.0):
        super(ContrastLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.desc_loss = 0
        self.det_loss = 0
        self.loss = 0

    def getscore(self, x):
        B, N, C = x.size()
        neighbor_radius = 5
        score_alpha = torch.zeros_like(x)

        # 为每个点 (i, j) 计算 score_alpha
        for i in range(N):
            # 确保邻居索引的有效范围
            st = max(0, i - neighbor_radius)
            en = min(N, i + neighbor_radius + 1)  # +1 因为切片不包括上限
            neighbor_values = x[:, st:en, :]  # 取出邻居值，形状 (B, 符合条件的邻居数, C)
            
            # 计算当前点的值
            Dij = x[:, i, :]  

            # 计算邻居的和，排除当前点
            neighbor_sum = neighbor_values.sum(dim=1) - Dij  # squeeze(1) 后的形状是 (B, C)

            # 在这里确保 neighbor_sum 是有效的
            neighbor_count = neighbor_values.size(1) - 1  # 不包括当前点
            if neighbor_count > 0:
                average_neighbor_sum = neighbor_sum / neighbor_count  # 确保除以邻居数量
            else:
                average_neighbor_sum = torch.zeros_like(Dij)  # 如果没有邻居，返回零

        
            # print(Dij.size(), average_neighbor_sum.size())

            # 计算 score_alpha
            score_alpha[:, i, :] = torch.log1p(torch.exp(Dij - average_neighbor_sum))

        max_D = x.max(dim=2).values  # max_D 的形状是 (B, N)
        score_beta = x / max_D.unsqueeze(2)  # 扩展 max_D 的形状以便广播

        score_product = score_alpha * score_beta  # 计算 score_product
        score = score_product.max(dim=2).values  # 在 C 维度上取最大值

        return score



    def forward(self, x, y):
        B, N, C = x.size()
        M =  N//3
        st = torch.randint(0, N - M + 1, (1,)).item()
        radius = M//2

        # score (B, N)
        score_x = self.getscore(x)
        score_y = self.getscore(y)
       
        for i in range(st, st+M):
            Ai = x[:, i, :]
            Bi = y[:, i, :]
            # 计算 d_pos, shape(B, 1)
            d_pos = torch.norm(Ai - Bi, p=2, dim=1)

            # 计算 d_neg
            # 有效索引范围
            neg_indices = []
            if i - radius > 0:
                neg_indices.extend(range(0, i - radius))  # [0, i-radius)
            if i + radius < N:
                neg_indices.extend(range(i + radius, N))  # [i+radius, N)

            # 计算与所有符合条件的 Bj 的距离
            d_neg_values = []
            for j in neg_indices:
                Bj = y[:, j, :]  # (B, C)
                # 计算 Ai 和 Bj 之间的距离, shape(B,)
                distance = torch.norm(Ai - Bj, p=2, dim=1)  # shape: (B,)
                d_neg_values.append(distance)
            
            # 如果没有找到符合条件的 Bj，则 d_neg 设置为无穷大
            if d_neg_values:
                # 将 d_neg_values 转换为 tensor，并找到每个样本的最小距离
                d_neg_values_tensor = torch.stack(d_neg_values, dim=1)  # shape: (B, N')
                d_neg = torch.min(d_neg_values_tensor, dim=1)[0]  # shape: (B,)
            else:
                d_neg = torch.full((B, 1), float('inf'))  

            # 计算 self.desc_loss 和 self.det_loss
            self.desc_loss += (torch.relu(d_pos - self.pos_margin) + torch.relu(self.neg_margin - d_neg)).mean()
            self.det_loss += ((d_pos - d_neg)*(score_x[:, i] + score_y[:, i])).mean()

        self.desc_loss /= M
        self.det_loss /= M

        self.loss = self.desc_loss + self.det_loss


        return self.loss
