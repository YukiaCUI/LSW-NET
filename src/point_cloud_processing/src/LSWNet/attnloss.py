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

# 定义高斯模糊和翻转的增强方法
class PointCloudAugmentation(object):
    def __init__(self):
        pass

    def add_full_random_noise_simultaneously(self, x):
        # 获取输入的形状 (B, T, C, P)
        B, N = x.shape
        
        # 生成所有维度的联合索引
        idx_B = torch.randperm(B)
        idx_N = torch.randperm(N)
        
        # 使用 meshgrid 来生成联合索引
        B_idx, N_idx = torch.meshgrid(idx_B, idx_N, indexing='ij')
        
        # 将四维的索引展开为 (B*T*C*P) 的一维索引
        combined_idx = torch.stack([B_idx.flatten(), N_idx.flatten()], dim=-1)
        
        # 重组数据
        x_shuffled = x[combined_idx[:, 0], combined_idx[:, 1]]

        x_shuffled = x_shuffled.view(B,N)
        
        return x_shuffled



    def __call__(self, x):
        # 计算总的元素数量
        total_elements = x.numel()

        # 随机选择 5% 的元素进行加噪
        num_noisy_elements = int(total_elements * 0.05)

        # 创建一个全 False 的掩码
        mask = torch.zeros_like(x, dtype=torch.bool)

        # 展开张量为一维，生成随机的索引
        flat_mask = mask.view(-1)
        indices = torch.randperm(total_elements)[:num_noisy_elements]
        flat_mask[indices] = True

        # 将掩码变回原来的形状
        mask = flat_mask.view_as(mask)

        # 生成相同形状的噪声
        noise = torch.randn_like(x)

        # 根据掩码进行加噪
        x_pos = torch.where(mask, x + noise, x)

        x_neg1 = self.add_full_random_noise_simultaneously(x)
        x_neg2 = self.add_full_random_noise_simultaneously(x)
        x_neg3 = self.add_full_random_noise_simultaneously(x)

        return x_pos, x_neg1, x_neg2, x_neg3
    
class AttnLoss(nn.Module):
    def __init__(self, T = 1.0, alpha=0.1, beta=100., gamma=0.1):
        super(AttnLoss, self).__init__()
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loss = 0
        self.loss_pos = 0
        self.loss_neg1 = 0
        self.loss_reglex = 0
        self.loss_tem = 0

    def forward(self, x, attn):
        if not x.requires_grad:
            x.requires_grad = True
    
        if not attn.requires_grad:
            attn.requires_grad = True
        a = 1e2
        b = 1e-3
        c = 1e-5
        d = 1e7
        # add_noise = PointCloudAugmentation()
        # x_pos, x_neg1, x_neg2, x_neg3 = add_noise(x)
        # 创建SampleGenerator实例
        x_pos = generate_positive_samples(x)
        x_neg1 = generate_negative_samples(x)
        xp = x - x_pos
        xn1 = x - x_neg1
        # xn2 = x - x_neg2
        # xn3 = x - x_neg3
        D = attn.size()[0] * attn.size()[1]
        self.loss_pos = (attn * xp ** 2).mean() * a
        self.loss_neg1 = (attn * xn1 ** 2).mean() * b
        # loss_neg2 = (attn * xn2 ** 2).mean()
        # loss_neg3 = (attn * xn3 ** 2).mean()
        # loss = loss_pos - loss_neg1 -loss_neg2-loss_neg3  
        # loss_contractive = torch.log(torch.exp(loss_pos/self.T)/(torch.exp(loss_neg1/self.T) + torch.exp(loss_neg2/self.T) + torch.exp(loss_neg3/self.T)))
        self.loss_reglex =  ((attn.sum() - self.gamma * D) ** 2) / D * c
        self.loss_tem = ((attn[1::2] - attn[::2]) ** 2).mean() * d
        self.loss = self.loss_pos - self.loss_neg1 + self.loss_reglex + self.loss_tem 
        return self.loss 




def generate_positive_samples(x):
    B, N = x.shape
    half_N = N // 2  # 将 N 分为两半

    # 1. 第一半数据：位姿微调
    displacement = torch.randn(B, half_N) * 0.01  # 小扰动
    positive_samples_1 = x[:, :half_N] + displacement  # 对前半部分进行扰动

    # 2. 第二半数据：添加障碍物
    positive_samples_2 = x[:, half_N:].clone()  # 复制后半部分
    num_replacements = max(1, int(0.1 * half_N))  # 确保替换至少一个片段

    for i in range(B):
        indices = torch.randint(0, half_N, (num_replacements,))  # 随机选择替换的索引
        random_points = torch.randn(num_replacements) * 0.5  # 随机生成噪声
        positive_samples_2[i, indices] = random_points

    # 合并正样本
    positive_samples = torch.cat((positive_samples_1, positive_samples_2), dim=1)
    return positive_samples

def generate_negative_samples(x):
    B, N = x.shape
    half_N = N // 2  # 将 N 分为两半

    # 1. 第一半数据：空旷环境采样
    negative_samples_1 = torch.randn(B, half_N) * 2.0  # 随机生成与锚点和正样本差异明显的数据
    
    # 2. 第二半数据：复杂环境采样
    negative_samples_2 = torch.randn(B, half_N) * 3.0  # 另一组随机数据，确保多样性

    # 合并负样本
    negative_samples = torch.cat((negative_samples_1, negative_samples_2), dim=1)
    return negative_samples

# class AttnLoss(nn.Module):
#     def __init__(self, alpha=1., beta=1., gamma=0.1):
#         super(AttnLoss, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma

#     def forward(self, attn, yp, yn):
#         D = attn.size()[0] * attn.size()[1]
#         self.loss1 = (attn * yp ** 2).mean()
#         self.loss2 = (attn * yn ** 2).mean()
#         self.loss3 = self.alpha * ((attn.sum() - self.gamma * D) ** 2) / D
#         self.loss4 = self.beta * ((attn[1::2] - attn[::2]) ** 2).mean()
#         self.loss = self.loss1 - self.loss2 + self.loss3 + self.loss4
#         return self.loss