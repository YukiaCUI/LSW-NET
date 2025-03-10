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


def polar_to_cartesian(radius_tensor):
    """Convert polar coordinates (radius) to Cartesian coordinates."""
    B, N = radius_tensor.shape
    angles = torch.linspace(0, 2 * torch.pi, N, device=radius_tensor.device).repeat(B, 1)  # Shape: (B, N)
    x = radius_tensor * torch.cos(angles)
    y = radius_tensor * torch.sin(angles)
    return torch.stack((x, y), dim=-1)  # Shape: (B, N, 2)

def compute_curvature(radius_tensor, r):
    # Convert polar coordinates to Cartesian
    cartesian_points = polar_to_cartesian(radius_tensor)  # Shape: (B, N, 2)
    B, N, _ = cartesian_points.shape

    # Step 1: Compute pairwise distances
    diff = cartesian_points.unsqueeze(2) - cartesian_points.unsqueeze(1)  # Shape: (B, N, N, 2)
    pairwise_distances = torch.norm(diff, dim=-1)  # Shape: (B, N, N)

    # Step 2: Find points within radius r
    mask = pairwise_distances <= r  # Shape: (B, N, N)
    mask &= ~torch.eye(N, device=cartesian_points.device, dtype=torch.bool).unsqueeze(0)  # Exclude self-points

    # Count neighbors and compute centroid
    neighbors_count = mask.sum(dim=2, keepdim=True)  # Shape: (B, N, 1)
    neighbors_count = torch.clamp(neighbors_count, min=1)  # Avoid division by zero

    # Compute centroid
    centroid = (diff * mask.unsqueeze(-1)).sum(dim=2) / neighbors_count  # Shape: (B, N, 2)

    # Step 3: Compute curvature via quadratic fitting
    diff_centered = diff - centroid.unsqueeze(2)  # Centered difference (B, N, N, 2)
    diff_centered *= mask.unsqueeze(-1)  # Exclude invalid neighbors

    x, y = diff_centered[..., 0], diff_centered[..., 1]
    A = torch.stack([x**2, x * y, y**2], dim=-1)  # Quadratic terms (B, N, N, 3)
    B_fit = torch.ones_like(x)  # Right-hand side (B, N, N)

    # Solve least squares for each point
    curvature = torch.zeros(B, N, device=cartesian_points.device)
    for b in range(B):  # Batch loop (cannot be fully vectorized)
        for n in range(N):  # Point loop
            valid = mask[b, n]  # Valid neighbors for this point
            if valid.sum() < 3:  # Skip if not enough neighbors
                continue
            A_valid = A[b, n, valid]
            B_valid = B_fit[b, n, valid]
            solution = torch.linalg.lstsq(A_valid, B_valid.unsqueeze(-1))
            # 获取解
            coeffs = solution.solution  # 最小二乘解
            a, _, c = coeffs.squeeze(1)  # Extract coefficients
            curvature[b, n] = 2 * (a + c)  # Compute curvature

    return curvature


    
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
        a = 5
        b = 1e-1
        c = 1e-5
        d = 1e10    
        # add_noise = PointCloudAugmentation()
        # x_pos, x_neg1, x_neg2, x_neg3 = add_noise(x)
        # 创建SampleGenerator实例
        x_pos = generate_positive_samples(x)
        x_neg1 = generate_negative_samples(x)
        xp = x - x_pos
        xn1 = x - x_neg1
        # xn2 = x - x_neg2
        # xn3 = x - x_neg3


        r = 0.5
        x_curve = compute_curvature(x, r)
        print("x_curve: ", x_curve)
        # x_normal = compute_normals(x, r)
        # xp_normal = compute_normals(x_pos, r)
        # xn1_normal = compute_normals(x_neg1, r)

        # print("x_normal: ", x_normal)
        # print("xp_normal: ", xp_normal)
        # print("xn1_normal: ", xn1_normal)

        # xp = torch.norm(x_normal - xp_normal, dim=-1)
        # xn1 = torch.norm(x_normal - xn1_normal, dim=-1)

        # print("xp: ", xp)
        # print("xn1: ", xn1)

        # print("x_normal.shape: ", x_normal.shape)

        D = attn.size()[0] * attn.size()[1]

        self.loss_pos = (attn * xp ** 2).mean()
        self.loss_neg1 = (attn * xn1 ** 2).mean()

        # print("loss_pos: ", self.loss_pos)
        # print("loss_neg1: ", self.loss_neg1)

        # self.loss_pos = torch.sqrt((attn * xp ** 2).mean()) * a
        # self.loss_neg1 = torch.sqrt((attn * xn1 ** 2).mean()) 
        # loss_neg2 = (attn * xn2 ** 2).mean()
        # loss_neg3 = (attn * xn3 ** 2).mean()
        # loss = loss_pos - loss_neg1 -loss_neg2-loss_neg3  
        # loss_contractive = torch.log(torch.exp(loss_pos/self.T)/(torch.exp(loss_neg1/self.T) + torch.exp(loss_neg2/self.T) + torch.exp(loss_neg3/self.T)))
        self.loss_reglex =  ((attn.sum() - self.gamma * D) ** 2) / D * c
        self.loss_tem = ((attn[1::2] - attn[::2]) ** 2).mean() * d
        # self.loss = self.loss_pos - self.loss_neg1 + self.loss_reglex + self.loss_tem 
        self.loss_con = self.loss_pos - self.loss_neg1
        epsilon = 1e-6
        # self.loss = self.loss_con + self.loss_reglex / ((self.loss_reglex/(self.loss_con + epsilon)).detach()) + self.loss_tem / ((self.loss_tem/(self.loss_con + epsilon)).detach())
        self.loss = self.loss_con + self.loss_reglex + self.loss_tem
        return self.loss 





def generate_positive_samples(x):
    B, N = x.shape
    half_N = N // 2  # 将 N 分为两半

    device = x.device  # 获取输入张量的设备

    # 1. 第一半数据：位姿微调
    displacement = torch.randn(B, half_N, device=device) * 0.1  # 小扰动
    positive_samples_1 = x[:, :half_N] + displacement  # 对前半部分进行扰动

    # 2. 第二半数据：添加障碍物
    positive_samples_2 = x[:, half_N:].clone()  # 复制后半部分
    num_replacements = max(1, int(0.1 * half_N))  # 确保替换至少一个片段

    for i in range(B):
        indices = torch.randint(0, half_N, (num_replacements,), device=device)  # 随机选择替换的索引
        random_points = torch.randn(num_replacements, device=device) * 0.5  # 随机生成噪声
        positive_samples_2[i, indices] = random_points

    # 合并正样本
    positive_samples = torch.cat((positive_samples_1, positive_samples_2), dim=1)
    return positive_samples

def generate_negative_samples(x):
    B, N = x.shape
    # half_N = N // 2  # 将 N 分为两半

    device = x.device  # 获取输入张量的设备

    # # 1. 随机选择已有样本的索引
    # negative_indices = torch.randint(0, B, (B,), device=device)  # 随机选择 B 个样本的索引
    # negative_samples = torch.empty(B, N, device=device)  # 创建空的负样本张量

    # for i in range(B):
    #     # 随机选择从 x 中提取的一部分作为负样本
    #     left = torch.randint(0, N - half_N + 1, (1,), device=device).item()  # 随机生成切片的左端点
    #     right = left + half_N  # 右端点
        
    #     # 复制当前样本的部分数据作为负样本
    #     negative_samples[i, :half_N] = x[negative_indices[i], left:right]
        
    #     # 2. 为负样本的后半部分添加一些扰动，模拟复杂环境
    #     noise = torch.randn(half_N, device=device) * 3.0  # 生成噪声
    #     negative_samples[i, half_N:] = x[negative_indices[i], half_N:] + noise

    negative_samples = torch.stack([row[torch.randperm(N)] for row in x]) 
    negative_samples = negative_samples.to(device)  

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