import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial

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
import moco.builder
import moco.loader
import moco.optimizer
import random

# 定义高斯模糊和翻转的增强方法
class PointCloudAugmentation(object):
    def __init__(self):
        pass

    def add_full_random_noise_simultaneously(self, x):
        # 获取输入的形状 (B, T, C, P)
        B, T, C, P = x.shape
        
        # 生成所有维度的联合索引
        idx_B = torch.randperm(B)
        idx_T = torch.randperm(T)
        idx_C = torch.randperm(C)
        idx_P = torch.randperm(P)
        
        # 使用 meshgrid 来生成联合索引
        B_idx, T_idx, C_idx, P_idx = torch.meshgrid(idx_B, idx_T, idx_C, idx_P, indexing='ij')
        
        # 将四维的索引展开为 (B*T*C*P) 的一维索引
        combined_idx = torch.stack([B_idx.flatten(), T_idx.flatten(), C_idx.flatten(), P_idx.flatten()], dim=-1)
        
        # 重组数据
        x_shuffled = x[combined_idx[:, 0], combined_idx[:, 1], combined_idx[:, 2], combined_idx[:, 3]]

        x_shuffled = x_shuffled.view(B, T, C, P)
        
        return x_shuffled



    def __call__(self, x):
        # 计算总的元素数量
        total_elements = x.numel()

        # 随机选择 10% 的元素进行加噪
        num_noisy_elements = int(total_elements * 0.1)

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
    def __init__(self, T = 1.0):
        super(AttnLoss, self).__init__()
        self.T = T

    def forward(self, x, attn):
        if not x.requires_grad:
            x.requires_grad = True
    
        if not attn.requires_grad:
            attn.requires_grad = True
        
        add_noise = PointCloudAugmentation()
        x_pos, x_neg1, x_neg2, x_neg3 = add_noise(x)
        xp = x - x_pos
        xn1 = x - x_neg1
        xn2 = x - x_neg2
        xn3 = x - x_neg3
        
        loss_pos = (attn * xp ** 2).mean()
        loss_neg1 = (attn * xn1 ** 2).mean()
        loss_neg2 = (attn * xn2 ** 2).mean()
        loss_neg3 = (attn * xn3 ** 2).mean()

        loss = -torch.log(torch.exp(loss_pos/self.T)/(torch.exp(loss_neg1/self.T) + torch.exp(loss_neg2/self.T) + torch.exp(loss_neg3/self.T)))
        return loss 