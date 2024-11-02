from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
import random


    
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

