import sys
sys.path.append("/home/shiwb/AttnSlam/src")
from TCNNet.config import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnLoss(nn.Module):
    def __init__(self,alpha=1., beta=1., gamma=0.1,T = 10):
        super(AttnLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.T = T

    def forward(self, attn, yp, yn ,yn1 ,yn2):
        D = attn.size()[0] * attn.size()[1]
        self.loss_p = (attn * yp ** 2).mean()
        self.loss_n = (attn * yn ** 2).mean()
        self.loss_n1 = (attn * yn1 ** 2).mean()
        self.loss_n2 = (attn * yn2 ** 2).mean()
        self.loss_contractive = -torch.log(torch.exp(self.loss_p/self.T)/(torch.exp(self.loss_n/self.T) + torch.exp(self.loss_n1/self.T) + torch.exp(self.loss_n2/self.T)))
        # self.loss = self.alpha * ((attn.sum() - self.gamma * D) ** 2) / D
        # self.loss4 = self.beta * ((attn[1::2] - attn[::2]) ** 2).mean()
        # self.loss = self.loss1 - self.loss2 + self.loss3 + self.loss4
        return self.loss_contractive