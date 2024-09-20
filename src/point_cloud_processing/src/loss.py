import sys
sys.path.append("/home/shiwb/AttnSlam/src")
from config import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnLoss(nn.Module):
    def __init__(self, alpha=1., beta=1., gamma=0.1):
        super(AttnLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, attn, yp, yn):
        D = attn.size()[0] * attn.size()[1]
        self.loss1 = (attn * yp ** 2).mean()
        self.loss2 = (attn * yn ** 2).mean()
        self.loss3 = self.alpha * ((attn.sum() - self.gamma * D) ** 2) / D
        self.loss4 = self.beta * ((attn[1::2] - attn[::2]) ** 2).mean()
        self.loss = self.loss1 - self.loss2 + self.loss3 + self.loss4
        return self.loss