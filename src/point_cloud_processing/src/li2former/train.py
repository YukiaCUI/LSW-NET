import numpy as np
from detector import Detector
from config.config import Config
from data_loader import DataGet 
from models.li2former import Li2Former
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from attnloss_mocov3 import MoCo_AttnLoss
import attnloss_mocov3
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

import vits

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

parser = argparse.ArgumentParser(description='MoCo ImageNet Pre-Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_small',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: vit_small)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.6, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--moco-m-cos', action='store_true',
                    help='gradually increase moco momentum to 1 with a '
                         'half-cycle cosine schedule')
parser.add_argument('--moco-t', default=1.0, type=float,
                    help='softmax temperature (default: 1.0)')

# vit specific configs:
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')

# other upgrades
parser.add_argument('--optimizer', default='lars', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer used (default: lars)')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (default: 0.08)')
parser.add_argument('--add_bf', default=0, type=int,
                    help='add_bf')



def train(config_path):
    args = parser.parse_args()
    moco_m = args.moco_m
    # 创建 Config 类的实例
    config = Config()
    # 加载 YAML 配置文件
    config.load(config_path)

    # 创建 TensorBoard 的日志记录器
    writer = SummaryWriter(log_dir='logs')  # 设置日志目录

    log = []
    checkpoints = []
    
    # 初始化 DataScan 类
    BatchLoader = DataGet(config)  
    dataloader = BatchLoader.get_batch()
    model_kwargs = config.sub("MODEL")("KWARGS")
    loss_kwargs  = config.sub("LOSS")("KWARGS")
    train_kwargs = config.sub("TRAINER")("KWARGS")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Li2Former(loss_kwargs, model_kwargs).to(device) 

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = train_kwargs["MAX_EPOCHS"]
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        for batch in dataloader:
            x = batch[0].to(device)

            optimizer.zero_grad()  # 清空梯度
            attn = model.run(x)    
            attn_cutouts = BatchLoader.get_attn_cutouts(attn)
            # print("x0.shape: ", x.shape)
            # print("attn0.shape: ", attn.shape)
            print("attn_cutouts0.shape: ", attn_cutouts.shape)
            
            # x.shape = (B, C=1800, T=5, P)
            # attn,shape = (B, 1, C=1800)
            # attn_cutouts.shape = (B, C=1800, 1, P)
    
            
            B, C, T, P = x.shape
            x = x[:, :, 0, :].unsqueeze(2)
            x = x.permute(0,2,1,3)
            x = x.repeat(1, 3, 1, 1)
            B, chanel, Ct, P = x.shape
            Ct = int(Ct /2)
            x = x.reshape(B, chanel, Ct , P*2)

            attn_cutouts = attn_cutouts.permute(0, 2, 1, 3)
            attn_cutouts = attn_cutouts.repeat(1, 3, 1, 1)
            B, chanel, Ct, P = attn_cutouts.shape
            Ct = int(Ct /2)
            attn_cutouts = attn_cutouts.reshape(B, chanel, Ct , P*2)

            masks = torch.randperm(Ct)[:224]
            x = x[:, :, masks, :]
            attn_cutouts = attn_cutouts[:, :, masks, :]
            # print("x.shape: ", x.shape)
            # print("attn_cutouts.shape: ", attn_cutouts.shape)
            # TODO: 定义损失函数MocoLoss   
            AttnLoss = attnloss_mocov3.MoCo_ViT(
            partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
            args.moco_dim, args.moco_mlp_dim, args.moco_t, add_bf=args.add_bf)
            # AttnLoss = MoCo_AttnLoss(model, dim=256, mlp_dim=4096, T=1.0, arch='resnet50', add_bf=0)
            
            attn_cutouts.to(device)
            AttnLoss.cuda()

            loss = AttnLoss(x, moco_m, attn_cutouts)
            loss.backward()
            optimizer.step()  

            # # 记录损失到 TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch)
            log.append({"loss": loss.item()})

    # 保存当前检查点
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'loss': loss.item(), 
    }
    
    # 添加检查点并根据损失进行排序
    checkpoints.append(checkpoint)
    checkpoints.sort(key=lambda x: x['loss'])  # 按损失升序排序

    # 保留最好的前 max_checkpoints 个检查点
    if len(checkpoints) > 3:
        checkpoints = checkpoints[:3]

    # 保存当前 epoch 的检查点
    torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')

    # 关闭 TensorBoard 的日志记录器
    writer.close()
        

if __name__ == "__main__":
    config_path = "/media/cyj/DATA/Self_Feature_LO/src/point_cloud_processing/src/cfgs/ros_li2former.yaml"
    train(config_path)
