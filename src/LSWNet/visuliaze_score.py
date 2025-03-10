import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from attnloss import AttnLoss
from contrastloss import ContrastLoss
from LSWEncoderOnly import EncoderOnly
from LSWNet import LSWNet

data_path = "/share/home/tj90055/dhj/Self_Feature_LO/src/point_cloud_processing/src/LSWNet/data/pre_train_data.npy"
data = np.load(data_path)
print(data.shape)
def get_image(points):
    x, y = [], []
    for k in range(len(points)):
        theta = k / 810 * np.pi
        px = - points[k] * np.cos(theta)
        py = points[k] * np.sin(theta)
        x.append(px)
        y.append(py)
    return x, y

# def get_color(attn):
#     c = []
#     for a in attn:
#         if a < 0.2:
#             c.append("#fff143")
#         elif a < 0.5:
#             c.append("#ffb61e")
#         elif a < 0.75:
#             c.append("#ff7500")
#         else:
#             c.append("r")
#     return c

def get_color(attn):
    c = []
    for a in attn:
        if a < 0.1:
            c.append("#ffff00")  # 黄色
        elif a < 0.3:
            c.append("#ffcc00")  # 深橙色
        elif a < 0.5:
            c.append("#ff9900")  # 橙色
        elif a < 0.7:
            c.append("#ff6600")  # 浅橙色
        elif a < 0.9:
            c.append("#ff3300")  # 深红色
        else:
            c.append("#ff0000")  # 红色
    return c

# def get_shape(attn):
#     s = []
#     for a in attn:
#         if a < 0.2:
#             s.append(0.3)
#         elif a < 0.5:
#             s.append(0.8)
#         elif a < 0.75:
#             s.append(1.5)
#         else:
#             s.append(2)
#     return s

def get_shape(attn):
    s = []
    for a in attn:
        if a < 0.1:
            s.append(0.8*2)  # 非常小
        elif a < 0.3:
            s.append(1.6*2)  # 较小
        elif a < 0.5:
            s.append(2.4*2)  # 小
        elif a < 0.7:
            s.append(3.2*2)  # 中等
        elif a < 0.9:
            s.append(4.0*2)  # 大
        else:
            s.append(4.8*2)  # 非常大
    return s

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 16))
# 随机选择一帧
def get_statistics(data):
    return {
        'max': np.max(data),
        'min': np.min(data),
        'mean': np.mean(data),
        'std': np.std(data),
    }

# 计算 weights_s 和 weights_t 的统计信息

points = data[:,2,:,:]
random_index = np.random.randint(0, points.shape[0])  # 随机选择 0 到 n-1 的索引
selected_frame = points[random_index:random_index + 1, : ,:]  # 取出 (1, N)
print("selected_frame.shape", selected_frame.shape)
selected_points =  np.squeeze(selected_frame[:,:,0])
print("selected_points.shape", selected_points.shape)
weights_s =  np.squeeze(selected_frame[:,:,1])
print("weights_s.shape", weights_s.shape)
weights_t =  np.squeeze(selected_frame[:,:,2])
print("weights_t.shape", weights_t.shape)
weights_s_stats = get_statistics(weights_s)
weights_t_stats = get_statistics(weights_t)
print("weights_s statistics:", weights_s_stats)
print("weights_t statistics:", weights_t_stats)
# 绘制直方图
plt.figure(figsize=(12, 6))
# 绘制 weights_s 的直方图
plt.subplot(1, 2, 1)
plt.hist(weights_s, bins=30, color='blue', alpha=0.7)
plt.title('Distribution of weights_s')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.savefig("score_s.png")

# 绘制 weights_t 的直方图
plt.subplot(1, 2, 2)
plt.hist(weights_t, bins=30, color='green', alpha=0.7)
plt.title('Distribution of weights_t')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.savefig("score_t.png")
# 分别进行归一化（除以各自的最大值）
weights_s_normalized = weights_s / np.max(weights_s)
print("weight_max: ", np.max(weights_s))
weights_t_normalized = weights_t / np.max(weights_s)
x, y = get_image(selected_points)
c = get_color(weights_s)
s = get_shape(weights_s)

plt.title("Scene" , fontsize=38)
plt.scatter(x, y, s=s, c=c)
# plt.scatter(x, y, c=c)
# 获取当前子图的坐标轴对象
ax = plt.gca()

# 设置横纵坐标轴标签的字体大小
ax.set_xlabel('', fontsize=30)
ax.set_ylabel('', fontsize=30)

# 设置横纵坐标轴刻度标签的字体大小
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(32)

plt.savefig("score.png")