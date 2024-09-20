#!/usr/bin/env python3.9
# coding = utf-8

import rospy
from sensor_msgs.msg import LaserScan
import sys
sys.path.append("/home/tjark/carto_oringin/src/point_cloud_processing/src") #添加路径，使得后面能够引用该目录下的模块
from config import *       #导入配置文件，包含了训练过程中的各种参数
import torch               #用于构建神经网络，torch是PyTorch的核心库，提供了张量操作、自动求导等功能
import torch.nn as nn      #torch.mm是PyTorch的神经网络模块，提供了各种神经网络层和损失函数的实现
import torch.nn.functional as F
import numpy as np


class Ctrans(nn.Module):
    def __init__(self, hidden_size, kernel_size=3):
        super(Ctrans, self).__init__()

        # 卷积1
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU()
        )

        num_heads = 4
        # 添加Transformer模块
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads),
            num_layers=6  # 增加 Transformer 的层数
        )

         # 卷积2
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        # 卷积3
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        # 卷积4
        self.conv4 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2)
        )
         # 卷积5
        self.conv5 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        # 卷积6
        self.conv6 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, kernel_size=kernel_size, padding=kernel_size//2)
        )

        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        
        X = X.reshape((X.shape[0], 1, X.shape[1]))

        output = self.conv1(X)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = output.permute(2, 0, 1)
        output = self.transformer(output)
        output = output.permute(1, 2, 0)
        output = self.conv6(output)
        output = self.sigmoid(output)
        output = output.reshape((X.shape[0], -1))
        
        return output

class TCN1(nn.Module):
    def __init__(self, hidden_size, kernel_size=3):
        super(TCN1, self).__init__()
        # 卷积1
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        # 卷积2
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        # 卷积3
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        # 卷积4
        self.conv4 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, kernel_size=kernel_size, padding=kernel_size//2)
        )
        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = X.reshape((X.shape[0], 1, X.shape[1]))           #reshape输入数据的格式，使其符合卷积层的要求
        output = self.conv1(X)                                      # 第一层卷积
        output = self.conv2(output)                                 # 第二层卷积
        output = self.conv3(output)                                 # 第三层卷积
        output = self.conv4(output)                                 # 第四层卷积
        output = self.sigmoid(output)                               #sigmoid函数
        output = output.reshape((X.shape[0], -1))                   # 将输出的格式再次reshape，使其符合最终输出的要求
        return output


class TCN2(nn.Module):
    def __init__(self, hidden_size, kernel_size=3):
        super(TCN2, self).__init__()
        # 卷积1
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        # 卷积2
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=2 * (kernel_size//2), dilation=2),
            nn.ReLU()
        )
        # 卷积3
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=4 * (kernel_size//2), dilation=4),
            nn.ReLU()
        )
        # 卷积4
        self.conv4 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, kernel_size=kernel_size, padding=8 * (kernel_size//2), dilation=8)
        )
        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        output = self.conv1(X)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.sigmoid(output)
        output = output.reshape((X.shape[0], -1))
        return output

class TCN3(nn.Module):
    def __init__(self, hidden_size, kernel_size=3):
        super(TCN3, self).__init__()
        # 卷积+下采样
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # 卷积+下采样
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # 反卷积+上采样
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=kernel_size+1, padding=kernel_size//2, stride=2),
            nn.ReLU()
        )
        # 反卷积+上采样
        self.conv4 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ConvTranspose1d(hidden_size, 1, kernel_size=kernel_size+1, padding=kernel_size//2, stride=2)
        )
        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        output = self.conv1(X)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.sigmoid(output)
        output = output.reshape((X.shape[0], -1))
        return output



def get_sequence(X):
    X = np.array(X)  # 原始激光数据
    X[X > 35.0] = 35.0
    # X[:-1] = X[:-1] - X[1:]  # 特征工程
    # X[-1] = X[ -2]
    # X[:-1] = X[:-1] - X[1:]
    # X[-1] = X[-2]
    return  X

def laser_scan_callback(msg):
    model_name, hidden_size, kernel_size = ["Ctrans", 64, 7]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # network = torch.load('/home/tjark/carto_oringin/src/point_cloud_processing/model/%s_hidden_%d_kernel_%d.pkl' % (model_name, hidden_size, kernel_size),map_location=device)
    network = torch.load('/home/tjark/carto_oringin/src/point_cloud_processing/model/%s_0512hidden_%d_kernel_%d.pkl' % (model_name, hidden_size, kernel_size),map_location=device)
    network.conv1 = network.conv1.to(device)
    network.conv2 = network.conv2.to(device)
    network.conv3 = network.conv3.to(device)
    network.conv4 = network.conv4.to(device)
    network.sigmoid = network.sigmoid.to(device)
    X = msg.ranges
    X = get_sequence(X)
    X = torch.tensor(X, dtype=torch.float32).to(device)
    attn = network(X.reshape((1, -1)))
    attn = attn.squeeze()  # 去除维度为1的维度
    attn_list = attn.tolist()  # 转换为Python列表
    msg.intensities = tuple(attn_list)  # 将list转换为tuple赋值给msg.intensities
    # msg.intensities = [1] * len(msg.intensities)
    # msg.header.frame_id = "base_link"
    pub = rospy.Publisher('processed_laser_scan', LaserScan, queue_size=100)
    pub.publish(msg)
    # rospy.loginfo(msg)

    
if __name__ == "__main__":
    rospy.init_node('scan_processed', anonymous=True)
    # sub = rospy.Subscriber('filtered_base_scan', LaserScan, laser_scan_callback)
    sub = rospy.Subscriber('/scan', LaserScan, laser_scan_callback)
    rospy.loginfo("processed_laser_scan published!")
    rospy.spin()
    











    
