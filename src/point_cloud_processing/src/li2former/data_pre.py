import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import numpy as np
from detector import Detector
from config.config import Config
from data import DataScan
from tqdm import tqdm
import os

# 自定义点云数据集
class PointCloudDataset(Dataset):
    def __init__(self, point_cloud_data, T):
        self.point_cloud_data = point_cloud_data
        self.T = T  # 时序长度（T帧）

    def __len__(self):
        return len(self.point_cloud_data) // self.T

    def __getitem__(self, idx):
        start_idx = idx * self.T
        return self.point_cloud_data[start_idx:start_idx + self.T]


def save_to_numpy(ranges_list, file_name):
    np_array = np.array(ranges_list, dtype=np.float32)
    np.save(file_name, np_array)

# 自定义算法模块，将P变成h*w
class CustomAlgorithm(nn.Module):
    def __init__(self, P, H, W):
        super(CustomAlgorithm, self).__init__()
        self.fc = nn.Linear(P, H * W)

    def forward(self, x):
        # x形状为(B, T, P)
        B, T, P = x.shape
        x = x.view(B * T, P)
        x = self.fc(x)  # 将P映射到H*W
        x = x.view(B, T, H, W)  # 调整为(B, T, H, W)
        return x

# 生成点云数据的Dataloader
def get_dataloader(point_cloud_data, batch_size, T):
    dataset = PointCloudDataset(point_cloud_data, T)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 模型执行流程
def process_point_cloud_data(point_cloud_data, batch_size, T):
    # 创建Dataloader
    dataloader = get_dataloader(point_cloud_data, batch_size, T)
    print(dataloader.shape)
    # # 定义CNN Encoder 和 自定义算法
    # cnn_encoder = CNNEncoder(input_size=L, output_size=P)
    # custom_algorithm = CustomAlgorithm(P=P, H=H, W=W)

    for batch_data in dataloader:
        batch_data = torch.tensor(batch_data, dtype=torch.float32)  # (B, T, L)
        
        # # CNN 处理，补齐成 (B, T, P)
        # batch_data = cnn_encoder(batch_data)

        # # 自定义算法处理，变成 (B, T, H, W)
        # batch_data = custom_algorithm(batch_data)

        # 最终结果
        print(batch_data.shape)

# 创建 Config 类的实例
config = Config()

# 获取当前工作目录
cwd = os.getcwd()

# 拼接相对路径
config_path = '/media/cyj/DATA/Self_Feature_LO/src/point_cloud_processing/src/cfgs/ros_li2former.yaml'
npy_path = '/media/cyj/DATA/Self_Feature_LO/src/point_cloud_processing/data/dianxin6_pre.npy'


# 加载 YAML 配置文件
config.load(config_path)
# 初始化 DataScan 类
cutouter = DataScan(config)
data_path = config("FILE_PATH")
# 遍历 N 维度，对每个点应用 scans2cutout 函数
cutouts = []
point_cloud_data = np.load(data_path) 
for i in tqdm(range(point_cloud_data.shape[0]), desc="Processing scans"):
    scan = point_cloud_data[i, :]
    cutouts.append(cutouter(scan))
    
save_to_numpy(cutouts, npy_path)
print(cutouts.shape)

# 超参数设置
B = 4  # batch size
T = 10  # 每批次T帧
# L = n  # 输入点数
# P =   # 目标补齐的点数
# H, W = 8, 8  # 最终H*W尺寸

# 执行处理
process_point_cloud_data(point_cloud_data, B, T)