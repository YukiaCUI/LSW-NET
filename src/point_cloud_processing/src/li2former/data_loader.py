import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import numpy as np
from detector import Detector
from config.config import Config
from cutouts import DataScan
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, TensorDataset

# 自定义点云数据集
class Dataloader(object):
    def __init__(self, config):        

  
        self.config = config

        # 拼接相对路径
        # config_path = '/media/cyj/DATA/Self_Feature_LO/src/point_cloud_processing/src/cfgs/ros_li2former.yaml'
        # npy_path = '/media/cyj/DATA/Self_Feature_LO/src/point_cloud_processing/data/dianxin6_pre.npy'
    def get_batch(self):

        # 初始化 DataScan 类
        cutouter = DataScan(self.config)
        data_path = self.config("FILE_PATH")
        # 遍历 N 维度，对每个点应用 scans2cutout 函数
        cutouts = []
        point_cloud_data = np.load(data_path) 
        
        #TODO: 仅使用前 100 个点
        point_cloud_data = point_cloud_data[:100]
        
        for i in tqdm(range(point_cloud_data.shape[0]), desc="Processing scans"):
            scan = point_cloud_data[i, :]
            cutout = cutouter(scan)
            if i >= 5:
                # 假设你已经有了 data，形状为 [1800, 5, 64]
                cutouts.append(cutout)
                
            # print(cutout.shape)
            
            # 从 i = 5 开始进行 append 操作
        
        # 创建一个 TensorDataset
        cutouts = torch.stack(cutouts, dim=0) 
        dataset = TensorDataset(cutouts)

        # print(config)
        #  读取 'TRAINER' 字段
        trainer_config = self.config.sub("TRAINER")

        # 获取 batch size
        batch_size = trainer_config("KWARGS", {}).get("BATCH_SIZE") 

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

