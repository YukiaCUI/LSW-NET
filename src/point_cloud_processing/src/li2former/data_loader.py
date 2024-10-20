import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import numpy as np
from detector import Detector
from config.config import Config
from cutouts import DataScan, attnScan  
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, TensorDataset

# 自定义点云数据集
class DataGet(object):
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
        cutouts_numpy = []
        point_cloud_data = np.load(data_path) 
        
        #TODO: 仅使用前 100 个点
        # point_cloud_data = point_cloud_data[:101]
        point_cloud_data = point_cloud_data[:14045]
        
        for i in tqdm(range(point_cloud_data.shape[0]), desc="Processing scans"):
            scan = point_cloud_data[i, :]
            cutout = cutouter(scan) # numpy
            ct = torch.from_numpy(cutout).float()
            
            
            if i >= 5:
                # 假设你已经有了 data，形状为 [1800, 5, 64]
                cutouts.append(ct)
                cutouts_numpy.append(cutout)
                
            # print(cutout.shape)
            
            # 从 i = 5 开始进行 append 操作
        
        # save cutouts_numpy as npy file
        cutouts_numpy = np.array(cutouts_numpy)
        
        np.save('/media/cyj/DATA/Self_Feature_LO/cutouts.npy', cutouts_numpy)

        # 创建一个 TensorDataset
        cutouts = torch.stack(cutouts, dim=0) 
        
        # print("cutous: ", cutouts.shape)
        dataset = TensorDataset(cutouts)

        # print(config)
        #  读取 'TRAINER' 字段
        trainer_config = self.config.sub("TRAINER")

        # 获取 batch size
        batch_size = trainer_config("KWARGS", {}).get("BATCH_SIZE") 

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # print("dataloader: ", dataloader.shape)
        return dataloader
    
    def get_attn_cutouts(self,attn):

        # 初始化 DataScan 类
        cutouter = attnScan(self.config)
        # torch to numpy
        attn = attn.cpu().detach().numpy()

        cutouts = []

        for batch in attn:
            batch = batch.flatten()
            # print("batch.shape: ", batch.shape)
            attn_cutout = cutouter(batch)
            cutouts.append(attn_cutout)
                    
        cutouts = torch.stack(cutouts, dim=0)
        return cutouts
            
        # for i in tqdm(range(point_cloud_data.shape[0]), desc="Processing scans"):
        #     scan = point_cloud_data[i, :]
        #     cutout = cutouter(scan)
        #     if i >= 5:
        #         # 假设你已经有了 data，形状为 [1800, 5, 64]
        #         cutouts.append(cutout)
                
        #     # print(cutout.shape)
            
        #     # 从 i = 5 开始进行 append 操作
        
        # # 创建一个 TensorDataset
        # cutouts = torch.stack(cutouts, dim=0) 
        
        # print("cutous: ", cutouts.shape)
        # dataset = TensorDataset(cutouts)

        # # print(config)
        # #  读取 'TRAINER' 字段
        # trainer_config = self.config.sub("TRAINER")

        # # 获取 batch size
        # batch_size = trainer_config("KWARGS", {}).get("BATCH_SIZE") 

        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # # print("dataloader: ", dataloader.shape)
        # return dataloader                                 
 
