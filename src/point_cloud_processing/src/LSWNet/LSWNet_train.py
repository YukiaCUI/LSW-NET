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

from attnloss import AttnLoss
from LSWEncoderOnly import EncoderOnly
from LSWNet import LSWNet

class PointCloudSequenceDataset(Dataset):
    def __init__(self, data, T):
        self.data = data
        self.T = T
        self.padding = T // 2

    def __len__(self):
        # 总长度为可以提取的序列数
        return len(self.data) - 2 * self.padding

    def __getitem__(self, idx):
        # 计算索引范围，中心帧前后各 padding 帧
        start = idx
        end = idx + self.T
        # 从原始数据中提取形状为 (T, N) 的子序列
        sequence = self.data[start:end]
        return torch.tensor(sequence, dtype=torch.float32)

def train(data_path, batch_size):
    
    hidden_size = 128
    kernel_size = 7
    learning_rate = 0.001
    num_epochs = 200
    
    # 加载数据
    point_cloud_data = np.load(data_path) 
    n, N = point_cloud_data.shape

    # 定义 T 的长度
    T = 5
    assert T % 2 == 1, "T 必须是奇数，以便能对称地选择前后帧"
    padding = T // 2  # 对称 padding，用于从每帧提取前后相邻帧
    valid_frames = n - 2 * padding  # 可用帧数量

    # 创建 Dataset 实例
    dataset = PointCloudSequenceDataset(point_cloud_data, T=T)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 实例化并加载保存的编码器权重
    encoder_only_model = EncoderOnly(hidden_size=hidden_size, kernel_size=kernel_size)
    encoder_params_path = "/share/home/tj90055/dhj/Self_Feature_LO/src/point_cloud_processing/model/LSencoder/encoder_params.pth"
    encoder_state_dict = torch.load(encoder_params_path, map_location=device)

    # 加载参数到模型
    encoder_only_model.conv_encoder1.load_state_dict(encoder_state_dict["conv_encoder1"])
    encoder_only_model.t_encoder1.load_state_dict(encoder_state_dict["t_encoder1"])
    encoder_only_model.conv_encoder2.load_state_dict(encoder_state_dict["conv_encoder2"])
    encoder_only_model.t_encoder2.load_state_dict(encoder_state_dict["t_encoder2"])
    encoder_only_model.conv_encoder3.load_state_dict(encoder_state_dict["conv_encoder3"])
    encoder_only_model.t_encoder3.load_state_dict(encoder_state_dict["t_encoder3"])
    encoder_only_model.T_encoder.load_state_dict(encoder_state_dict["T_encoder"])
    encoder_only_model = encoder_only_model.to(device)
    
    # 实例化模型
    model = LSWNet(hidden_size=hidden_size, kernel_size=kernel_size)
    model = model.to(device)
    
    # 定义损失函数和优化器
    attnloss = AttnLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 创建 TensorBoard 日志文件夹
    log_dir = os.path.join("/share/home/tj90055/dhj/Self_Feature_LO/src/point_cloud_processing/src/LSWNet/logs", time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)


    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # 将数据移到设备
            batch = batch.to(device)
            
            # 调整形状为 (B, T, N, 1)
            inputs = batch.unsqueeze(-1)
            B, T, N, _ = inputs.shape

            # input使用一阶段编码器进行encoder
            encoder_only_model.eval()
            with torch.no_grad():
                encoder_output = encoder_only_model(inputs)
                # encoder_output 即为编码器的结果
            
            # 调整形状 (B * N // 8, T, -1)-->(B, T, N//8, -1)
            x_encoder = encoder_output.view(B, N//8, T, -1).permute(0, 2, 1, 3).contiguous()

            # 前向传播
            weights = model(x_encoder)
            
            # 计算损失：输入和输出的差异
            # inputs(B, T, N, 1) weight(B, N, 1)
            points = inputs[:, 2, :, :].squeeze()
            weights = weights.squeeze()
            loss = attnloss(points, weights)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            running_loss += loss.item()
            # 记录每个步骤的损失到 TensorBoard
            writer.add_scalar('Loss/train', attnloss.loss.item(), epoch * len(dataloader) + step)
            writer.add_scalar('loss_pos/train', attnloss.loss_pos.item(), epoch * len(dataloader) + step)
            writer.add_scalar('loss_neg1/train', attnloss.loss_neg1.item(), epoch * len(dataloader) + step)
            writer.add_scalar('loss_reglex/train', attnloss.loss_reglex.item(), epoch * len(dataloader) + step)
            writer.add_scalar('loss_tem/train', attnloss.loss_tem.item(), epoch * len(dataloader) + step)

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # 记录每个 epoch 的平均损失
        writer.add_scalar('Loss/train/average', avg_loss, epoch)

    # 创建模型文件夹
    model_dir = os.path.join("/share/home/tj90055/dhj/Self_Feature_LO/src/point_cloud_processing/src/LSWNet/model")
    os.makedirs(model_dir, exist_ok=True)
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    # 保存模型的文件名，添加时间戳
    model_file_path = os.path.join(model_dir, f"model_{current_time}.pth")

    # 保存模型的状态字典
    torch.save(model.state_dict(), model_file_path)
    print("Encoder parameters saved successfully.")

    # 关闭 TensorBoard writer
    writer.close()
        
if __name__ == "__main__":
    data_path = "/share/home/tj90055/dhj/Self_Feature_LO/dianxin6.npy"
    batch_size = 64
    train(data_path, batch_size)
