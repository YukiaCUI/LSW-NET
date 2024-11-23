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
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from attnloss import AttnLoss
from multi_loss import MultiLoss
from contrastloss import ContrastLoss
from LSWEncoderOnly import EncoderOnly
from LSWNet import LSWNet
from utils.curvature import compute_curvature_least_squares



def train(data_path, batch_size):
    
    hidden_size = 128
    kernel_size = 7
    learning_rate = 0.001
    num_epochs = 10    
    data = np.load(data_path)
    # 创建 DataLoader 进行随机采样
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 实例化并加载保存的编码器权重
    encoder_only_model = EncoderOnly(hidden_size=hidden_size, kernel_size=kernel_size)
    encoder_params_path = "/share/home/tj90055/dhj/Self_Feature_LO/src/point_cloud_processing/model/LSencoder/encoder_params003.pth"
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
    # attnloss = AttnLoss()
    # contrastloss = ContrastLoss()
    multi_loss = MultiLoss()
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
            inputs = batch[...,0].unsqueeze(-1)
            curvatures = batch[...,1]
            curvatures = curvatures[:,2,:,:].squeeze(-1)
            t_mse = batch[...,2]
            t_mse = t_mse[:,2,:,:].squeeze(-1)
            B, T, N, _ = inputs.shape

            # input使用一阶段编码器进行encoder
            encoder_only_model.eval()
            with torch.no_grad():
                encoder_output = encoder_only_model(inputs)
                # encoder_output 即为编码器的结果
            
            # 调整形状 (B * N // 8, T, -1)-->(B, T, N//8, -1)
            x_encoder = encoder_output.view(B, N//8, T, -1).permute(0, 2, 1, 3).contiguous()
            features = model(x_encoder)
            # feature_2 = model(x_encoder)
            # 计算损失：输入和输出的差异
            # inputs(B, T, N, 1) weight(B, T, N)
            # points = inputs.squeeze()

            # points shape: (B, T, N)
            # weights(B, N)
            # print(points.shape)
            # print(weights.shape)
            # loss = attnloss(points, weights)
            loss = multi_loss(features, curvatures, t_mse)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
            running_loss += loss.item()
            # 记录每个步骤的损失到 TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + step)
            writer.add_scalar('loss_spatem/train', multi_loss.loss_spatem.item(), epoch * len(dataloader) + step)
            writer.add_scalar('loss_consist/train', multi_loss.loss_consist.item(), epoch * len(dataloader) + step)
            writer.add_scalar('loss_reglex/train', multi_loss.l2_reg_loss.item(), epoch * len(dataloader) + step)

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
    data_paths = [
        "/share/home/tj90055/dhj/Self_Feature_LO/src/point_cloud_processing/src/LSWNet/data/pre_train_data.npy"]
    batch_size = 64
    train(data_paths, batch_size)



