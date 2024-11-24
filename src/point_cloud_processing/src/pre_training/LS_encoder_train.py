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
import math

class PositionalEncodingSine(nn.Module):
    def __init__(self, d_model, max_len=5000):  
        super(PositionalEncodingSine, self).__init__()

        self.max_len = max_len  # max_len 初始化为 511

        # 初始化位置编码矩阵
        pe = torch.zeros(self.max_len, d_model)

        # 生成位置索引：[0, 1, 2, ..., max_len-1]
        position = torch.arange(0, self.max_len).unsqueeze(1)

        # 计算每个位置的 div_term (与 d_model 相关)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # 奇数维度使用正弦，偶数维度使用余弦
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 扩展维度，变为 (1, max_len, d_model)
        pe = pe.unsqueeze(0)    # [1, max_len, d_model]

        # 注册为 buffer，这样它就不会被视为需要训练的参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)

        # 补充最后一个位置的编码，以保证位置编码长度与输入序列长度一致
        if seq_len > self.max_len:
            # 如果输入的序列长度超过最大位置编码长度，手动添加额外的编码
            extra_pos_encoding = self.pe[:, -1:, :]  # 取最后一个位置的编码
            pos_encoding = torch.cat([self.pe[:, :self.max_len, :], extra_pos_encoding.expand(1, seq_len - self.max_len, -1)], dim=1)
        else:
            pos_encoding = self.pe[:, :seq_len, :]

        return x + pos_encoding


class LSEncoder(nn.Module):
    def __init__(self, hidden_size, kernel_size=3, seq_len=1024):
        super(LSEncoder, self).__init__()
    
        # Encoder Layers
        self.conv_encoder1 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
                # position encoding
        self.pe_sine1 = PositionalEncodingSine(d_model=hidden_size, max_len=seq_len//2)

        self.t_encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=1
        )

        self.conv_encoder2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.pe_sine2 = PositionalEncodingSine(d_model=hidden_size, max_len=seq_len//4)

        self.t_encoder2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=1
        )

        self.conv_encoder3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.pe_sine3 = PositionalEncodingSine(d_model=hidden_size, max_len=seq_len//8)

        self.t_encoder3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=1
        )

        self.pe_sineT = PositionalEncodingSine(d_model=hidden_size, max_len=5)
        
        self.T_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=3
        )

        # Decoder Layers
        self.T_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=3
        )

        self.conv_decoder3 = nn.Sequential(
            nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.t_decoder3 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=1
        )

        self.conv_decoder2 = nn.Sequential(
            nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.t_decoder2 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=1
        )

        self.conv_decoder1 = nn.Sequential(
            nn.ConvTranspose1d(hidden_size, 1, kernel_size=2, stride=2),
            nn.LeakyReLU()
        )
        self.t_decoder1 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=1
        )

        
    def forward(self, x):
        # Input (B, T, N, 1)
        B, T, N, _ = x.size()
        mask = torch.rand(B, T, N, 1) < 0.3  # 创建掩码，30% 的数据被选中
        mask = mask.to(x.device)
        x = x * mask.float()  # 应用掩码，设置为0
        # Encoder
        x = x.view(B * T, 1, N)  # Reshape for Conv1d
        
        x1 = self.conv_encoder1(x)
        x1 = x1.transpose(1, 2)
        # print("x1.size",x1.size())
        # x1 shape: B*T, N/2, 128
        x1 = self.pe_sine1(x1)

        x1 = self.t_encoder1(x1)
        x1 = x1.transpose(1, 2)

        x2 = self.conv_encoder2(x1)
        x2 = x2.transpose(1, 2)
        x2 = self.pe_sine2(x2)
        x2 = self.t_encoder2(x2)
        x2 = x2.transpose(1, 2)

        x3 = self.conv_encoder3(x2)
        x3 = x3.transpose(1, 2)

        x3 = self.pe_sine3(x3)

        x3 = self.t_encoder3(x3)
        x3 = x3.transpose(1, 2)

        x = x3.view(B, T, -1, x3.size(-1))
        x = x.permute(0, 2, 1, 3).contiguous().view(B * N // 8, T, -1)
        x = self.pe_sineT(x)
        x_encoder = self.T_encoder(x)

        # Decoder
        x = self.T_decoder(x_encoder, x_encoder)
        x = x.view(B, N // 8, T, -1).permute(0, 2, 1, 3).contiguous().view(B * T, -1, x.size(-1))
        
        x3 = x3.transpose(1, 2)
        x = x + x3  # Skip connection from encoder3
        x = x.transpose(1, 2)
        x = self.t_decoder3(x, x)
        x = x.transpose(1, 2)
        x = self.conv_decoder3(x)

        x = x + x2  # Skip connection from encoder2
        x = x.transpose(1, 2)
        x = self.t_decoder2(x, x)
        x = x.transpose(1, 2)
        x = self.conv_decoder2(x)

        x = x + x1  # Skip connection from encoder1
        x = x.transpose(1, 2)
        x = self.t_decoder1(x, x)
        x = x.transpose(1, 2)
        x = self.conv_decoder1(x)

        x = x.view(B, T, N, 1)  # Reshape back to (B, T, N, 1)

        return x


class PointCloudSequenceDataset(Dataset):
    def __init__(self, data, T):
        self.data = data
        self.T = T
        self.padding = T // 2  # 前后各取的帧数，T 必须为奇数

    def __len__(self):
        # 数据长度保持与原始数据一致
        return len(self.data)

    def __getitem__(self, idx):
        if idx < self.padding:
            # 如果索引在前两帧范围
            start = 0
            end = self.T
        elif idx >= len(self.data) - self.padding:
            # 如果索引在最后两帧范围
            start = len(self.data) - self.T
            end = len(self.data)
        else:
            # 一般情况，中间帧处理
            start = idx - self.padding
            end = idx + self.padding + 1

        # 提取子序列
        sequence = self.data[start:end]
        return torch.tensor(sequence, dtype=torch.float32)



def train(data_paths, batch_size):
    hidden_size = 128
    kernel_size = 7
    learning_rate = 0.001
    num_epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    

    # 创建 TensorBoard 日志文件夹
    log_dir = os.path.join(
        "/share/home/tj90055/dhj/Self_Feature_LO/src/point_cloud_processing/src/pre_training/logs",
        time.strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # 构建数据集
    T = 5  # 每帧取 T 个帧打包

    # 加载每个数据集并构建 PointCloudSequenceDataset
    datasets = []
    for path in data_paths:
        # 加载点云数据 (n, N)
        point_cloud_data = np.load(path)
        n, N = point_cloud_data.shape
        column_to_add = np.full((point_cloud_data.shape[0], 1), 35.0)
        point_cloud_data = np.hstack((point_cloud_data, column_to_add))
        point_cloud_data[point_cloud_data > 35] = 35.0
        dataset = PointCloudSequenceDataset(point_cloud_data, T)
        datasets.append(dataset)

    # 合并多个数据集
    combined_dataset = ConcatDataset(datasets)

    # 创建 DataLoader 进行随机采样
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    # 实例化模型
    model = LSEncoder(hidden_size=hidden_size, kernel_size=kernel_size, seq_len=N).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
    
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # 将数据移到设备
            batch = batch.to('cuda' if torch.cuda.is_available() else 'cpu')
        
            # 调整形状为 (B, T, N, 1)
            batch = batch.unsqueeze(-1)
        
            # 前向传播
            output = model(batch)
        
            # 计算损失：输入和输出的差异
            loss = criterion(output, batch)
        
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            # 记录每个步骤的损失到 TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + step)

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # 记录每个 epoch 的平均损失
        writer.add_scalar('Loss/train/average', avg_loss, epoch)


    # 保存编码器参数
    encoder_state_dict = {
        "conv_encoder1": model.conv_encoder1.state_dict(),
        "pe_sine1": model.pe_sine1.state_dict(),  # 保存位置编码1的参数
        "t_encoder1": model.t_encoder1.state_dict(),
        
        "conv_encoder2": model.conv_encoder2.state_dict(),
        "pe_sine2": model.pe_sine2.state_dict(),  # 保存位置编码2的参数
        "t_encoder2": model.t_encoder2.state_dict(),
        
        "conv_encoder3": model.conv_encoder3.state_dict(),
        "pe_sine3": model.pe_sine3.state_dict(),  # 保存位置编码3的参数
        "t_encoder3": model.t_encoder3.state_dict(),
        
        "pe_sineT": model.pe_sineT.state_dict(),  # 保存位置编码T的参数
        "T_encoder": model.T_encoder.state_dict(),
    }
    
    save_path = "/share/home/tj90055/dhj/Self_Feature_LO/src/point_cloud_processing/model/LSencoder/encoder_params1125.pth"
    torch.save(encoder_state_dict, save_path)
    print(f"Encoder parameters saved to {save_path}")

    # 关闭 TensorBoard writer
    writer.close()


if __name__ == "__main__":
    data_paths = [
        "/share/home/tj90055/dhj/Self_Feature_LO/dianxin1.npy",
        "/share/home/tj90055/dhj/Self_Feature_LO/dianxin6.npy",
        "/share/home/tj90055/dhj/Self_Feature_LO/dianxinb1.npy"
    ]
    batch_size = 64
    train(data_paths, batch_size)

        

