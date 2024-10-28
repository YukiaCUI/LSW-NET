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


class LSEncoder(nn.Module):
    def __init__(self, hidden_size, kernel_size=3):
        super(LSEncoder, self).__init__()
    
        # Encoder Layers
        self.conv_encoder1 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
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
        self.t_encoder3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True),
            num_layers=1
        )
        
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

        # Encoder
        x = x.view(B * T, 1, N)  # Reshape for Conv1d
        x1 = self.conv_encoder1(x)
        x1 = x1.transpose(1, 2)
        x1 = self.t_encoder1(x1)
        x1 = x1.transpose(1, 2)

        x2 = self.conv_encoder2(x1)
        x2 = x2.transpose(1, 2)
        x2 = self.t_encoder2(x2)
        x2 = x2.transpose(1, 2)

        x3 = self.conv_encoder3(x2)
        x3 = x3.transpose(1, 2)
        x3 = self.t_encoder3(x3)
        x3 = x3.transpose(1, 2)

        x = x3.view(B, T, -1, x3.size(-1))
        x = x.permute(0, 2, 1, 3).contiguous().view(B * N // 8, T, -1)
        x_encoder = self.T_encoder(x)

        # Decoder
        x = self.T_decoder(x_encoder, x_encoder)
        x = x.view(B, N // 8, T, -1).permute(0, 2, 1, 3).contiguous().view(B * T, -1, x.size(-1))

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
    num_epochs = 100
    
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
    
    # 实例化模型
    model = LSEncoder(hidden_size=hidden_size, kernel_size=kernel_size)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 创建 TensorBoard 日志文件夹
    log_dir = os.path.join("logs", time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

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
        "t_encoder1": model.t_encoder1.state_dict(),
        "conv_encoder2": model.conv_encoder2.state_dict(),
        "t_encoder2": model.t_encoder2.state_dict(),
        "conv_encoder3": model.conv_encoder3.state_dict(),
        "t_encoder3": model.t_encoder3.state_dict(),
        "T_encoder": model.T_encoder.state_dict(),
    }

    torch.save(encoder_state_dict, "encoder_params.pth")
    print("Encoder parameters saved successfully.")

    # 关闭 TensorBoard writer
    writer.close()
        
if __name__ == "__main__":
    data_path = "/share/home/tj90055/dhj/Self_Feature_LO/dianxin6.npy"
    batch_size = 64
    train(data_path, batch_size)
