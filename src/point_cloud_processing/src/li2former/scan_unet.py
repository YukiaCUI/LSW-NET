import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from detector import Detector
from config.config import Config
from data_loader import DataGet 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 定义 U-Net 模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 编码器
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 解码器
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),  # 用于与跳跃连接融合后的通道数
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)
        )

    def forward(self, x):
        # 编码阶段
        x1 = self.encoder1(x)  # 输出: (B, 16, H/2, W/2)
        x2 = self.encoder2(x1) # 输出: (B, 32, H/4, W/4)

        # 解码阶段，跳跃连接在此处
        d1 = self.decoder1(x2)  # 输出: (B, 16, H/2, W/2)
        d1 = torch.cat([d1, x1], dim=1)  # 跳跃连接: 将编码器 x1 的输出与解码器 d1 的输出在通道维度上拼接
        
        # 最后一步解码
        out = self.decoder2(d1)  # 输出: (B, 1, H, W)

        # print("x1.shape: ", x1.shape)
        # print("x2.shape: ", x2.shape)
        # print("d1.shape: ", d1.shape)
        # print("out.shape: ", out.shape)


        return out

# 自定义数据集
class PointCloudDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = sample.reshape(1, sample.shape[0], sample.shape[1])  # 变换为 (1, H, W)
        if self.transform:
            sample = self.transform(sample)
        return sample, sample  # 返回输入和目标相同


config_path = "/media/cyj/DATA/Self_Feature_LO/src/point_cloud_processing/src/cfgs/ros_li2former.yaml"

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

# 初始化 U-Net 模型
model = UNet()

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, scans in enumerate(dataloader):
        B,C,T,P = scans[0].shape
        input = scans[0].permute(0,2,1,3).reshape(-1,C,P)
        input = input.unsqueeze(1)
        
        optimizer.zero_grad()  # 清空梯度
        outputs = model(input)  # 前向传播
        loss = criterion(outputs, input)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss += loss.item()

    # 记录每个 epoch 的平均损失到 TensorBoard
    avg_loss = running_loss / len(dataloader)
    writer.add_scalar('Training Loss', avg_loss, epoch)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# 关闭 TensorBoard 记录器
writer.close()

# 保存模型
torch.save(model.state_dict(), 'unet_pretrained_encoder.pth')
print("模型已保存！")
