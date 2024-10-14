import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 定义 U-Net 模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 输入: (B, 1, H, W) -> 输出: (B, 16, H, W)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 输出: (B, 16, H/2, W/2)

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # 输出: (B, 32, H/2, W/2)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 输出: (B, 32, H/4, W/4)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 输出: (B, 16, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),           # 输出: (B, 1, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2),    # 输出: (B, 1, H, W)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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

# 示例数据生成
num_samples = 100  # 训练样本数量
H = 64  # 点云段数
W = 1024  # 每段点云的点数
data = [np.random.rand(H, W).astype(np.float32) for _ in range(num_samples)]

# 创建数据集和数据加载器
dataset = PointCloudDataset(data)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 初始化 U-Net 模型
model = UNet()

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用均方误差损失
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in dataloader:
        optimizer.zero_grad()  # 清空梯度
        outputs = model(images)  # 前向传播
        loss = criterion(outputs, targets)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')

# 保存模型
torch.save(model.state_dict(), 'unet_pretrained_encoder.pth')
print("模型已保存！")
