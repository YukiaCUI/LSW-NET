from config.config import Config
from data_loader import DataGet 
from models.li2former import Li2Former
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from attnloss import AttnLoss
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter




def train(config_path):

    # 创建 Config 类的实例
    config = Config()
    # 加载 YAML 配置文件
    config.load(config_path)

    # 获取当前时间，并创建子文件夹
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join('logs', current_time)
    os.makedirs(log_dir, exist_ok=True)

    # 创建 checkpoint 文件夹
    checkpoint_dir = os.path.join(log_dir, 'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 创建 TensorBoard 的日志记录器
    writer = SummaryWriter(log_dir=log_dir)  # 设置日志目录

    log = []
    checkpoints = []
    
    # 初始化 DataScan 类
    BatchLoader = DataGet(config)  
    dataloader = BatchLoader.get_batch()
    model_kwargs = config.sub("MODEL")("KWARGS")
    loss_kwargs  = config.sub("LOSS")("KWARGS")
    train_kwargs = config.sub("TRAINER")("KWARGS")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Li2Former(loss_kwargs, model_kwargs).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = train_kwargs["MAX_EPOCHS"]
    attnloss = AttnLoss()
    attnloss.cuda()
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        for batch_idx, batch in enumerate(dataloader):
            x = batch[0].to(device)

            optimizer.zero_grad()  # 清空梯度
            attn = model.run(x)    
            attn_cutouts = BatchLoader.get_attn_cutouts(attn)
            attn_cutouts = attn_cutouts.to(device)
            # 使用 expand 扩展
            attn_cutouts_expanded = attn_cutouts.expand(-1, -1, 5, -1)
            
            loss = attnloss(x, attn_cutouts_expanded)
            loss.backward()
            optimizer.step()

            # 记录每个 batch 的损失到 TensorBoard
            current_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Loss/train', loss.item(), current_step)
            
            # 打印损失
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')

            # 记录损失到 log
            log.append({"epoch": epoch + 1, "batch": batch_idx + 1, "loss": loss.item()})

        # 保存当前检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,  # 假设当前损失值保存在 current_loss 变量中
        }

        checkpoints.append(checkpoint)
        checkpoints.sort(key=lambda x: x['loss'])  # 按损失升序排序

        if len(checkpoints) > 3:
            # 当前 checkpoint 是否在 top 3 中
            if checkpoint not in checkpoints[:3]:
                print(f"当前 checkpoint 的损失为 {loss}，未进入 Top 3 未保存。")
            checkpoints = checkpoints[:3]

        # 判断当前是否在 top 3 中并进行保存
        if checkpoint in checkpoints:
            save_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, save_path)
            print(f"Checkpoint 保存成功，保存路径为: {save_path}")


    # 关闭 TensorBoard 的日志记录器
    writer.close()

        


        
if __name__ == "__main__":
    config_path = "/share/home/tj90055/dhj/Self_Feature_LO/src/point_cloud_processing/src/cfgs/ros_li2former.yaml"
    train(config_path)
