import numpy as np
from detector import Detector
from config.config import Config
from data_loader import DataGet 
from models.li2former import Li2Former
import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train(config_path):
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
    model_kwargs = config.sub("MODEL")("KWARGS")
    loss_kwargs  = config.sub("LOSS")("KWARGS")
    train_kwargs = config.sub("TRAINER")("KWARGS")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Li2Former(loss_kwargs, model_kwargs).to(device) 

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = train_kwargs["MAX_EPOCHS"]
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        for batch in dataloader:
            x = batch[0].to(device)

            optimizer.zero_grad()  # 清空梯度
            attn = model.run(x)    
            attn_cutouts = BatchLoader.get_attn_cutouts(attn)
            # print("x0.shape: ", x.shape)
            # print("attn0.shape: ", attn.shape)
            print("attn_cutouts0.shape: ", attn_cutouts.shape)
            
            # x.shape = (B, C=1800, T=5, P)
            # attn,shape = (B, 1, C=1800)
            # attn_cutouts.shape = (B, C=1800, 1, P)
    
            

            # reshape to satisfy the input of the loss functione
            B, C, T, P = x.shape
            x = x[:, :, 0, :].unsqueeze(2)
            x = x.permute(0,2,1,3)
            x = x.repeat(1, 3, 1, 1)
            B, chanel, Ct, P = x.shape
            x = x.reshape(B, chanel, Ct/2, P*2)

            attn_cutouts = attn_cutouts.permute(0, 2, 1, 3)
            attn_cutouts = attn_cutouts.repeat(1, 3, 1, 1)
            B, chanel, Ct, P = attn_cutouts.shape
            attn_cutouts = attn_cutouts.reshape(B, chanel, Ct/2, P*2)

            masks = torch.randperm(Ct)[:224]
            x = x[:, :, masks, :]
            attn_cutouts = attn_cutouts[:, :, masks, :]


            # TODO: 定义损失函数MocoLoss
            loss = attnLoss(x, attn_cutouts)
            loss.backward()
            optimizer.step()  

            # # 记录损失到 TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch)
            log.append({"loss": loss.item()})

    # 保存当前检查点
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'loss': loss.item(), 
    }
    
    # 添加检查点并根据损失进行排序
    checkpoints.append(checkpoint)
    checkpoints.sort(key=lambda x: x['loss'])  # 按损失升序排序

    # 保留最好的前 max_checkpoints 个检查点
    if len(checkpoints) > 3:
        checkpoints = checkpoints[:3]

    # 保存当前 epoch 的检查点
    torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')

    # 关闭 TensorBoard 的日志记录器
    writer.close()
        

if __name__ == "__main__":
    config_path = "/media/cyj/DATA/Self_Feature_LO/src/point_cloud_processing/src/cfgs/ros_li2former.yaml"
    train(config_path)
