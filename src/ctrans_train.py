# coding=utf-8 
import sys
sys.path.append("/share/home/tj90055/dhj/Self_Feature_LO/src/point_cloud_processing/src")
from data import ScanData
from loss import AttnLoss
from config import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm

class CNN(nn.Module):
    def __init__(self, hidden_size, kernel_size=3):
        super(CNN, self).__init__()
        # 卷积1
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        # 卷积2
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=2 * (kernel_size//2), dilation=2),
            nn.ReLU()
        )
        # 卷积3
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=4 * (kernel_size//2), dilation=4),
            nn.ReLU()
        )
        # 卷积4
        self.conv4 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, kernel_size=kernel_size, padding=8 * (kernel_size//2), dilation=8)
        )
        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        output = self.conv1(X)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.sigmoid(output)
        output = output.reshape((X.shape[0], -1))
        return output

class Trans(nn.Module):
    def __init__(self, hidden_size, kernel_size=3):
        super(Trans, self).__init__()

        # 卷积1
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU()
        )

        num_heads = 4
        # 添加Transformer模块
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads,batch_first=True),
            num_layers=6  # 增加 Transformer 的层数
        )

          # 卷积2
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, kernel_size=kernel_size, padding=kernel_size//2)
        )

        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        
        X = X.reshape((X.shape[0], 1, X.shape[1]))

        output = self.conv1(X)

        output = output.permute(2, 0, 1)
        output = self.transformer(output)
        output = output.permute(1, 2, 0)
        output = self.conv2(output)
        output = self.sigmoid(output)

        output = output.reshape((X.shape[0], -1))
        
        return output


class Ctrans(nn.Module):
    def __init__(self, hidden_size, kernel_size=3):
        super(Ctrans, self).__init__()

        # 卷积1
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU()
        )

        num_heads = 4
        # 添加Transformer模块
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True),
            num_layers=6  # 增加 Transformer 的层数
        )

         # 卷积2
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        # 卷积3
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        # 卷积4
        self.conv4 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2)
        )
         # 卷积5
        self.conv5 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        # 卷积6
        self.conv6 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, kernel_size=kernel_size, padding=kernel_size//2)
        )

        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        
        X = X.reshape((X.shape[0], 1, X.shape[1]))

        output = self.conv1(X)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = output.permute(2, 0, 1)
        output = self.transformer(output)
        output = output.permute(1, 2, 0)
        output = self.conv6(output)
        output = self.sigmoid(output)
        output = output.reshape((X.shape[0], -1))
        
        return output




def train():
    scan = ScanData(data_file, data_length, batch_size, raw_length, sample_times)
    print("----- loading data completely -----")
    for model_name, hidden_size, kernel_size in [["Ctrans", 64, 7], ["Trans", 64, 7], ["CNN", 64, 7]]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = {"CNN": CNN, "Trans": Trans, "Ctrans": Ctrans}[model_name]
        network = net(hidden_size, kernel_size).to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        attnLoss = AttnLoss()  
        print("----- model name: %s -----" % model_name)
        print("----- hidden size: %d -----" % hidden_size)
        print("----- device : %s -----" % device)
        print("----- alpha: %.2f, beta: %.2f -----" % (attnLoss.alpha, attnLoss.beta))
        print("----- learning rate: %f -----" % learning_rate)
        # total_loss, loss1, loss2, loss3, loss4 = 0, 0, 0, 0, 0
        loss_contractive, loss_p, loss_n, loss_n1, loss_n2 = 0, 0, 0, 0, 0

        log = []
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # 创建SummaryWriter，使用当前时间为子目录名
        writer = SummaryWriter('runs/' + model_name + '1027hidden_' + str(hidden_size) + '_kernel_' + str(kernel_size) + '_' + current_time)

        for i in tqdm(range(total_steps + 1), desc="Training Steps"):
            X, yp, yn ,yn1 ,yn2= scan.get_next_batch()
            X = torch.tensor(X, dtype=torch.float32).to(device)
            yp = torch.tensor(yp, dtype=torch.float32).to(device)
            yn = torch.tensor(yn, dtype=torch.float32).to(device)
            yn1 = torch.tensor(yn1, dtype=torch.float32).to(device)
            yn2 = torch.tensor(yn2, dtype=torch.float32).to(device)

            attn = network(X)
            mask = torch.tensor(scan.get_attn_mask())
            attn = attn.to(device) * mask.to(device)
            loss = attnLoss(attn, yp, yn ,yn1 ,yn2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_contractive += loss.item()
            loss_p += attnLoss.loss_p.item()  # 累加正样本损失
            loss_n += attnLoss.loss_n.item()  # 累加负样本损失
            loss_n1 += attnLoss.loss_n1.item()  # 累加负样本损失1
            loss_n2 += attnLoss.loss_n2.item()  # 累加负样本损失2
            log.append({
                "loss_contractive": loss_contractive,
                "loss_p": loss_p,
                "loss_n": loss_n,
                "loss_n1": loss_n1,
                "loss_n2": loss_n2
            })

            if i % print_gap == 0 and i > 0:
                # print("step:%d" % i, "loss = %.3f - %.3f + %.3f + %.3f + %.3f + %.3f" % (
                # loss_contractive / print_gap, loss_p / print_gap, loss_n / print_gap, loss_n1 / print_gap, loss_n2 / print_gap))
                print("step:%d" % i, "loss= %.3f" % (loss_contractive / print_gap))
                print("loss_p: ", loss_p / print_gap)
                print("loss_n: ", loss_n / print_gap)
                print("loss_n1: ", loss_n1 / print_gap)
                print("loss_n2: ", loss_n2 / print_gap)
               
                writer.add_scalar('Loss/contractive', loss_contractive / print_gap, i)
                writer.add_scalar('Loss/p', loss_p / print_gap, i)
                writer.add_scalar('Loss/n', loss_n / print_gap, i)
                writer.add_scalar('Loss/n1', loss_n1 / print_gap, i)
                writer.add_scalar('Loss/n2', loss_n2 / print_gap, i)
                loss_contractive, loss_p, loss_n, loss_n1, loss_n2 = 0, 0, 0, 0, 0

            if i % save_gap == 0 and i > 0:
                # 保存模型到runs子目录下的model文件夹
                model_save_path = 'runs/' + model_name + '1027hidden_' + str(hidden_size) + '_kernel_' + str(kernel_size) + '_' + current_time + '/model.pth'
                torch.save(network.state_dict(), model_save_path)
                print("----- saving model to %s -----" % model_save_path)
                
                # 保存日志文件到runs子目录下的log.json
                log_save_path = 'runs/' + model_name + '1027hidden_' + str(hidden_size) + '_kernel_' + str(kernel_size) + '_' + current_time + '/log.json'
                with open(log_save_path, "w", encoding="utf8") as f:
                    f.write(json.dumps(log))
                print("----- saving log to %s -----" % log_save_path)

        
        writer.close()  # 关闭SummaryWriter

# 确保在调用train()函数之前设置好以下变量
# data_file, data_length, batch_size, raw_length, sample_times, total_steps, print_gap, save_gap, learning_rate


if __name__ == "__main__":
    train()
