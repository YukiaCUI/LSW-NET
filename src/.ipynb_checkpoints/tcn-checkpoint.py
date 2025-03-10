import sys
sys.path.append("E:/learngit/attnslam/src") #添加路径，使得后面能够引用该目录下的模块
from data import ScanData  #导入ScanData类，用于读取数据
from loss import AttnLoss  #导入AttnLoss类，用于定义损失函数
from config import *       #导入配置文件，包含了训练过程中的各种参数
import torch               #用于构建神经网络，torch是PyTorch的核心库，提供了张量操作、自动求导等功能
import torch.nn as nn      #torch.mm是PyTorch的神经网络模块，提供了各种神经网络层和损失函数的实现
import torch.nn.functional as F
import json                #用于处理json格式的数据


class TCN1(nn.Module):
    def __init__(self, hidden_size, kernel_size=3):
        super(TCN1, self).__init__()
        # 卷积1
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU()
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
            nn.Conv1d(hidden_size, 1, kernel_size=kernel_size, padding=kernel_size//2)
        )
        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X = X.reshape((X.shape[0], 1, X.shape[1]))           #reshape输入数据的格式，使其符合卷积层的要求
        output = self.conv1(X)                                      # 第一层卷积
        output = self.conv2(output)                                 # 第二层卷积
        output = self.conv3(output)                                 # 第三层卷积
        output = self.conv4(output)                                 # 第四层卷积
        output = self.sigmoid(output)                               #sigmoid函数
        output = output.reshape((X.shape[0], -1))                   # 将输出的格式再次reshape，使其符合最终输出的要求
        return output


class TCN2(nn.Module):
    def __init__(self, hidden_size, kernel_size=3):
        super(TCN2, self).__init__()
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


class TCN3(nn.Module):
    def __init__(self, hidden_size, kernel_size=3):
        super(TCN3, self).__init__()
        # 卷积+下采样
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # 卷积+下采样
        self.conv2 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # 反卷积+上采样
        self.conv3 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ConvTranspose1d(hidden_size, hidden_size, kernel_size=kernel_size+1, padding=kernel_size//2, stride=2),
            nn.ReLU()
        )
        # 反卷积+上采样
        self.conv4 = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.ConvTranspose1d(hidden_size, 1, kernel_size=kernel_size+1, padding=kernel_size//2, stride=2)
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




def train():
    scan = ScanData(data_file, data_length, batch_size, raw_length, sample_times)                       # 初始化数据读取类
    print("----- loading data completely -----")
    for model_name, hidden_size, kernel_size in [["tcn1", 64, 13], ["tcn2", 64, 7], ["tcn3", 64, 9]]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                          # 判断是否有GPU可用，若有则使用GPU
        TCN = {"tcn1": TCN1, "tcn2": TCN2, "tcn3": TCN3}[model_name]                                     # 根据model_name的不同选择对应的模型
        network = TCN(hidden_size, kernel_size).to(device)                                               # 构建网络并将其放到GPU上进行计算
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)                             # 定义优化器
        attnLoss = AttnLoss(0.1, 100.0)                                                               # 定义损失函数
        print("----- model name: %s -----" % model_name)
        print("----- hidden size: %d -----" % hidden_size)
        print("----- device : %s -----" % device)
        print("----- alpha: %.2f, beta: %.2f -----" % (attnLoss.alpha, attnLoss.beta))
        print("----- learning rate: %f -----" % learning_rate)
        total_loss, loss1, loss2, loss3, loss4 = 0, 0, 0, 0, 0                                           # 定义损失函数的变量

        log = []                                                                                         # 记录日志
        for i in range(total_steps + 1):
            X, yp, yn = scan.get_next_batch()                                                              # 获取下一批数据
            X = torch.tensor(X, dtype=torch.float32).to(device)                                            # 转换数据类型，并将其放到GPU上进行计算
            yp = torch.tensor(yp, dtype=torch.float32).to(device)
            yn = torch.tensor(yn, dtype=torch.float32).to(device)

            attn = network(X)                                                                              #将输入数据X传入神经网络模型network进行前向计算，得到输出attn。
            mask = torch.tensor(scan.get_attn_mask())                                                      #获取注意力掩码mask，用于限制注意力的作用范围。
            attn = attn.to(device) * mask.to(device)                                                       #将attn和mask移到同一设备上，并将attn与mask相乘，限制注意力的作用范围。
            loss = attnLoss(attn, yp, yn)                                                                  #计算损失函数，输入为attn、正样本yp和负样本yn。
            optimizer.zero_grad()                                                                          #清空优化器的梯度缓存。
            loss.backward()                                                                                #反向传播，计算梯度。
            optimizer.step()                                                                               #更新模型参数。

            loss1 += attnLoss.loss1
            loss2 += attnLoss.loss2
            loss3 += attnLoss.loss3
            loss4 += attnLoss.loss4                                                                        #累加每个训练步骤的四个损失函数的值
            total_loss += loss                                                                             #累加每个训练步骤的总损失函数的值
            log.append({"loss1": attnLoss.loss1.item(),
                        "loss2": attnLoss.loss2.item(),
                        "loss3": attnLoss.loss3.item(),
                        "loss4": attnLoss.loss4.item(),
                        "total_loss": loss.item()})                                                         # 将当前训练步骤的损失函数值记录到日志中

            if i % print_gap == 0 and i > 0:                                                                 #每隔一定步数，输出当前训练步骤的损失函数值。
                print("step:%d" % i, "loss = %.3f - %.3f + %.3f + %.3f = %.3f" % (
                loss1 / print_gap, loss2 / print_gap, loss3 / print_gap, loss4 / print_gap, total_loss / print_gap))
                total_loss, loss1, loss2, loss3, loss4 = 0, 0, 0, 0, 0
            if i % save_gap == 0 and i > 0:                                                                 #每隔一定步数，保存模型和日志文件。
                torch.save(network, 'E:/learngit/attnslam/model/%s_hidden_%d_kernel_%d.pkl' % (model_name, hidden_size, kernel_size))
                print("----- saving model %s_hidden_%d_kernel_%d.pkl -----" % (model_name, hidden_size, kernel_size))
                with open('E:/learngit/attnslam/log/%s_hidden_%d_kernel_%d_log.json' % (model_name, hidden_size, kernel_size), "w", encoding="utf8") as f:
                    f.write(json.dumps(log))


if __name__ == "__main__":
    train()
