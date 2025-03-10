#!/usr/bin/env python
import rospy
import message_filters
from sensor_msgs.msg import PointCloud2
from argparse import ArgumentParser
from sensor_msgs.msg import LaserScan
from dataplay import ScanData  #导入ScanData类，用于读取数据
from config import *       #导入配置文件，包含了训练过程中的各种参数
import torch               #用于构建神经网络，torch是PyTorch的核心库，提供了张量操作、自动求导等功能
import torch.nn as nn      #torch.mm是PyTorch的神经网络模块，提供了各种神经网络层和损失函数的实现
import torch.nn.functional as F
import json                #用于处理json格式的数据
from TCN_model import * 

def laser_callback(msg):
    # 处理激光点云数据
    scan_data = {}
    threshold = 15.0 # 设置距离阈值
    for i, r in enumerate(msg.ranges):
        angle = msg.angle_min + i * msg.angle_increment
        key = "{:.2f}_{:.2f}_{:.2f}".format(r, angle, msg.range_max)
        if r <= 30 : # 测量到障碍物，使用 "full" 作为键名
            scan_data[key] = {"full": [list(msg.ranges),list(msg.ranges),list(msg.ranges),list(msg.ranges),list(msg.ranges),list(msg.ranges)]}
        else: # 没有测量到障碍物，使用 "empty" 作为键名
            scan_data[key] = {"empty": [list(msg.ranges),list(msg.ranges),list(msg.ranges),list(msg.ranges),list(msg.ranges),list(msg.ranges)]}
    data = scan_data
    path = "/home/rosnoetic/pointcloud/src/point_cloud_processing/data/point_cloud.txt"   
    file = open(path,"w")
    file.write(str(data))
    file.close()
    
    scan = ScanData(data, data_length, batch_size, raw_length, sample_times)                       # 初始化数据读取类
    print("----- loading data completely -----")
    for model_name, hidden_size, kernel_size in [["tcn1", 64, 13], ["tcn2", 64, 7], ["tcn3", 64, 9]]:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")                          # 判断是否有GPU可用，若有则使用GPU
        TCN = {"tcn1": TCN1, "tcn2": TCN2, "tcn3": TCN3}[model_name]                                     # 根据model_name的不同选择对应的模型
        network = TCN(hidden_size, kernel_size).to(device)                                               # 构建网络并将其放到GPU上进行计算
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)                             # 定义优化器
        attnLoss = AttnLoss(0.1, 100.0)                                                                  # 定义损失函数
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
                torch.save(network, '/home/rosnoetic/pointcloud/src/point_cloud_processing/model/%s_hidden_%d_kernel_%d.pkl' % (model_name, hidden_size, kernel_size))
                print("----- saving model %s_hidden_%d_kernel_%d.pkl -----" % (model_name, hidden_size, kernel_size))
                with open('/home/rosnoetic/pointcloud/src/point_cloud_processing/log/%s_hidden_%d_kernel_%d_log.json' % (model_name, hidden_size, kernel_size), "w", encoding="utf8") as f:
                    f.write(json.dumps(log))

    pub = rospy.Publisher('processed_pointcloud', LaserScan, queue_size=100)
    pub.publish(attn*msg.ranges)
    # 输出词典类型数据
    print(attn) 

def main():
    rospy.init_node('laser_listener', anonymous=True)
    rospy.Subscriber("front_scan", LaserScan, laser_callback)
    rospy.spin()

if __name__ == '__main__':
    main()