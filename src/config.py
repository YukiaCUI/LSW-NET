
# coding=utf-8 
# 数据文件
data_folder = ""
data_file = ["data_0.json", "data_1.json", "data_2.json", "data_3.json", "data_4.json",
             "data_5.json","data_6.json", "data_7.json" ,"data_8.json" ,"data_9.json" ]
# 模型参数
hidden_size = 64
kernel_size = 7
learning_rate = 5e-5
data_length = 768
batch_size = 128
raw_length = 768
sample_times = 5

# 训练参数
total_steps = 50000
print_gap = 1
show_gap = 1
save_gap = 5000