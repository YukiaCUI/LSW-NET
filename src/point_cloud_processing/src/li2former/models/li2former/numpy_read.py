import numpy as np

# 指定.npy文件的路径
file_path = '/home/tju_dhj/Self_Feature_LO/data/dianxin6.npy'

# 使用numpy的load函数读取.npy文件
data = np.load(file_path)

# 打印读取的数据
print(data)