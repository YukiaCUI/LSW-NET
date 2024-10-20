import json
import numpy as np
import random


class ScanData(object):
    def __init__(self, data_file, data_length, batch_size, raw_length, sample_times, inf=35):
        """
        激光数据
        :param data_file: json格式的文件名, 可以是str和list(str)
        :param data_length: 每个激光数据的有效长度
        :param batch_size: 每个batch的数据量
        :param raw_length日志
        for i in range(total_steps + 1):: 每个数据的原始长度
        :param sample_times: 每个数据的采样次数
        """
        if isinstance(data_file, str):
            data_file = [data_file]
        self.data = {}
        for fname in data_file:
            with open("E:/learngit/attnslam/data/" + fname, "r", encoding="utf8") as f:
                self.data.update(json.loads(f.read()))
        self.data_length = data_length
        self.batch_size = batch_size
        self.raw_length = raw_length
        self.sample_times = sample_times
        self.keys = list(self.data.keys())
        random.shuffle(self.keys)
        self.cursor = 0
        self.inf = inf

    def get_next_batch(self):
        """
        获取一个batch的数据
        """
        positive_index = self.keys[self.cursor: self.cursor + self.batch_size // 2]         # 样本采样
        if len(positive_index) < self.batch_size // 2:                                      # 如果当前剩余样本不够一个batch
            positive_index += self.keys[:self.cursor+self.batch_size//2-len(self.keys)]     # 从前面补齐
        positive_isempty = np.random.randint(0, 2, (self.batch_size,))                      # 用于判断正采样数据有无障碍物
        negative_index = random.sample(self.keys, self.batch_size)                          # 负采样
        negative_isempty = np.random.randint(0, 2, (self.batch_size,))                      # 用于判断负采样数据有无障碍物
        pos = random.randint(0, self.sample_times - 2)                                      # 随机生成一个循环采样的位置
        left = np.random.randint(0, self.raw_length - self.data_length + 1)                 # 随机生成数据切片的左端点
        right = left + self.data_length                                                     # 数据切片的又端点
        X = []
        yp = []
        yn = []
        for i in range(self.batch_size):
            X.append(self.data[positive_index[i//2]]["empty"][pos][left:right])
            if positive_isempty[i] == 1:
                yp.append(self.data[positive_index[i//2]]["empty"][pos + 1][left:right])
            else:
                yp.append(self.data[positive_index[i//2]]["full"][pos][left:right])
            if negative_isempty[i] == 1:
                yn.append(self.data[negative_index[i]]["empty"][pos][left:right])
            else:
                yn.append(self.data[negative_index[i]]["full"][pos][left:right])
            if i % 2 == 0:
                pos += 1
            else:
                pos = (pos + 1) % (self.sample_times - 1)
        self.X = np.array(X)                                                                     # 原始激光数据
        self.X[self.X > 35] = 35.0
        yp = self.X - np.array(yp)
        yn = self.X - np.array(yn)
        self.mask = np.ones_like(self.X)
        self.mask[self.X == self.inf] = 0
        X = self.X.copy()
        X[:, :-1] = X[:, :-1] - X[:, 1:]                                                    # 特征工程
        X[:, -1] = X[:, -2] 
        X[:, :-1] = X[:, :-1] - X[:, 1:]
        X[:, -1] = X[:, -2]
        # X = (X - X.mean()) / X.std()                                                        # 标准化
        self.cursor = (self.cursor + self.batch_size) % len(self.keys)                      # 更新cursor
        return X, yp, yn

    def get_attn_mask(self):
        return self.mask

    def get_X(self):
        return self.X

    def get_sequence(self, empty=True):
        index = random.choice(self.keys)
        if empty:
            X = self.data[index]["empty"]
        else:
            X = self.data[index]["full"]
        self.X = np.array(X)  # 原始激光数据
        self.X[self.X > 35] = 35.0
        X = self.X.copy()
        X[:, :-1] = X[:, :-1] - X[:, 1:]  # 特征工程
        X[:, -1] = X[:, -2]
        X[:, :-1] = X[:, :-1] - X[:, 1:]
        X[:, -1] = X[:, -2]
        return self.X, X

    def get_obstacle(self, delta=0):
        index = random.choice(self.keys)
        X1 = self.data[index]["empty"]
        X2 = self.data[index]["full"]
        i = random.choice(range(6 - delta))
        self.X = np.array([X1[i], X2[i+delta]])
        self.X[self.X > 35] = 35.0
        X = self.X.copy()
        X[:, :-1] = X[:, :-1] - X[:, 1:]
        X[:, -1] = X[:, -2]
        X[:, :-1] = X[:, :-1] - X[:, 1:]
        X[:, -1] = X[:, -2]
        return self.X, X


if __name__ == "__main__":
    data_file = ["data_0.json", "data_1.json", "data_2.json", "data_3.json", "data_4.json",
                 "data_5.json", "data_6.json", "data_7.json", "data_8.json", "data_9.json"]
    data_length = 768
    batch_size = 256
    raw_length = 768
    sample_times = 5
    scan = ScanData(data_file, data_length, batch_size, raw_length, sample_times)
    X, yp, yn = scan.get_next_batch()
