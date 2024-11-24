import numpy as np
from tqdm import tqdm
from utils.curvature import compute_curvature_least_squares

class PointCloudSequenceDataset:
    def __init__(self, data, T):
        """
        初始化数据集
        :param data: 输入的点云数据 (N, M) 形状的 NumPy 数组，其中 N 是数据的总数，M 是每个数据点的特征数量
        :param T: 子序列的长度，T 必须为奇数
        """
        self.data = data
        self.T = T
        self.padding = T // 2  # 前后各取的帧数，T 必须为奇数

    def __len__(self):
        # 返回数据集的长度
        return len(self.data)

    def __getitem__(self, idx):
        if idx < self.padding:
            # 如果索引在前两帧范围
            start = 0
            end = self.T
            
        elif idx >= len(self.data) - self.padding:
            # 如果索引在最后两帧范围
            start = len(self.data) - self.T
            end = len(self.data)
        else:
            # 一般情况，中间帧处理
            start = idx - self.padding
            end = idx + self.padding + 1

        # 提取子序列
        sequence = self.data[start:end]
        # # 将 self.data[idx] 挪到 sequence[self.padding] 处
        # if not np.array_equal(sequence[self.padding], self.data[idx]):
        #     # 找到 self.data[idx] 在 sequence 中的位置
        #     data_to_move = self.data[idx]

        #     # 确保 data_to_move 的形状与 sequence 中的元素一致
        #     # 如果需要，可以通过 reshape 来确保形状一致
        #     data_to_move = np.reshape(data_to_move, sequence[self.padding].shape)

        #     # 从 sequence 中删除该元素
        #     sequence = np.delete(sequence, np.where(np.all(sequence == data_to_move, axis=1))[0], axis=0)

        #     # 将该元素插入到 sequence[self.padding] 的位置
        #     sequence = np.insert(sequence, self.padding, data_to_move, axis=0)

        return sequence  # 返回 NumPy 数组



# 构建数据集
T = 5  # 每帧取 T 个帧打包
data_paths = [
        "/share/home/tj90055/dhj/Self_Feature_LO/dianxin1.npy",
        "/share/home/tj90055/dhj/Self_Feature_LO/dianxin6.npy",
        "/share/home/tj90055/dhj/Self_Feature_LO/dianxinb1.npy"
    ]
# 加载每个数据集并构建 PointCloudSequenceDataset
datasets = []
for path in data_paths:
    # 加载点云数据 (n, N)
    point_cloud_data = np.load(path)
    column_to_add = np.full((point_cloud_data.shape[0], 1), 35.0)
    point_cloud_data = np.hstack((point_cloud_data, column_to_add))
    point_cloud_data[point_cloud_data > 35] = 35.0
    curvatures = []

    # point_cloud_data = point_cloud_data[:10]

    # 计算曲率
    for idx, data in enumerate(tqdm(point_cloud_data)):
        curvature = compute_curvature_least_squares(data, r=10)
        curvatures.append(curvature)
    curvatures = np.array(curvatures)

    print("curvature shape:", curvatures.shape)
    print("point_cloud_data shape:", point_cloud_data.shape)

    point_cloud_data = np.stack((point_cloud_data, curvatures), axis=-1)
    
    dataset = PointCloudSequenceDataset(point_cloud_data, T)
    
    # 将所有子序列收集到一个 NumPy 数组中
    sequences = []
    
    for idx in tqdm(range(len(dataset))):
        sequence = dataset[idx]  # 获取子序列
        sequences.append(sequence)  # 将每个子序列添加到列表

    # 将收集到的子序列转换为 (n, T, N, 2) 的形状
    sequences = np.array(sequences)
    print("Sequences shape:", sequences.shape)  # 输出最终形状 (n, T, N, 2)

    x = sequences[:, :, :, 0]
    mse = np.var(x, axis=1, keepdims=True)
    mse_repeated = np.repeat(mse, T, axis=1)
    final_data = np.concatenate((sequences, mse_repeated[:, :, :, np.newaxis]), axis=-1)
    print("final_data shape:", final_data.shape)

    datasets.append(final_data)

datasets = np.concatenate(datasets, axis=0)
print("final Dataset shape:", datasets.shape)
# save datasets as npy
np.save("/share/home/tj90055/dhj/Self_Feature_LO/src/point_cloud_processing/src/LSWNet/data/pre_train_data.npy", datasets)

