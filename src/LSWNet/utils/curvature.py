import numpy as np
from scipy.spatial import KDTree
def polar_to_cartesian(radius_array):
    """Convert polar coordinates (radius) to Cartesian coordinates."""
    
    N = radius_array.shape[0]  # 获取点的数量
    angles = np.linspace(0, 2 * np.pi, N)  # Shape: (N,)
    
    x = radius_array * np.cos(angles)
    y = radius_array * np.sin(angles)
    
    return np.stack((x, y), axis=-1)  # Shape: (N, 2)
def compute_curvature_least_squares(points, r):
    """
    计算二维点云中每个点的曲率，基于 KD-Tree 和最小二乘法拟合。

    参数:
        points: np.ndarray, 形状为 (N, 2)，表示点云的 (x, y) 坐标。
        r: float, 邻域半径。
    
    返回:
        curvatures: np.ndarray, 形状为 (N,)，每个点的曲率。
    """
    points = polar_to_cartesian(points)
    # 构建 KD-Tree
    tree = KDTree(points)
    
    # 初始化曲率数组
    curvatures = np.zeros(points.shape[0])
    
    # 遍历每个点，计算邻域内的曲率
    for i, point in enumerate(points):
        # 获取半径为 r 的邻域点索引
        indices = tree.query_ball_point(point, r)
        
        if len(indices) < 10:
            # 邻域内点数少于 10，无法拟合曲线
            curvatures[i] = 0
            continue
        
        # 获取邻域点
        neighbors = points[indices]
        
        # 中心化邻域点
        neighbors -= neighbors.mean(axis=0)
        
        # 使用最小二乘法拟合二次多项式 y = ax^2 + bx + c
        x, y = neighbors[:, 0], neighbors[:, 1]
        A = np.vstack([x**2, x, np.ones_like(x)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b, _ = coeffs
        
        # 曲率公式: k = 2|a| / (1 + b^2)^(3/2)
        curvature = 2 * abs(a) / (1 + b**2)**1.5
        curvatures[i] = curvature
    
    return curvatures
