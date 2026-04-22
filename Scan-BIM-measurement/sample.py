import numpy as np
from scipy.spatial import cKDTree
import warnings
import random
import torch
warnings.filterwarnings('ignore')  # 忽略一些不重要的警告

# 设置随机种子以确保可重复性
np.random.seed(42)
random.seed(42)

# 向量化的最远点采样函数（优化版本）

def farthest_point_sampling(points, n_samples):
    """
    最远点采样实现 - 优化版本
    使用NumPy的广播和向量化操作，提升性能
    """
    points = np.asarray(points, dtype=np.float32)
    N = points.shape[0]
    
    if N <= n_samples:
        return np.arange(N)
    
    # 初始化
    sampled_indices = np.zeros(n_samples, dtype=np.int64)
    distances = np.full(N, np.inf)
    
    # 第一个点随机选择
    sampled_indices[0] = np.random.randint(0, N)
    
    # 计算第一个点到所有点的距离
    diff = points - points[sampled_indices[0]]
    distances = np.sum(diff * diff, axis=1)
    
    # 迭代采样
    for i in range(1, n_samples):
        # 选择距离最大的点
        next_idx = np.argmax(distances)
        sampled_indices[i] = next_idx
        
        # 更新距离：计算新采样点到所有点的距离，并保留最小值
        diff = points - points[next_idx]
        new_distances = np.sum(diff * diff, axis=1)
        distances = np.minimum(distances, new_distances)
    
    return sampled_indices

# 向量化的特征加权最远点采样函数

def feature_fps(points, features, n_samples):
    """
    Feature-aware FPS - NumPy向量化版本
    原理：在FPS的基础上，加入特征权重
    """
    points = np.asarray(points, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    N = points.shape[0]
    n_samples = min(n_samples, N)
    
    if N <= n_samples:
        return np.arange(N)
    
    # 初始化
    sampled_indices = np.zeros(n_samples, dtype=np.int64)
    distances = np.full(N, np.inf)
    is_selected = np.zeros(N, dtype=bool)
    
    # 第一个点：选特征分数最高的
    first_idx = np.argmax(features)
    sampled_indices[0] = first_idx
    is_selected[first_idx] = True
    
    # 计算初始距离
    diff = points - points[first_idx]
    distances = np.sum(diff * diff, axis=1)
    
    # 迭代采样
    for i in range(1, n_samples):
        # 计算加权距离（避开已选择的点）
        weighted_distances = distances * (1 + features)
        weighted_distances[is_selected] = -1.0  # 排除已选中的点
        
        # 选择加权距离最大的点
        next_idx = np.argmax(weighted_distances)
        sampled_indices[i] = next_idx
        is_selected[next_idx] = True
        
        # 更新距离
        diff = points - points[next_idx]
        new_distances = np.sum(diff * diff, axis=1)
        distances = np.minimum(distances, new_distances)
    
    return sampled_indices

# 统一采样接口函数

def random_sampling(points, num_samples, method='random'):
    """
    统一采样接口，支持多种采样方法
    
    Args:
        points: (N, 3) 原始点云
        num_samples: 目标采样点数
        method: 采样方法，可选值: 'random', 'fps', 'structure_preserving'
    
    Returns:
        sampled_points: (M, 3) 采样后点云
        sampled_indices: (M,) 采样索引
        point_types: (M,) 点类型 (0=平面, 1=边缘, 2=角点)，仅在method='structure_preserving'时有效
    """
    points = np.asarray(points, dtype=np.float32)
    N = points.shape[0]
    
    if method == 'random':
        # 随机采样
        if N <= num_samples:
            indices = np.arange(N)
        else:
            indices = np.random.choice(N, num_samples, replace=False)
        sampled_points = points[indices]
        point_types = np.zeros(len(indices), dtype=int)  # 默认全为平面点
        return sampled_points, indices, point_types
    
    elif method == 'fps':
        # 最远点采样
        indices = farthest_point_sampling(points, num_samples)
        sampled_points = points[indices]
        point_types = np.zeros(len(indices), dtype=int)  # 默认全为平面点
        return sampled_points, indices, point_types
    
    elif method == 'structure_preserving':
        # 结构保留采样
        sampler = HierarchicalStructurePreservingSampling(
            target_num_points=num_samples,
            corner_ratio=0.15,
            edge_ratio=0.35,
            plane_ratio=0.50
        )
        sampled_points, indices, point_types = sampler.sample(points)
        return sampled_points, indices, point_types
    
    else:
        raise ValueError(f"不支持的采样方法: {method}")

def _knn_indices(points_np, k):
    tree = cKDTree(points_np)
    _, idx = tree.query(points_np, k=k)
    if k == 1:
        idx = idx.reshape(-1, 1)
    return idx

def _structure_scores(points_np, k, device=None, amp=False):
    """
    计算结构分数 - NumPy优化版本
    使用NumPy替代PyTorch，避免CPU-GPU开销
    """
    N = points_np.shape[0]
    idx = _knn_indices(points_np, k)
    
    # 获取邻居点
    neigh = points_np[idx.reshape(-1)].reshape(N, k, 3)
    
    # 计算均值
    mu = np.mean(neigh, axis=1, keepdims=True)
    
    # 中心化
    cen = neigh - mu
    
    # 计算协方差矩阵
    cov = np.matmul(cen.transpose(0, 2, 1), cen) / float(k)
    
    # 特征值分解
    evals, evecs = np.linalg.eigh(cov)
    eps = 1e-8
    
    # 获取特征值（降序排列）
    l1 = np.maximum(evals[:, 2], eps)
    l2 = np.maximum(evals[:, 1], eps)
    l3 = np.maximum(evals[:, 0], eps)
    
    # 计算结构分数
    L = (l1 - l2) / l1
    P = (l2 - l3) / l1
    S = l3 / l1
    
    # 计算法线
    normals = evecs[:, :, 0]
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
    
    # 计算法线一致性
    neigh_normals = normals[idx.reshape(-1)].reshape(N, k, 3)
    dots = np.abs(np.sum(neigh_normals * normals[:, np.newaxis, :], axis=2))
    C = np.mean(dots, axis=1)
    
    # 裁剪到[0, 1]
    L = np.clip(L, 0.0, 1.0)
    P = np.clip(P, 0.0, 1.0)
    S = np.clip(S, 0.0, 1.0)
    C = np.clip(C, 0.0, 1.0)
    
    return L, P, S, C

# 角点检测器
class CornerDetector:
    """
    角点检测器 - 针对钢结构优化版本
    利用钢结构中直角特征丰富这一特点
    """

    def detect(self, points, k=20):
        """
        针对钢结构优化的角点检测
        利用方形管结构中直角特征丰富这一特点
        """
        points = np.asarray(points, dtype=np.float32)
        N = points.shape[0]
        scores = np.zeros(N, dtype=np.float32)
        
        # 对于大型点云，降低k值以提高性能
        if N > 50000:
            k = min(k, 10)
        
        # 使用cKDTree进行高效的近邻搜索
        tree = cKDTree(points)
        
        # 批量处理
        batch_size = min(1000, N)
        
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            
            # 批量查询k近邻
            _, all_indices = tree.query(points[start_idx:end_idx], k=k)
            
            for i in range(end_idx - start_idx):
                global_idx = start_idx + i
                indices = all_indices[i]
                
                if len(indices) < 3:
                    continue

                neighbors = points[indices]
                
                # 计算局部点的中心
                center = np.mean(neighbors, axis=0)
                
                # 计算协方差矩阵
                diff = neighbors - center
                cov_matrix = np.dot(diff.T, diff) / len(neighbors)
                
                # 特征值分解
                eigenvalues, _ = np.linalg.eigh(cov_matrix)
                eigenvalues = np.sort(eigenvalues)[::-1]  # 降序
                
                # 确保特征值非负
                eigenvalues = np.maximum(eigenvalues, 0)
                
                # 针对钢结构的角点检测：
                # 角点特征：三个特征值相对均衡且都不小
                total_eigen = np.sum(eigenvalues) + 1e-8
                if total_eigen > 1e-6:
                    # 归一化的特征值
                    norm_eigen = eigenvalues / total_eigen
                    # 角点分数：特征值分布的均匀性，但偏向于三个值都较大的情况
                    uniformity = 1.0 - np.var(norm_eigen)  # 方差越小越均匀
                    # 同时考虑特征值的大小，避免将噪声点识别为角点
                    magnitude = np.mean(norm_eigen)
                    corner_score = uniformity * magnitude
                    scores[global_idx] = corner_score

        # 归一化到[0, 1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        return scores

# 边缘检测器
class EdgeDetector:
    """
    边缘检测器 - 针对钢结构优化版本
    利用钢结构中直边特征丰富这一特点
    """

    def detect(self, points, k=20, corner_scores=None):
        """
        针对钢结构优化的边缘检测
        利用方形管结构中直边特征丰富这一特点
        """
        points = np.asarray(points, dtype=np.float32)
        N = points.shape[0]
        scores = np.zeros(N, dtype=np.float32)
        
        # 对于大型点云，降低k值以提高性能
        if N > 50000:
            k = min(k, 10)
        
        # 使用cKDTree进行高效的近邻搜索
        tree = cKDTree(points)
        
        # 批量处理
        batch_size = min(1000, N)
        
        for start_idx in range(0, N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            
            # 批量查询k近邻
            _, all_indices = tree.query(points[start_idx:end_idx], k=k)
            
            for i in range(end_idx - start_idx):
                global_idx = start_idx + i
                indices = all_indices[i]
                
                if len(indices) < 3:
                    continue

                neighbors = points[indices]
                
                # 计算局部点的中心
                center = np.mean(neighbors, axis=0)
                
                # 计算协方差矩阵
                diff = neighbors - center
                cov_matrix = np.dot(diff.T, diff) / len(neighbors)
                
                # 特征值分解
                eigenvalues, _ = np.linalg.eigh(cov_matrix)
                eigenvalues = np.sort(eigenvalues)[::-1]  # 降序
                
                # 确保特征值非负
                eigenvalues = np.maximum(eigenvalues, 0)
                
                # 针对钢结构的边缘检测：
                # 边缘特征：一个特征值小，两个特征值大
                total_eigen = np.sum(eigenvalues) + 1e-8
                if total_eigen > 1e-6:
                    # 归一化的特征值
                    norm_eigen = eigenvalues / total_eigen
                    # 边缘分数：最小特征值小，其他特征值大
                    # 使用差值来衡量特征，差值越大说明越像边缘
                    edge_score = max(0, (norm_eigen[0] - norm_eigen[1]) * 2.0 + 
                                     (norm_eigen[1] - norm_eigen[2]) * 1.0)
                    scores[global_idx] = edge_score

        # 归一化到[0, 1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        # 与角点的关系处理：降低角点区域的边缘分数
        if corner_scores is not None:
            scores = scores * (1 - corner_scores * 0.7)  # 更强地抑制角点区域的边缘分数
        
        return scores

# 分层结构保留采样器
class HierarchicalStructurePreservingSampling:
    """
    分层结构保留采样器 - 针对钢结构优化版本
    针对以方形管焊接为主的钢结构点云特点优化
    """

    def __init__(self, target_num_points=1000, corner_ratio=0.25, edge_ratio=0.60, plane_ratio=0.15, k=None, device=None):
        """
        初始化分层结构保留采样器
        调整默认比例以适应钢结构点云特征:
        - 角点: 30% (钢结构中角点特征丰富且重要)
        - 边缘: 50% (方形管的边缘是主要结构特征)
        - 平面: 20% (相对较少但必要的区域)
        """
        self.target_num = target_num_points
        # 配额分配 (针对钢结构点云优化)
        self.corner_ratio = corner_ratio
        self.edge_ratio = edge_ratio
        self.plane_ratio = plane_ratio
        self.k = k
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.corner_detector = CornerDetector()
        self.edge_detector = EdgeDetector()

    def sample(self, points, return_indices=False):
        if points is None or len(points) == 0:
            raise ValueError("输入点云不能为空")
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("点云格式必须为(N, 3)")
        N = points.shape[0]
        if N <= self.target_num:
            types = np.zeros(N, dtype=int)
            return points, np.arange(N), types
        k = self.k or min(20, max(10, int(N * 0.01)))
        # 使用NumPy版本的结构分数计算
        L, P, S, C = _structure_scores(points, k)
        corner_score = S * (1.0 - C) * (1.0 - L)
        edge_score = L * (1.0 - C) * (1.0 - 0.5 * P)
        plane_score = P * C * (1.0 - 0.5 * L) * (1.0 - 0.5 * S)
        corner_order = np.argsort(corner_score)[::-1]
        edge_order = np.argsort(edge_score)[::-1]
        plane_order = np.argsort(plane_score)[::-1]
        n_corner = min(int(self.target_num * self.corner_ratio), len(corner_order))
        n_edge = min(int(self.target_num * self.edge_ratio), len(edge_order))
        n_plane = min(int(self.target_num * self.plane_ratio), len(plane_order))
        corner_pool = corner_order
        edge_pool = edge_order
        edge_pool = edge_pool[~np.isin(edge_pool, corner_pool[:n_corner])]
        plane_pool = plane_order
        plane_pool = plane_pool[~np.isin(plane_pool, corner_pool[:n_corner])]
        plane_pool = plane_pool[~np.isin(plane_pool, edge_pool[:n_edge])]
        if n_corner > 0:
            c_top = corner_pool[: max(n_corner * 3, n_corner)]
            if len(c_top) <= n_corner:
                c_sel = c_top
            else:
                # 使用NumPy版本的最远点采样
                c_local = farthest_point_sampling(points[c_top], n_corner)
                c_sel = c_top[c_local]
        else:
            c_sel = np.array([], dtype=np.int64)
        if n_edge > 0:
            if len(edge_pool) <= n_edge:
                e_sel = edge_pool
            else:
                e_feat = 0.7 * L[edge_pool] + 0.3 * (1.0 - C[edge_pool])
                # 使用NumPy版本的特征FPS
                e_local = feature_fps(points[edge_pool], e_feat, n_edge)
                e_sel = edge_pool[e_local]
        else:
            e_sel = np.array([], dtype=np.int64)
        if n_plane > 0:
            if len(plane_pool) <= n_plane:
                p_sel = plane_pool
            else:
                # 使用NumPy版本的最远点采样
                p_local = farthest_point_sampling(points[plane_pool], n_plane)
                p_sel = plane_pool[p_local]
        else:
            p_sel = np.array([], dtype=np.int64)
        sel = np.concatenate([c_sel, e_sel, p_sel])
        if sel.shape[0] < self.target_num:
            remaining = np.setdiff1d(np.arange(N), sel)
            add_cnt = self.target_num - sel.shape[0]
            # 使用NumPy版本的最远点采样
            add_local = farthest_point_sampling(points[remaining], min(add_cnt, remaining.shape[0]))
            sel = np.concatenate([sel, remaining[add_local]])
        pts = points[sel]
        types = np.concatenate([np.full(c_sel.shape[0], 2), np.full(e_sel.shape[0], 1), np.full(p_sel.shape[0], 0)])
        return pts, sel, types

    def _simple_feature_sampling(self, features, n_target):
        """
        简化版特征采样
        """
        N = len(features)
        if N <= n_target:
            return np.arange(N)
        
        # 确保分数非负
        features = np.maximum(features, 0)
        total_score = features.sum()
        
        if total_score > 0:
            probs = features / total_score
        else:
            probs = np.ones(N) / N
            
        return np.random.choice(N, n_target, replace=False, p=probs)

    def _simplified_fps(self, points, n_target, device=None):
        N = len(points)
        if N <= n_target:
            return np.arange(N)
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(points, np.ndarray):
            pts = torch.from_numpy(points).float().to(device)
        else:
            pts = points.to(device)
        sampled_indices = torch.zeros(n_target, dtype=torch.long, device=device)
        distances = torch.full((N,), float('inf'), device=device)
        sampled_indices[0] = torch.randint(0, N, (1,), device=device).item()
        distances = torch.sum((pts - pts[sampled_indices[0]:sampled_indices[0]+1]) ** 2, dim=1)
        max_iterations = min(n_target, 100)
        for i in range(1, max_iterations):
            next_idx = torch.argmax(distances)
            sampled_indices[i] = next_idx
            new_distances = torch.sum((pts - pts[next_idx:next_idx+1]) ** 2, dim=1)
            distances = torch.min(distances, new_distances)
        if n_target > max_iterations:
            remaining_count = n_target - max_iterations
            remaining_indices = torch.arange(N, device=device)
            mask = torch.ones(N, dtype=torch.bool, device=device)
            mask[sampled_indices[:max_iterations]] = False
            remaining_indices = remaining_indices[mask]
            if remaining_indices.shape[0] >= remaining_count:
                add = remaining_indices[:remaining_count]
                sampled_indices[max_iterations:n_target] = add
        return sampled_indices.cpu().numpy()

# GPU加速的最远点采样函数
def farthest_point_sampling_gpu(points, n_samples, device='cuda'):
    """
    GPU加速的最远点采样
    使用PyTorch CUDA实现并行计算
    """
    # 转换为PyTorch张量并移至GPU
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float().to(device)
    
    N, _ = points.shape
    
    if N <= n_samples:
        return torch.arange(N, device=device).cpu().numpy()
    
    # 初始化
    sampled_indices = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)
    
    # 第一个点随机选择
    sampled_indices[0] = torch.randint(0, N, (1,), device=device).item()
    
    # 迭代采样
    for i in range(1, n_samples):
        # 选择距离最大的点
        next_idx = torch.argmax(distances)
        sampled_indices[i] = next_idx
        
        # 计算所有点到新采样点的距离
        new_distances = torch.sum((points - points[next_idx:next_idx+1]) ** 2, dim=1)
        # 更新距离矩阵，保留较小值
        distances = torch.min(distances, new_distances)
    
    return sampled_indices.cpu().numpy()

# GPU加速的特征加权最远点采样函数
def feature_fps_gpu(points, features, n_samples, device='cuda'):
    """
    GPU加速的特征加权最远点采样
    使用PyTorch CUDA实现并行计算
    """
    # 转换为PyTorch张量并移至GPU
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).float().to(device)
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float().to(device)
    
    N, _ = points.shape
    n_samples = min(n_samples, N)
    
    # 初始化
    sampled_indices = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)
    is_selected = torch.zeros(N, dtype=torch.bool, device=device)
    
    # 第一个点：选特征分数最高的
    first_idx = torch.argmax(features)
    sampled_indices[0] = first_idx
    is_selected[first_idx] = True
    
    # 计算初始距离
    distances = torch.sum((points - points[first_idx:first_idx+1]) ** 2, dim=1)
    
    # 迭代采样
    for i in range(1, n_samples):
        # 计算加权距离（避开已选择的点）
        weighted_distances = distances * (1 + features)
        weighted_distances[is_selected] = -1.0  # 排除已选中的点
        
        # 选择加权距离最大的点
        next_idx = torch.argmax(weighted_distances)
        sampled_indices[i] = next_idx
        is_selected[next_idx] = True
        
        # 更新距离
        new_distances = torch.sum((points - points[next_idx:next_idx+1]) ** 2, dim=1)
        distances = torch.min(distances, new_distances)
    
    return sampled_indices.cpu().numpy()

# GPU版本的角点检测器
class GPUCornerDetector:
    """
    GPU加速的角点检测器
    使用PyTorch CUDA实现并行计算
    """
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def detect(self, points, k=20):
        L, P, S, C = _structure_scores(np.asarray(points, dtype=np.float32), k, device=self.device)
        return S * (1.0 - C) * (1.0 - L)

# GPU版本的边缘检测器
class GPUEdgeDetector:
    """
    GPU加速的边缘检测器
    使用PyTorch CUDA实现并行计算
    """
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def detect(self, points, k=20, corner_scores=None):
        L, P, S, C = _structure_scores(np.asarray(points, dtype=np.float32), k, device=self.device)
        if corner_scores is not None:
            S = np.asarray(corner_scores)
            C = C * (1 - S * 0.5)
        return L * (1.0 - C) * (1.0 - 0.5 * P)

# GPU加速的分层结构保留采样器
class GPUHierarchicalStructurePreservingSampling:
    """
    GPU加速的分层结构保留采样器
    使用PyTorch CUDA进行并行计算，大幅提升性能
    """
    
    def __init__(self, target_num_points=1000, corner_ratio=0.30, edge_ratio=0.50, plane_ratio=0.20, device='cuda'):
        """
        初始化GPU加速的分层结构保留采样器
        
        Args:
            device: 计算设备，默认为'cuda'（GPU）
        """
        self.target_num = target_num_points
        self.corner_ratio = corner_ratio
        self.edge_ratio = edge_ratio
        self.plane_ratio = plane_ratio
        self.device = device
        
        # 使用GPU版本的检测器
        self.corner_detector = GPUCornerDetector(device=device)
        self.edge_detector = GPUEdgeDetector(device=device)
    
    def sample(self, points, return_indices=False):
        points_np = np.asarray(points, dtype=np.float32)
        N = points_np.shape[0]
        if N <= self.target_num:
            return points_np, np.arange(N), np.zeros(N, dtype=int)
        k = min(20, max(10, int(N * 0.01)))
        L, P, S = _structure_scores(points_np, k, device=self.device)
        corner_order = np.argsort(S)[::-1]
        edge_order = np.argsort(L)[::-1]
        plane_order = np.argsort(P)[::-1]
        n_corner = min(int(self.target_num * self.corner_ratio), len(corner_order))
        n_edge = min(int(self.target_num * self.edge_ratio), len(edge_order))
        n_plane = min(int(self.target_num * self.plane_ratio), len(plane_order))
        corner_pool = corner_order
        edge_pool = edge_order
        edge_pool = edge_pool[~np.isin(edge_pool, corner_pool[:n_corner])]
        plane_pool = plane_order
        plane_pool = plane_pool[~np.isin(plane_pool, corner_pool[:n_corner])]
        plane_pool = plane_pool[~np.isin(plane_pool, edge_pool[:n_edge])]
        if n_corner > 0:
            c_scores = np.maximum(S[corner_pool], 0)
            if len(corner_pool) <= n_corner or c_scores.sum() == 0:
                c_sel = corner_pool[:n_corner]
            else:
                probs = c_scores / c_scores.sum()
                idx_local = np.random.choice(len(corner_pool), n_corner, replace=False, p=probs)
                c_sel = corner_pool[idx_local]
        else:
            c_sel = np.array([], dtype=np.int64)
        if n_edge > 0:
            if len(edge_pool) <= n_edge:
                e_sel = edge_pool
            else:
                e_local = feature_fps_gpu(points_np[edge_pool], L[edge_pool], n_edge, device=self.device)
                e_sel = edge_pool[e_local]
        else:
            e_sel = np.array([], dtype=np.int64)
        if n_plane > 0:
            if len(plane_pool) <= n_plane:
                p_sel = plane_pool
            else:
                p_local = farthest_point_sampling_gpu(points_np[plane_pool], n_plane, device=self.device)
                p_sel = plane_pool[p_local]
        else:
            p_sel = np.array([], dtype=np.int64)
        sel = np.concatenate([c_sel, e_sel, p_sel])
        if sel.shape[0] < self.target_num:
            remaining = np.setdiff1d(np.arange(N), sel)
            add_cnt = self.target_num - sel.shape[0]
            add_local = farthest_point_sampling_gpu(points_np[remaining], min(add_cnt, remaining.shape[0]), device=self.device)
            sel = np.concatenate([sel, remaining[add_local]])
        pts = points_np[sel]
        types = np.concatenate([np.full(c_sel.shape[0], 2), np.full(e_sel.shape[0], 1), np.full(p_sel.shape[0], 0)])
        return pts, sel, types
    
    def _dynamic_quota_allocation(self, n_corner_available, n_edge_available, n_plane_available):
        """
        强制配额分配：确保最终结果接近预设比例
        """
        # 计算理想配额
        ideal_corner = int(self.target_num * self.corner_ratio)
        ideal_edge = int(self.target_num * self.edge_ratio)
        ideal_plane = self.target_num - ideal_corner - ideal_edge
        
        # 可用点数
        total_available = n_corner_available + n_edge_available + n_plane_available
        
        # 最低保证数量
        min_corner_guaranteed = max(50, int(self.target_num * 0.1))
        min_edge_guaranteed = max(200, int(self.target_num * 0.3))
        
        # 步骤1: 优先保证角点和边缘点的最低数量
        actual_corner = min(ideal_corner, n_corner_available)
        actual_edge = min(ideal_edge, n_edge_available)
        actual_plane = min(ideal_plane, n_plane_available)
        
        # 确保角点达到最低保证数量
        if actual_corner < min_corner_guaranteed and n_corner_available > actual_corner:
            needed = min_corner_guaranteed - actual_corner
            actual_corner += min(needed, n_corner_available - actual_corner)
        
        # 确保边缘点达到最低保证数量
        if actual_edge < min_edge_guaranteed and n_edge_available > actual_edge:
            needed = min_edge_guaranteed - actual_edge
            actual_edge += min(needed, n_edge_available - actual_edge)
        
        # 步骤2: 计算当前已分配点数和剩余点数
        allocated = actual_corner + actual_edge + actual_plane
        remaining = self.target_num - allocated
        
        # 步骤3: 如果还有剩余点数，按优先级分配
        if remaining > 0:
            # 优先级顺序：角点 > 边缘 > 平面
            # 1. 先尝试增加角点
            max_corner_extra = int(ideal_corner * 1.5)
            if n_corner_available > actual_corner:
                add = min(remaining, max_corner_extra - actual_corner, n_corner_available - actual_corner)
                actual_corner += add
                remaining -= add
            
            # 2. 再尝试增加边缘点
            max_edge_extra = int(ideal_edge * 1.3)
            if remaining > 0 and n_edge_available > actual_edge:
                add = min(remaining, max_edge_extra - actual_edge, n_edge_available - actual_edge)
                actual_edge += add
                remaining -= add
            
            # 3. 最后分配给平面点
            if remaining > 0 and n_plane_available > actual_plane:
                add = min(remaining, n_plane_available - actual_plane)
                actual_plane += add
        
        # 步骤4: 如果角点或边缘点仍然不足，从平面点中借用
        total_assigned = actual_corner + actual_edge + actual_plane
        if total_assigned < self.target_num:
            # 确保总数达到目标
            shortage = self.target_num - total_assigned
            actual_plane += shortage
        
        # 确保不会超出可用数量
        actual_corner = min(actual_corner, n_corner_available)
        actual_edge = min(actual_edge, n_edge_available)
        actual_plane = min(actual_plane, n_plane_available)
        
        # 最终调整确保总数正确
        total_assigned = actual_corner + actual_edge + actual_plane
        if total_assigned > self.target_num:
            # 如果超出了，从平面点中减少
            excess = total_assigned - self.target_num
            actual_plane = max(0, actual_plane - excess)
        elif total_assigned < self.target_num:
            # 如果还不够，再从各类型中按比例补充
            shortage = self.target_num - total_assigned
            if n_corner_available > actual_corner:
                add = min(shortage // 3, n_corner_available - actual_corner)
                actual_corner += add
                shortage -= add
            if shortage > 0 and n_edge_available > actual_edge:
                add = min(shortage // 2, n_edge_available - actual_edge)
                actual_edge += add
                shortage -= add
            if shortage > 0 and n_plane_available > actual_plane:
                actual_plane += min(shortage, n_plane_available - actual_plane)
        
        return actual_corner, actual_edge, actual_plane
    
    def _sample_corners(self, corner_points, corner_scores, n_target):
        """
        角点采样：基于重要性 - GPU加速版本
        """
        n_corners = len(corner_points)

        if n_corners <= n_target:
            # 全部保留
            return np.arange(n_corners)
        else:
            # 按分数加权采样
            # 确保分数非负且避免除零错误
            scores = np.maximum(corner_scores, 0)
            total = scores.sum()
            
            if total == 0:
                # 如果所有分数都为0，使用均匀分布
                probs = np.ones_like(scores) / len(scores)
            else:
                probs = scores / total
                
            sampled = np.random.choice(
                n_corners,
                n_target,
                replace=False,
                p=probs
            )
            return sampled

    def _sample_edges(self, edge_points, edge_scores, n_target):
        """
        边缘采样：使用GPU加速的Feature-FPS
        """
        n_edges = len(edge_points)

        if n_edges <= n_target:
            return np.arange(n_edges)
        else:
            # 使用GPU加速的特征FPS
            sampled = feature_fps_gpu(
                edge_points,
                edge_scores,
                n_target,
                device=self.device
            )
            return sampled

    def _sample_planes(self, plane_points, n_target):
        """
        平面采样：使用GPU加速的标准FPS
        """
        n_planes = len(plane_points)

        if n_planes <= n_target:
            return np.arange(n_planes)
        else:
            # 使用GPU加速的FPS
            sampled = farthest_point_sampling_gpu(
                plane_points,
                n_target,
                device=self.device
            )
            return sampled
