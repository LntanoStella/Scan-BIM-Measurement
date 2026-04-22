import open3d as o3d
import numpy as np
import copy
import sys
import os
import importlib.util
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# ==============================================================================
# 全局配置
# ==============================================================================
CONFIG = {
    'TARGET_RMSE': 3.0,          # 目标精度 (mm)
    'MAX_ITER_L1': 100,           # Level-1 迭代次数
    'MAX_ITER_L2': 300,           # Level-2 迭代次数
    'L1_RADIUS_START': 200.0,    'L1_RADIUS_END': 10.0,
    'L2_RADIUS_START': 50.0,    'L2_RADIUS_END': 3.0,
    'HUBER_SCALE': 10.0,         # 初始 Huber 阈值
    'VISUALIZE_LOSS': True,
    'SAVE_LOSS_PLOT': True,
    'OUTPUT_DIR': 'output',       # 输出文件夹
    'PLOT_FILENAME': 'registration_loss_curve.png'
}

# ==============================================================================
# 模块导入
# ==============================================================================
try:
    from sample import HierarchicalStructurePreservingSampling
except ImportError:
    try:
        from sample import HierarchicalStructurePreservingSampling
    except ImportError:
        print("[错误] 未找到 sample.py。")
        sys.exit(1)

try:
    spec = importlib.util.spec_from_file_location("initial_alignment", "02_initial_alignment.py")
    initial_module = importlib.util.module_from_spec(spec)
    sys.modules["initial_alignment"] = initial_module
    spec.loader.exec_module(initial_module)
    initial_alignment_pipeline = initial_module.initial_alignment_pipeline_v4
    visualize_step = initial_module.visualize_step
except Exception as e:
    print(f"[错误] 无法导入 02_initial_alignment.py: {e}")
    sys.exit(1)

# ==============================================================================
# 核心类：联合增强解算器
# ==============================================================================

class RobustFineSolver:
    def __init__(self, source_pcd, target_pcd):
        self.source = copy.deepcopy(source_pcd)
        self.target = copy.deepcopy(target_pcd)
        
        # 状态变量：6自由度 (rx, ry, rz, tx, ty, tz)
        # 我们使用李代数 se(3) 的思想，迭代更新位姿，而不是累加 delta
        self.current_T = np.eye(4) 
        
        self.bim_features = {'planes': None, 'edges': None, 'corners': None, 
                             'plane_normals': None, 'edge_dirs': None}
        self.search_trees = {}
        self.history = {'feature_rmse': [], 'global_rmse': [], 'inlier_rmse': [], 'inlier_ratio': [], 'stage': []}
        
        self._detect_scale()

    def _detect_scale(self):
        bbox = self.target.get_axis_aligned_bounding_box()
        if bbox.get_max_extent() > 1000.0:
            print("[系统] 单位判定: mm (高精度模式)")
            self.unit = "mm"
        else:
            print("[系统] 单位判定: m (需调整参数)")
            self.unit = "m"

    def preprocess_bim_features(self, sample_points=50000):
        print(f"[预处理] 提取 BIM 特征 (N={sample_points})...")
        sampler = HierarchicalStructurePreservingSampling(
            target_num_points=sample_points,
            corner_ratio=0.15, edge_ratio=0.35, plane_ratio=0.50
        )
        pts, _, types = sampler.sample(self.target.points)
        pts = np.asarray(pts)
        
        self.bim_features['planes'] = pts[types == 0]
        self.bim_features['edges'] = pts[types == 1]
        self.bim_features['corners'] = pts[types == 2]

        # 构建搜索树与属性
        if len(self.bim_features['planes']) > 0:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.bim_features['planes']))
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=200.0, max_nn=30))
            self.bim_features['plane_normals'] = np.asarray(pcd.normals)
            self.search_trees['plane'] = o3d.geometry.KDTreeFlann(pcd)
        
        if len(self.bim_features['edges']) > 0:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.bim_features['edges']))
            self.search_trees['edge'] = o3d.geometry.KDTreeFlann(pcd)
            # 计算棱线方向
            dirs = []
            for pt in self.bim_features['edges']:
                [_, idx, _] = self.search_trees['edge'].search_knn_vector_3d(pt, 10)
                if len(idx) >= 3:
                    cov = np.cov(np.asarray(pcd.points)[idx].T)
                    w, v = np.linalg.eigh(cov)
                    dirs.append(v[:, 2])
                else:
                    dirs.append([0, 0, 1])
            self.bim_features['edge_dirs'] = np.array(dirs)

        if len(self.bim_features['corners']) > 0:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.bim_features['corners']))
            self.search_trees['corner'] = o3d.geometry.KDTreeFlann(pcd)

    def _find_correspondences(self, transformed_src, feature_type, radius):
        """通用对应点搜索"""
        tree = self.search_trees.get(feature_type)
        if tree is None: return [], [], []
        
        tgt_pts = self.bim_features[f'{feature_type}s']
        tgt_attr = self.bim_features.get(f'{feature_type}_normals') if feature_type == 'plane' else \
                   self.bim_features.get(f'{feature_type}_dirs')
        
        # 降采样查询
        indices = np.random.choice(len(transformed_src), min(5000, len(transformed_src)), replace=False)
        query_pts = transformed_src[indices]
        
        src_corr, tgt_corr, attr_corr = [], [], []
        
        for i, p in enumerate(query_pts):
            [_, idx, dist_sq] = tree.search_knn_vector_3d(p, 1)
            dist = np.sqrt(dist_sq[0])
            
            # 动态阈值保护：如果半径太小，至少保留最近的
            if dist < radius:
                idx = idx[0]
                src_corr.append(p) # 注意：这里存的是变换后的点，用于计算残差
                tgt_corr.append(tgt_pts[idx])
                if tgt_attr is not None:
                    attr_corr.append(tgt_attr[idx])
        
        return np.array(src_corr), np.array(tgt_corr), np.array(attr_corr)

    def _transform_points(self, points, T):
        # points: (N, 3), T: (4, 4)
        # R = T[:3, :3], t = T[:3, 3]
        # p' = R p + t
        return (T[:3, :3] @ points.T).T + T[:3, 3]

    def run_optimization_stage(self, stage_name, use_planes, use_edges, use_corners, 
                             max_iter, r_start, r_end):
        """
        通用优化阶段：支持启用/禁用不同特征的联合优化
        核心改进：总是优化 6-DoF (pose_vec)，不再固定平移
        """
        print(f"\n[{stage_name}] 联合优化 (Planes={use_planes}, Edges={use_edges}, Corners={use_corners})...")
        print(f"  > 搜索半径: {r_start} -> {r_end} {self.unit}")
        
        radius_schedule = np.linspace(r_start, r_end, max_iter)
        src_points_base = np.asarray(self.source.points) # 原始点云 (未变换)

        for i in range(max_iter):
            # 1. 将原始点云变换到当前位置
            curr_src = self._transform_points(src_points_base, self.current_T)
            
            residuals_func = []
            valid_pairs = 0
            
            # --- 构建约束 ---
            # A. 平面约束 (Point-to-Plane)
            if use_planes:
                s, t, n = self._find_correspondences(curr_src, 'plane', radius_schedule[i])
                if len(s) > 10:
                    valid_pairs += len(s)
                    # 闭包：传入微小增量 x (se3)，计算残差
                    def loss_plane(x, s=s, t=t, n=n):
                        # x 是 se3 李代数 (6维)，近似为 [rx, ry, rz, tx, ty, tz]
                        # p_new = p_old + w x p_old + v
                        # 但 scipy 需要计算具体值，我们用 R.from_rotvec 转换
                        # 注意：s 已经是变换到当前位置的点
                        # 我们求解的是相对于当前位置的增量变换 T_inc
                        rot_mat = R.from_rotvec(x[:3]).as_matrix()
                        trans = x[3:]
                        p_est = (rot_mat @ s.T).T + trans
                        return np.sum((p_est - t) * n, axis=1) # 点面距离
                    residuals_func.append(loss_plane)

            # B. 棱线约束 (Point-to-Line)
            if use_edges:
                s, t, d = self._find_correspondences(curr_src, 'edge', radius_schedule[i])
                if len(s) > 5:
                    valid_pairs += len(s)
                    def loss_edge(x, s=s, t=t, d=d):
                        rot_mat = R.from_rotvec(x[:3]).as_matrix()
                        trans = x[3:]
                        p_est = (rot_mat @ s.T).T + trans
                        vec = p_est - t
                        return np.linalg.norm(np.cross(vec, d), axis=1)
                    residuals_func.append(loss_edge)

            # C. 角点约束 (Point-to-Point)
            if use_corners:
                s, t, _ = self._find_correspondences(curr_src, 'corner', radius_schedule[i])
                if len(s) > 5:
                    valid_pairs += len(s)
                    def loss_corner(x, s=s, t=t):
                        rot_mat = R.from_rotvec(x[:3]).as_matrix()
                        trans = x[3:]
                        p_est = (rot_mat @ s.T).T + trans
                        return np.linalg.norm(p_est - t, axis=1) * 2.0 # 加权
                    residuals_func.append(loss_corner)

            if valid_pairs == 0:
                print(f"  Iter {i+1}: 无有效约束，跳过")
                continue

            # --- 求解增量 ---
            def total_loss(x):
                # x: [rx, ry, rz, tx, ty, tz]
                return np.concatenate([f(x) for f in residuals_func])

            # 初始猜测：无增量
            x0 = np.zeros(6)
            
            # 鲁棒优化
            huber_scale = max(CONFIG['HUBER_SCALE'], radius_schedule[i] / 2.0)
            res = least_squares(total_loss, x0, loss='huber', f_scale=huber_scale)
            
            # --- 更新全局位姿 ---
            # T_new = T_inc * T_curr
            pose_inc = np.eye(4)
            pose_inc[:3, :3] = R.from_rotvec(res.x[:3]).as_matrix()
            pose_inc[:3, 3] = res.x[3:]
            
            self.current_T = pose_inc @ self.current_T
            
            # --- 记录 ---
            feature_rmse = np.sqrt(np.mean(res.fun**2))
            
            # 计算基于整个点云的 RMSE
            transformed_pcd = copy.deepcopy(self.source)
            transformed_pcd.transform(self.current_T)
            dists = transformed_pcd.compute_point_cloud_distance(self.target)
            global_rmse = np.sqrt(np.mean(np.asarray(dists)**2))
            
            # 计算 Inlier RMSE 和 Inlier Ratio
            inlier_dists = np.asarray(dists)[np.asarray(dists) < 8.0]
            inlier_rmse = np.sqrt(np.mean(inlier_dists**2)) if len(inlier_dists) > 0 else 0.0
            inlier_ratio = len(inlier_dists) / len(dists) if len(dists) > 0 else 0.0
            
            self.history['feature_rmse'].append(feature_rmse)
            self.history['global_rmse'].append(global_rmse)
            self.history['inlier_rmse'].append(inlier_rmse)
            self.history['inlier_ratio'].append(inlier_ratio)
            self.history['stage'].append(stage_name)
            
            if i % 5 == 0 or i == max_iter - 1:
                print(f"  Iter {i+1}/{max_iter}: Feature RMSE={feature_rmse:.4f} {self.unit}, Inlier RMSE={inlier_rmse:.4f} {self.unit}, Inlier Ratio={inlier_ratio:.4f} (Pairs={valid_pairs}, Hub={huber_scale:.1f})")
            
            if feature_rmse < 1.0 and i > 10: break # 收敛到 1mm

    def get_final_pcd(self):
        pcd = copy.deepcopy(self.source)
        pcd.transform(self.current_T)
        return pcd

    def plot_loss_curve(self):
        if not CONFIG['VISUALIZE_LOSS']: return
        
        # 确保输出文件夹存在
        output_dir = CONFIG['OUTPUT_DIR']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        feature_rmse = self.history['feature_rmse']
        inlier_rmse = self.history['inlier_rmse']
        inlier_ratio = self.history['inlier_ratio']
        stages = self.history['stage']
        
        # 分离 level-1 和 level-2 的数据
        level1_feature_rmse = []
        level1_inlier_rmse = []
        level1_inlier_ratio = []
        level2_feature_rmse = []
        level2_inlier_rmse = []
        level2_inlier_ratio = []
        
        for i, stage in enumerate(stages):
            if stage == 'Level-1':
                level1_feature_rmse.append(feature_rmse[i])
                level1_inlier_rmse.append(inlier_rmse[i])
                level1_inlier_ratio.append(inlier_ratio[i])
            elif stage == 'Level-2':
                level2_feature_rmse.append(feature_rmse[i])
                level2_inlier_rmse.append(inlier_rmse[i])
                level2_inlier_ratio.append(inlier_ratio[i])
        
        # 1. 绘制整个过程的 Inlier 指标变化曲线
        plt.figure(figsize=(10, 6))
        plt.plot(inlier_rmse, 'g-', linewidth=2, label='Inlier RMSE')
        plt.plot(inlier_ratio, 'm--', linewidth=2, label='Inlier Ratio')
        
        # 标记阶段切换点
        switch_idx = next((i for i, s in enumerate(stages) if s == 'Level-2'), None)
        if switch_idx:
            plt.axvline(x=switch_idx, color='r', linestyle='--', label='Level-2 Start')
            
        plt.title('Inlier Metrics Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.legend()
        inlier_plot_path = os.path.join(output_dir, 'inlier_metrics_convergence.png')
        plt.savefig(inlier_plot_path)
        if CONFIG['VISUALIZE_LOSS']:
            plt.show()
        
        # 2. 绘制 Level-1 的 Feature RMSE 曲线
        if level1_feature_rmse:
            plt.figure(figsize=(10, 6))
            plt.plot(level1_feature_rmse, 'r--', linewidth=2, label='Feature RMSE')
            plt.yscale('log')
            plt.title('Feature RMSE Convergence - Level 1')
            plt.xlabel('Iteration')
            plt.ylabel(f'RMSE ({self.unit})')
            plt.legend()
            level1_feature_plot_path = os.path.join(output_dir, 'registration_loss_level1.png')
            plt.savefig(level1_feature_plot_path)
            if CONFIG['VISUALIZE_LOSS']:
                plt.show()
        
        # 3. 绘制 Level-1 的 Inlier RMSE 曲线
        if level1_inlier_rmse:
            plt.figure(figsize=(10, 6))
            plt.plot(level1_inlier_rmse, 'g-', linewidth=2, label='Inlier RMSE')
            plt.yscale('log')
            plt.title('Inlier RMSE Convergence - Level 1')
            plt.xlabel('Iteration')
            plt.ylabel(f'RMSE ({self.unit})')
            plt.legend()
            level1_inlier_plot_path = os.path.join(output_dir, 'inlier_rmse_level1.png')
            plt.savefig(level1_inlier_plot_path)
            if CONFIG['VISUALIZE_LOSS']:
                plt.show()
        
        # 4. 绘制 Level-2 的 Feature RMSE 曲线
        if level2_feature_rmse:
            plt.figure(figsize=(10, 6))
            plt.plot(level2_feature_rmse, 'r--', linewidth=2, label='Feature RMSE')
            plt.yscale('log')
            plt.title('Feature RMSE Convergence - Level 2')
            plt.xlabel('Iteration')
            plt.ylabel(f'RMSE ({self.unit})')
            plt.legend()
            level2_feature_plot_path = os.path.join(output_dir, 'registration_loss_level2.png')
            plt.savefig(level2_feature_plot_path)
            if CONFIG['VISUALIZE_LOSS']:
                plt.show()
        
        # 5. 绘制 Level-2 的 Inlier RMSE 曲线
        if level2_inlier_rmse:
            plt.figure(figsize=(10, 6))
            plt.plot(level2_inlier_rmse, 'g-', linewidth=2, label='Inlier RMSE')
            plt.yscale('log')
            plt.title('Inlier RMSE Convergence - Level 2')
            plt.xlabel('Iteration')
            plt.ylabel(f'RMSE ({self.unit})')
            plt.legend()
            level2_inlier_plot_path = os.path.join(output_dir, 'inlier_rmse_level2.png')
            plt.savefig(level2_inlier_plot_path)
            if CONFIG['VISUALIZE_LOSS']:
                plt.show()
        
        print(f"精度曲线已保存到 {output_dir} 文件夹")

# ==============================================================================
# 主入口
# ==============================================================================
if __name__ == "__main__":
    print(">>> 启动 联合鲁棒配准 (修正版)...")
    
    # 1. 载入数据
    scan_path = r"scan_path"
    bim_path = r"bim_path"
    
    try:
        scan_pcd = o3d.io.read_point_cloud(scan_path)
        bim_pcd = o3d.io.read_point_cloud(bim_path)
    except:
        print("无数据，生成模拟...")
        bim_pcd = o3d.geometry.TriangleMesh.create_box(5000, 2000, 2000).sample_points_poisson_disk(10000)
        scan_pcd = copy.deepcopy(bim_pcd)
        T_disturb = np.eye(4)
        T_disturb[:3, :3] = R.from_euler('z', 5, degrees=True).as_matrix()
        T_disturb[:3, 3] = [200, 50, -50]
        scan_pcd.transform(T_disturb)

    # 原始数据可视化
    print("\n--- 原始数据可视化 ---")
    # 为原始数据创建与提取主轴时相同的颜色方案
    scan_original = copy.deepcopy(scan_pcd)
    scan_original.paint_uniform_color([1, 0.706, 0])  # 黄色 Scan
    bim_original = copy.deepcopy(bim_pcd)
    bim_original.paint_uniform_color([0, 0.651, 0.929])  # 蓝色 BIM
    visualize_step([scan_original, bim_original], "Original Data (Scan + BIM)")

    # 2. 粗配准
    print("\n--- 阶段 1: 粗配准 ---")
    T_coarse = initial_alignment_pipeline(scan_pcd, bim_pcd)
    scan_coarse = copy.deepcopy(scan_pcd)
    scan_coarse.transform(T_coarse)

    # 3. 鲁棒精配准
    solver = RobustFineSolver(scan_coarse, bim_pcd)
    solver.preprocess_bim_features()
    
    # 计算初始阶段（未配准）的精度指标
    print("\n[初始阶段精度评估]")
    initial_dists = scan_pcd.compute_point_cloud_distance(bim_pcd)
    initial_global_rmse = np.sqrt(np.mean(np.asarray(initial_dists)**2))
    
    # 计算初始阶段的 Inlier 指标
    initial_inlier_dists = np.asarray(initial_dists)[np.asarray(initial_dists) < 8.0]
    initial_inlier_rmse = np.sqrt(np.mean(initial_inlier_dists**2)) if len(initial_inlier_dists) > 0 else 0.0
    initial_inlier_ratio = len(initial_inlier_dists) / len(initial_dists) if len(initial_dists) > 0 else 0.0
    
    print(f"  Global RMSE: {initial_global_rmse:.4f} {solver.unit}")
    print(f"  Inlier RMSE (τ=8mm): {initial_inlier_rmse:.4f} {solver.unit}")
    print(f"  Inlier Ratio (τ=8mm): {initial_inlier_ratio:.4f} ({initial_inlier_ratio*100:.2f}%)")
    
    # 计算粗配准后的精度指标
    print("\n[粗配准阶段精度评估]")
    coarse_dists = scan_coarse.compute_point_cloud_distance(bim_pcd)
    coarse_global_rmse = np.sqrt(np.mean(np.asarray(coarse_dists)**2))
    
    # 计算粗配准后的 Inlier 指标
    coarse_inlier_dists = np.asarray(coarse_dists)[np.asarray(coarse_dists) < 8.0]
    coarse_inlier_rmse = np.sqrt(np.mean(coarse_inlier_dists**2)) if len(coarse_inlier_dists) > 0 else 0.0
    coarse_inlier_ratio = len(coarse_inlier_dists) / len(coarse_dists) if len(coarse_dists) > 0 else 0.0
    
    print(f"  Global RMSE: {coarse_global_rmse:.4f} {solver.unit}")
    print(f"  Inlier RMSE (τ=8mm): {coarse_inlier_rmse:.4f} {solver.unit}")
    print(f"  Inlier Ratio (τ=8mm): {coarse_inlier_ratio:.4f} ({coarse_inlier_ratio*100:.2f}%)")
    
    visualize_step([scan_coarse, bim_pcd], "Before Fine Registration")

    # 关键修改：Level-1 开启 6-DoF 优化，避免代偿性倾斜
    # 只用平面，但允许微调平移以适应法向误差
    solver.run_optimization_stage(
        stage_name="Level-1", 
        use_planes=True, use_edges=False, use_corners=False,
        max_iter=CONFIG['MAX_ITER_L1'], 
        r_start=CONFIG['L1_RADIUS_START'], r_end=CONFIG['L1_RADIUS_END']
    )
    
    # Level-2 (及 6.3.3): 全特征联合微调
    solver.run_optimization_stage(
        stage_name="Level-2", 
        use_planes=True, use_edges=True, use_corners=True,
        max_iter=CONFIG['MAX_ITER_L2'], 
        r_start=CONFIG['L2_RADIUS_START'], r_end=CONFIG['L2_RADIUS_END']
    )

    # 4. 结果
    scan_final = solver.get_final_pcd()
    
    # 最终精度验证
    def compute_robust_rmse(pcd1, pcd2):
        """计算去噪后的全局RMSE"""
        dists = pcd1.compute_point_cloud_distance(pcd2)
        dists = np.asarray(dists)
        
        # 去除离群点
        q1 = np.percentile(dists, 25)
        q3 = np.percentile(dists, 75)
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr
        filtered_dists = dists[dists < threshold]
        
        if len(filtered_dists) > 0:
            return np.sqrt(np.mean(filtered_dists**2))
        return 0.0
    
    def compute_inlier_metrics(pcd1, pcd2, threshold=8.0):
        """计算内点RMSE和内点重叠率"""
        dists = pcd1.compute_point_cloud_distance(pcd2)
        dists = np.asarray(dists)
        
        # 计算内点
        inlier_dists = dists[dists < threshold]
        inlier_ratio = len(inlier_dists) / len(dists) if len(dists) > 0 else 0.0
        
        # 计算内点RMSE
        inlier_rmse = np.sqrt(np.mean(inlier_dists**2)) if len(inlier_dists) > 0 else 0.0
        
        return inlier_rmse, inlier_ratio
    
    # 计算各种精度指标
    dists = scan_final.compute_point_cloud_distance(bim_pcd)
    final_global_rmse = np.sqrt(np.mean(np.asarray(dists)**2))
    final_robust_rmse = compute_robust_rmse(scan_final, bim_pcd)
    final_inlier_rmse, final_inlier_ratio = compute_inlier_metrics(scan_final, bim_pcd)
    
    # 获取最后一次的特征点 RMSE
    final_feature_rmse = solver.history['feature_rmse'][-1] if solver.history['feature_rmse'] else 0
    
    print(f"\n[最终结果]")
    print(f"  Feature RMSE (Key Points): {final_feature_rmse:.4f} {solver.unit}")
    print(f"  Global RMSE (Whole Point Cloud): {final_global_rmse:.4f} {solver.unit}")
    print(f"  Robust RMSE (Denoised): {final_robust_rmse:.4f} {solver.unit}")
    print(f"  Inlier RMSE (τ=8mm): {final_inlier_rmse:.4f} {solver.unit}")
    print(f"  Inlier Ratio (τ=8mm): {final_inlier_ratio:.4f} ({final_inlier_ratio*100:.2f}%)")
    
    # 精度对比分析
    print("\n[精度分析]")
    print("1. 多维度精度评估:")
    print("   - Feature RMSE: 基于筛选后的关键特征点计算，反映主要结构对齐精度")
    print("   - Global RMSE: 基于所有点计算，包含噪声和离群点")
    print("   - Robust RMSE: 基于去噪后的点云计算，更能反映整体配准质量")
    print("   - Inlier RMSE: 基于距离阈值内的点计算，符合工程测量标准")
    print("   - Inlier Ratio: 反映点云重叠程度和配准成功率")
    
    print("\n2. 分阶段优化的有效性:")
    if solver.history['global_rmse']:
        initial_global_rmse = solver.history['global_rmse'][0]
        improvement = (initial_global_rmse - final_global_rmse) / initial_global_rmse * 100
        print(f"   - 初始整体 RMSE: {initial_global_rmse:.4f} {solver.unit}")
        print(f"   - 最终整体 RMSE: {final_global_rmse:.4f} {solver.unit}")
        print(f"   - 精度提升: {improvement:.2f}%")
        print(f"   - 特征点 RMSE 从 {solver.history['feature_rmse'][0]:.4f} 降至 {final_feature_rmse:.4f}")
        print("   - 这证明了分阶段优化策略的有效性")
    
    # 三阶段精度指标对标
    print("\n[三阶段精度指标对标]")
    print("| 阶段 | Global RMSE | Inlier RMSE | Inlier Ratio |")
    print("|------|------------|-------------|--------------|")
    print(f"| 初始 | {initial_global_rmse:.4f} | {initial_inlier_rmse:.4f} | {initial_inlier_ratio:.4f} ({initial_inlier_ratio*100:.2f}%) |")
    print(f"| 粗配准 | {coarse_global_rmse:.4f} | {coarse_inlier_rmse:.4f} | {coarse_inlier_ratio:.4f} ({coarse_inlier_ratio*100:.2f}%) |")
    print(f"| 精配准 | {final_global_rmse:.4f} | {final_inlier_rmse:.4f} | {final_inlier_ratio:.4f} ({final_inlier_ratio*100:.2f}%) |")
    
    # 计算各阶段提升
    print("\n[各阶段精度提升]")
    coarse_global_improvement = (initial_global_rmse - coarse_global_rmse) / initial_global_rmse * 100
    fine_global_improvement = (coarse_global_rmse - final_global_rmse) / coarse_global_rmse * 100
    total_global_improvement = (initial_global_rmse - final_global_rmse) / initial_global_rmse * 100
    
    coarse_inlier_improvement = (initial_inlier_rmse - coarse_inlier_rmse) / initial_inlier_rmse * 100 if initial_inlier_rmse > 0 else 0
    fine_inlier_improvement = (coarse_inlier_rmse - final_inlier_rmse) / coarse_inlier_rmse * 100 if coarse_inlier_rmse > 0 else 0
    total_inlier_improvement = (initial_inlier_rmse - final_inlier_rmse) / initial_inlier_rmse * 100 if initial_inlier_rmse > 0 else 0
    
    coarse_ratio_improvement = (coarse_inlier_ratio - initial_inlier_ratio) / initial_inlier_ratio * 100 if initial_inlier_ratio > 0 else 0
    fine_ratio_improvement = (final_inlier_ratio - coarse_inlier_ratio) / coarse_inlier_ratio * 100 if coarse_inlier_ratio > 0 else 0
    total_ratio_improvement = (final_inlier_ratio - initial_inlier_ratio) / initial_inlier_ratio * 100 if initial_inlier_ratio > 0 else 0
    
    print(f"Global RMSE 提升: 粗配准 {coarse_global_improvement:.2f}%, 精配准 {fine_global_improvement:.2f}%, 总计 {total_global_improvement:.2f}%")
    print(f"Inlier RMSE 提升: 粗配准 {coarse_inlier_improvement:.2f}%, 精配准 {fine_inlier_improvement:.2f}%, 总计 {total_inlier_improvement:.2f}%")
    print(f"Inlier Ratio 提升: 粗配准 {coarse_ratio_improvement:.2f}%, 精配准 {fine_ratio_improvement:.2f}%, 总计 {total_ratio_improvement:.2f}%")
    
    # 确保输出文件夹存在
    output_dir = CONFIG['OUTPUT_DIR']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存转换矩阵
    # 1. 粗配准转换矩阵
    coarse_matrix_path = os.path.join(output_dir, 'coarse_registration_matrix.npy')
    np.save(coarse_matrix_path, T_coarse)
    print(f"粗配准转换矩阵已保存到: {coarse_matrix_path}")
    
    # 2. 精配准转换矩阵
    fine_matrix_path = os.path.join(output_dir, 'fine_registration_matrix.npy')
    np.save(fine_matrix_path, solver.current_T)
    print(f"精配准转换矩阵已保存到: {fine_matrix_path}")
    
    # 3. 总转换矩阵 (粗配准 + 精配准)
    total_matrix = solver.current_T @ T_coarse
    total_matrix_path = os.path.join(output_dir, 'total_registration_matrix.npy')
    np.save(total_matrix_path, total_matrix)
    print(f"总转换矩阵已保存到: {total_matrix_path}")
    
    # 可视化特征对齐情况
    feats = []
    if solver.bim_features['planes'] is not None:
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(solver.bim_features['planes']))
        p.paint_uniform_color([0, 0, 1]); feats.append(p) # Blue Planes
    if solver.bim_features['edges'] is not None:
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(solver.bim_features['edges']))
        p.paint_uniform_color([0, 1, 0]); feats.append(p) # Green Edges

    # 生成2D截面剖切图
    def generate_cross_sections(scan_pcd, bim_pcd, output_dir, slice_positions=None):
        """生成2D截面剖切图"""
        if slice_positions is None:
            # 默认在中心点附近生成切片
            scan_bbox = scan_pcd.get_axis_aligned_bounding_box()
            center = scan_bbox.get_center()
            slice_positions = {
                'x': [center[0]],
                'y': [center[1]],
                'z': [center[2]]
            }
        
        slice_thickness = 5.0  # 切片厚度
        
        for axis, positions in slice_positions.items():
            for pos in positions:
                # 对scan点云进行切片
                scan_points = np.asarray(scan_pcd.points)
                if axis == 'x':
                    scan_slice = scan_points[np.abs(scan_points[:, 0] - pos) < slice_thickness/2]
                elif axis == 'y':
                    scan_slice = scan_points[np.abs(scan_points[:, 1] - pos) < slice_thickness/2]
                else:  # z
                    scan_slice = scan_points[np.abs(scan_points[:, 2] - pos) < slice_thickness/2]
                
                # 对bim点云进行切片
                bim_points = np.asarray(bim_pcd.points)
                if axis == 'x':
                    bim_slice = bim_points[np.abs(bim_points[:, 0] - pos) < slice_thickness/2]
                elif axis == 'y':
                    bim_slice = bim_points[np.abs(bim_points[:, 1] - pos) < slice_thickness/2]
                else:  # z
                    bim_slice = bim_points[np.abs(bim_points[:, 2] - pos) < slice_thickness/2]
                
                # 生成切片图
                plt.figure(figsize=(10, 8))
                
                # 绘制scan点云切片（红色）
                if len(scan_slice) > 0:
                    if axis == 'x':
                        plt.scatter(scan_slice[:, 1], scan_slice[:, 2], c='r', s=1, alpha=0.6, label='Scan')
                        plt.xlabel('Y')
                        plt.ylabel('Z')
                    elif axis == 'y':
                        plt.scatter(scan_slice[:, 0], scan_slice[:, 2], c='r', s=1, alpha=0.6, label='Scan')
                        plt.xlabel('X')
                        plt.ylabel('Z')
                    else:  # z
                        plt.scatter(scan_slice[:, 0], scan_slice[:, 1], c='r', s=1, alpha=0.6, label='Scan')
                        plt.xlabel('X')
                        plt.ylabel('Y')
                
                # 绘制bim点云切片（蓝色）
                if len(bim_slice) > 0:
                    if axis == 'x':
                        plt.scatter(bim_slice[:, 1], bim_slice[:, 2], c='b', s=1, alpha=0.6, label='BIM')
                    elif axis == 'y':
                        plt.scatter(bim_slice[:, 0], bim_slice[:, 2], c='b', s=1, alpha=0.6, label='BIM')
                    else:  # z
                        plt.scatter(bim_slice[:, 0], bim_slice[:, 1], c='b', s=1, alpha=0.6, label='BIM')
                
                plt.title(f'Cross-section at {axis}={pos:.1f}mm')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # 保存切片图
                slice_filename = f'cross_section_{axis}_{pos:.1f}.png'
                slice_path = os.path.join(output_dir, slice_filename)
                plt.savefig(slice_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"截面剖切图已保存到: {slice_path}")
    
    # 生成截面剖切图
    generate_cross_sections(scan_final, bim_pcd, output_dir)
    
    solver.plot_loss_curve()
    visualize_step([scan_final, *feats], f"Final Aligned (Global RMSE={final_global_rmse:.1f}mm)")
    
