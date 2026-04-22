import open3d as o3d
import numpy as np
import copy
import time
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# 尝试导入 sample.py 中的特征采样类
try:
    from sample import HierarchicalStructurePreservingSampling
except ImportError:
    print("错误: 未找到 sample.py。请确保 sample.py 与本脚本在同一目录下。")
    exit(1)

class DecoupledSolver:
    """
    基于几何约束解耦的分级对齐解算器
    """
    def __init__(self, source_pcd, target_pcd, downsample_voxel=None):
        """
        初始化解算器
        :param source_pcd: 扫描点云 (待配准)
        :param target_pcd: BIM参考点云 (基准)
        :param downsample_voxel: 降采样体素大小 (可选, 加速计算)
        """
        self.source = copy.deepcopy(source_pcd)
        self.target = copy.deepcopy(target_pcd)
        
        # 预处理：降采样 (仅用于加速 Source 的对应点搜索)
        if downsample_voxel is not None:
            self.source_down = self.source.voxel_down_sample(downsample_voxel)
        else:
            self.source_down = self.source

        # 变换状态
        self.current_R = np.eye(3)
        self.current_t = np.zeros(3)
        
        # 优化历史记录
        self.history = {'iter': [], 'rmse': [], 'stage': []}

        # BIM 特征数据 (采样后的)
        self.bim_planes = None
        self.bim_edges = None
        self.bim_corners = None
        
        # 搜索树
        self.tree_plane = None
        self.tree_edge = None
        self.tree_corner = None
        
        # 辅助数据
        self.bim_plane_normals = None
        self.bim_edge_dirs = None

    def preprocess_target_features(self, target_points_num=5000):
        """
        使用 sample.py 提取 BIM 的平面、棱线、角点特征
        [关键修改]: 必须指定 target_points_num < 原始点数，以触发特征采样逻辑
        """
        original_count = len(self.target.points)
        print(f"[预处理] 正在构建结构化参考点云 (原始点数: {original_count}, 目标: {target_points_num})...")
        
        # 确保目标点数不大于原始点数，否则 sample.py 会直接返回原始点且不做特征分类
        target_num = min(target_points_num, int(original_count * 0.5)) 
        
        sampler = HierarchicalStructurePreservingSampling(
            target_num_points=target_num, 
            corner_ratio=0.15, 
            edge_ratio=0.35, 
            plane_ratio=0.50
        )
        
        # 提取特征 (points, indices, types)
        # types: 0=Plane, 1=Edge, 2=Corner
        pts, _, types = sampler.sample(self.target.points)
        pts = np.asarray(pts)
        
        # 分离点云
        self.bim_planes = pts[types == 0]
        self.bim_edges = pts[types == 1]
        self.bim_corners = pts[types == 2]
        
        print(f"  - 采样后平面点数: {len(self.bim_planes)}")
        print(f"  - 采样后棱线点数: {len(self.bim_edges)}")
        print(f"  - 采样后角点点数: {len(self.bim_corners)}")
        
        if len(self.bim_edges) == 0:
            print("[警告] 未检测到棱线点！请检查 sample.py 参数或点云尺度。")
        if len(self.bim_corners) == 0:
            print("[警告] 未检测到角点！")

        # 1. 计算平面法向量 (用于 Level-1 Point-to-Plane)
        if len(self.bim_planes) > 0:
            pcd_plane = o3d.geometry.PointCloud()
            pcd_plane.points = o3d.utility.Vector3dVector(self.bim_planes)
            pcd_plane.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            self.bim_plane_normals = np.asarray(pcd_plane.normals)
            self.tree_plane = o3d.geometry.KDTreeFlann(pcd_plane)

        # 2. 计算棱线主方向 (用于 Level-2 Point-to-Line)
        if len(self.bim_edges) > 0:
            pcd_edge = o3d.geometry.PointCloud()
            pcd_edge.points = o3d.utility.Vector3dVector(self.bim_edges)
            # 利用 KDTree 搜索局部邻域，计算 PCA 第一主成分作为线方向
            kdtree_edge = o3d.geometry.KDTreeFlann(pcd_edge)
            edge_dirs = []
            for pt in self.bim_edges:
                # 注意：如果点太少，k=10可能会报错，做个保护
                k_search = min(10, len(self.bim_edges))
                [_, idx, _] = kdtree_edge.search_knn_vector_3d(pt, k_search)
                neighbors = self.bim_edges[idx]
                if len(neighbors) >= 3:
                    cov = np.cov(neighbors.T)
                    eig_vals, eig_vecs = np.linalg.eigh(cov)
                    edge_dirs.append(eig_vecs[:, 2]) # 最大特征值对应的特征向量
                else:
                    edge_dirs.append(np.array([0,0,1])) # fallback
            self.bim_edge_dirs = np.array(edge_dirs)
            self.tree_edge = o3d.geometry.KDTreeFlann(pcd_edge)

        # 3. 角点树
        if len(self.bim_corners) > 0:
            pcd_corner = o3d.geometry.PointCloud()
            pcd_corner.points = o3d.utility.Vector3dVector(self.bim_corners)
            self.tree_corner = o3d.geometry.KDTreeFlann(pcd_corner)

    def transform_point_cloud(self, pcd):
        """应用当前变换到点云"""
        pcd_t = copy.deepcopy(pcd)
        T = np.eye(4)
        T[:3, :3] = self.current_R
        T[:3, 3] = self.current_t
        pcd_t.transform(T)
        return pcd_t

    # ------------------------------------------------------------------
    # Level-1: 姿态锁定 (Rotation Locking)
    # ------------------------------------------------------------------
    def run_level1(self, max_icp_iter=10):
        print("\n=== 启动 Level-1: 姿态锁定 (面域约束) ===")
        if self.tree_plane is None or len(self.bim_planes) == 0:
            print("  [错误] 无有效的平面特征点，跳过 Level-1")
            return

        print("  > 固定平移，优化旋转，最小化点-面距离")
        src_pts_all = np.asarray(self.source_down.points)
        
        for i in range(max_icp_iter):
            # 1. 变换源点云
            curr_src = (self.current_R @ src_pts_all.T).T + self.current_t
            
            # 2. 寻找平面对应点
            corres_src = []
            corres_tgt_pt = []
            corres_tgt_norm = []
            
            # 随机采样一部分源点加速 (例如 2000 个)
            indices = np.random.choice(len(curr_src), min(2000, len(curr_src)), replace=False)
            subset_src = curr_src[indices]
            
            for k, p in enumerate(subset_src):
                [_, idx, dist_sq] = self.tree_plane.search_knn_vector_3d(p, 1)
                dist = np.sqrt(dist_sq[0])
                # 距离阈值宽松一点，保证能拉回来
                if dist < 0.5: 
                    idx = idx[0]
                    corres_src.append(src_pts_all[indices[k]]) # 原始坐标
                    corres_tgt_pt.append(self.bim_planes[idx])
                    corres_tgt_norm.append(self.bim_plane_normals[idx])
            
            if len(corres_src) < 10:
                print("  [警告] 有效平面对应点不足，停止 Level-1")
                break
                
            corres_src = np.array(corres_src)
            corres_tgt_pt = np.array(corres_tgt_pt)
            corres_tgt_norm = np.array(corres_tgt_norm)
            
            # 3. 优化
            def loss_func(r_vec):
                rot_mat = R.from_rotvec(r_vec).as_matrix()
                p_est = (rot_mat @ corres_src.T).T + self.current_t
                vec = p_est - corres_tgt_pt
                dists = np.sum(vec * corres_tgt_norm, axis=1)
                return dists

            r0 = R.from_matrix(self.current_R).as_rotvec()
            res = least_squares(loss_func, r0, loss='huber', f_scale=0.1)
            
            self.current_R = R.from_rotvec(res.x).as_matrix()
            rmse = np.sqrt(np.mean(res.fun**2))
            
            self.history['iter'].append(len(self.history['iter']))
            self.history['rmse'].append(rmse)
            self.history['stage'].append('Level-1')
            
            print(f"  Iteration {i+1}: RMSE = {rmse:.6f}, Point Pairs = {len(corres_src)}")
            if rmse < 1e-4: break

    # ------------------------------------------------------------------
    # Level-2: 位置锁定 (Position Locking)
    # ------------------------------------------------------------------
    def run_level2(self, max_icp_iter=10):
        print("\n=== 启动 Level-2: 位置锁定 (边角约束) ===")
        if self.tree_edge is None and self.tree_corner is None:
            print("  [错误] 无有效的棱线或角点特征，跳过 Level-2")
            return

        print("  > 固定旋转，优化平移，最小化点-线/点-点距离")
        src_pts_all = np.asarray(self.source_down.points)
        
        for i in range(max_icp_iter):
            # 1. 变换源点云
            curr_src = (self.current_R @ src_pts_all.T).T + self.current_t
            
            c_src_edge, c_tgt_edge_pt, c_tgt_edge_dir = [], [], []
            c_src_corner, c_tgt_corner_pt = [], []
            
            # 随机采样
            indices = np.random.choice(len(curr_src), min(3000, len(curr_src)), replace=False)
            subset_src = curr_src[indices]
            
            # A. 棱线搜索
            if self.tree_edge:
                for k, p in enumerate(subset_src):
                    [_, idx, dist_sq] = self.tree_edge.search_knn_vector_3d(p, 1)
                    if dist_sq[0] < 0.2**2: # 20cm
                        idx = idx[0]
                        c_src_edge.append(src_pts_all[indices[k]])
                        c_tgt_edge_pt.append(self.bim_edges[idx])
                        c_tgt_edge_dir.append(self.bim_edge_dirs[idx])
            
            # B. 角点搜索
            if self.tree_corner:
                for k, p in enumerate(subset_src):
                    [_, idx, dist_sq] = self.tree_corner.search_knn_vector_3d(p, 1)
                    if dist_sq[0] < 0.2**2:
                        idx = idx[0]
                        c_src_corner.append(src_pts_all[indices[k]])
                        c_tgt_corner_pt.append(self.bim_corners[idx])

            n_edges = len(c_src_edge)
            n_corners = len(c_src_corner)
            
            if n_edges + n_corners < 5:
                print("  [警告] 有效边角对应点不足，停止 Level-2")
                break
            
            c_src_edge = np.array(c_src_edge) if n_edges > 0 else np.empty((0,3))
            c_tgt_edge_pt = np.array(c_tgt_edge_pt) if n_edges > 0 else np.empty((0,3))
            c_tgt_edge_dir = np.array(c_tgt_edge_dir) if n_edges > 0 else np.empty((0,3))
            c_src_corner = np.array(c_src_corner) if n_corners > 0 else np.empty((0,3))
            c_tgt_corner_pt = np.array(c_tgt_corner_pt) if n_corners > 0 else np.empty((0,3))

            # 3. 优化
            def loss_func(t_vec):
                residuals = []
                # 棱线误差
                if n_edges > 0:
                    p_est = (self.current_R @ c_src_edge.T).T + t_vec
                    vec = p_est - c_tgt_edge_pt
                    cross_prod = np.cross(vec, c_tgt_edge_dir)
                    dists_edge = np.linalg.norm(cross_prod, axis=1)
                    residuals.append(dists_edge)
                # 角点误差
                if n_corners > 0:
                    p_est = (self.current_R @ c_src_corner.T).T + t_vec
                    dists_corner = np.linalg.norm(p_est - c_tgt_corner_pt, axis=1)
                    residuals.append(dists_corner * 2.0)
                return np.concatenate(residuals)

            res = least_squares(loss_func, self.current_t, loss='linear')
            self.current_t = res.x
            
            rmse = np.sqrt(np.mean(res.fun**2))
            self.history['iter'].append(len(self.history['iter']))
            self.history['rmse'].append(rmse)
            self.history['stage'].append('Level-2')
            
            print(f"  Iteration {i+1}: RMSE = {rmse:.6f}, Edges={n_edges}, Corners={n_corners}")
            if rmse < 1e-4: break

    # ------------------------------------------------------------------
    # 辅助与可视化
    # ------------------------------------------------------------------
    def get_final_transformation(self):
        T = np.eye(4)
        T[:3, :3] = self.current_R
        T[:3, 3] = self.current_t
        return T

    def plot_rmse_curve(self):
        iters = self.history['iter']
        rmses = self.history['rmse']
        stages = self.history['stage']
        plt.figure(figsize=(10, 6))
        plt.plot(iters, rmses, 'b.-', label='RMSE')
        switch_idx = next((i for i, s in enumerate(stages) if s == 'Level-2'), None)
        if switch_idx:
            plt.axvline(x=switch_idx, color='r', linestyle='--', label='Level-2 Start')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE (m)')
        plt.title('Optimization Convergence Curve (Decoupled)')
        plt.legend()
        plt.grid(True)
        plt.show()

def visualize_result(source, target, features, title="Result"):
    """
    可视化结果
    """
    s_viz = copy.deepcopy(source)
    s_viz.paint_uniform_color([1, 0.706, 0]) # Scan = Yellow
    
    # 构建特征点云
    vis_list = [s_viz]
    
    # 平面点 - 蓝色
    if len(features['planes']) > 0:
        p_plane = o3d.geometry.PointCloud()
        p_plane.points = o3d.utility.Vector3dVector(features['planes'])
        p_plane.paint_uniform_color([0, 0, 0.8])
        vis_list.append(p_plane)
    else:
        # 如果特征为空，显示原始 Target - 青色
        t_viz = copy.deepcopy(target)
        t_viz.paint_uniform_color([0, 0.651, 0.929])
        vis_list.append(t_viz)

    # 棱线 - 绿色
    if len(features['edges']) > 0:
        p_edge = o3d.geometry.PointCloud()
        p_edge.points = o3d.utility.Vector3dVector(features['edges'])
        p_edge.paint_uniform_color([0, 0.8, 0])
        # 将点渲染大一点需要使用 Material，Open3D标准绘图不支持点大小调节，
        # 简单起见，我们只显示点云
        vis_list.append(p_edge)

    # 角点 - 红色
    if len(features['corners']) > 0:
        p_corner = o3d.geometry.PointCloud()
        p_corner.points = o3d.utility.Vector3dVector(features['corners'])
        p_corner.paint_uniform_color([0.8, 0, 0])
        vis_list.append(p_corner)
        
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])
    vis_list.append(axis)

    print(f"[{title}] 正在渲染... 请在弹出的窗口查看。")
    o3d.visualization.draw_geometries(vis_list, window_name=title, width=1024, height=768)

# ==============================================================================
# 主函数
# ==============================================================================
if __name__ == "__main__":
    # 1. 加载数据
    # 请替换为你的真实数据路径
    scan_path = r"scan_path"
    bim_path = r"bim_path"
    
    try:
        print(f"正在加载数据: {scan_path}, {bim_path}")
        scan_pcd = o3d.io.read_point_cloud(scan_path)
        bim_pcd = o3d.io.read_point_cloud(bim_path)
        
        if not scan_pcd.has_points() or not bim_pcd.has_points():
            raise ValueError("点云为空，请检查路径。")
    except Exception as e:
        print(f"错误: {e}")
        exit(1)

    # 2. 初始化解算器
    # 注意：不对 Target 降采样，因为我们要用它提取特征
    solver = DecoupledSolver(scan_pcd, bim_pcd, downsample_voxel=0.05)
    
    # 3. 预处理 BIM 特征 (关键: target_points_num 设置为 5000 以触发采样)
    solver.preprocess_target_features(target_points_num=5000)
    
    features_viz = {
        'planes': solver.bim_planes,
        'edges': solver.bim_edges,
        'corners': solver.bim_corners
    }

    # 可视化 a) 初始状态
    visualize_result(scan_pcd, bim_pcd, features_viz, title="a) Initial State")

    # 4. 执行 Level-1 (姿态锁定)
    solver.run_level1(max_icp_iter=15)
    
    # 可视化 b) 姿态锁定后
    scan_l1 = solver.transform_point_cloud(scan_pcd)
    visualize_result(scan_l1, bim_pcd, features_viz, title="b) After Level-1 (Attitude Locked)")

    # 5. 执行 Level-2 (位置锁定)
    solver.run_level2(max_icp_iter=15)
    
    # 6. 最终评估
    T_final = solver.get_final_transformation()
    scan_final = solver.transform_point_cloud(scan_pcd)
    
    # 计算最终 RMSE
    dists = scan_final.compute_point_cloud_distance(bim_pcd)
    final_rmse = np.sqrt(np.mean(np.asarray(dists)**2))
    
    print("\n================ 评估报告 ================")
    print("1. 最终变换矩阵 T:")
    print(T_final)
    print(f"2. 配准重对齐误差 (RMSE): {final_rmse*1000:.2f} mm")
    
    r_euler = R.from_matrix(T_final[:3,:3]).as_euler('xyz', degrees=True)
    print(f"3. 估计旋转 (Euler deg): {r_euler}")
    print(f"4. 估计平移 (m): {T_final[:3, 3]}")
    print("==========================================")

    # 可视化 c) 最终状态
    visualize_result(scan_final, bim_pcd, features_viz, title="c) Final Result (Position Locked)")
    
    # 绘制误差曲线
    solver.plot_rmse_curve()