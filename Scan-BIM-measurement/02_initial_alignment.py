import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt

# ==============================================================================
# 模块 1：高级可视化辅助函数
# ==============================================================================

def create_pca_axis_lineset(pcd, scale_ratio=0.5):
    """
    计算并创建点云 PCA 主轴的可视化 LineSet
    自适应长度：根据点云包围盒大小自动调整主轴显示长度
    红色 = e1 (长轴), 绿色 = e2 (宽轴), 蓝色 = e3 (高轴)
    """
    points_np = np.asarray(pcd.points)
    centroid = np.mean(points_np, axis=0)
    
    # 1. 计算 PCA
    cov = np.cov(points_np.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # 排序：从大到小
    sort_indices = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, sort_indices]
    
    # 确保右手定则
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1
        
    # 2. 自适应计算显示长度
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_max_bound() - bbox.get_min_bound()
    scale = np.linalg.norm(extent) * scale_ratio

    # 3. 构建坐标轴线段
    points = [
        centroid,
        centroid + basis[:, 0] * scale, 
        centroid + basis[:, 1] * scale, 
        centroid + basis[:, 2] * scale
    ]
    lines = [[0, 1], [0, 2], [0, 3]]
    # 使用更鲜艳的颜色
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set, basis

def visualize_step(geometries, window_name, width=1024, height=768):
    """3D 通用渲染函数"""
    print(f"[{window_name}] 3D渲染中... (请调整视角，按 'Q' 键继续)")
    
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height, left=50, top=50)
    
    # 添加几何对象
    for geom in geometries:
        vis.add_geometry(geom)
    
    # 获取渲染选项并设置线条宽度
    opt = vis.get_render_option()
    opt.line_width = 500.0  # 加宽线条，使 PCA 主轴更明显
    
    # 渲染并等待用户操作
    vis.run()
    vis.destroy_window()

def visualize_grid_overlap(mask_scan, mask_bim, iou, title):
    """
    2D 网格可视化函数 (使用 Matplotlib)
    """
    h, w = mask_scan.shape
    img = np.zeros((h, w, 3), dtype=np.float32)
    
    # Scan = Red, BIM = Green, Overlap = Yellow
    img[:, :, 0] = mask_scan  # Red
    img[:, :, 1] = mask_bim   # Green
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img, origin='lower') 
    plt.title(f"{title}\nGrid IoU: {iou:.4f}")
    plt.axis('off')
    plt.tight_layout()
    print(f"[{title}] 2D Grid渲染中... (请关闭绘图窗口以继续)")
    plt.show()

# ==============================================================================
# 模块 2：核心算法步骤 (修复 MemoryError)
# ==============================================================================

def normalize_centroid(pcd):
    """Step 1: 质心归化"""
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    pcd_centered = copy.deepcopy(pcd)
    pcd_centered.translate(-centroid)
    return pcd_centered, centroid

def get_candidate_rotations(U_scan, U_bim):
    """Step 2: 构建候选旋转"""
    # 基础旋转: R * U_scan = U_bim
    R_base = U_bim @ U_scan.T
    
    candidates = []
    # 生成 4 个正交候选
    for k in range(4):
        theta = k * np.pi / 2.0
        c, s = np.cos(theta), np.sin(theta)
        R_z = np.array([[c, -s, 0], [s,  c, 0], [0,  0, 1]])
        R_cand = R_z @ R_base
        candidates.append(R_cand)
    return candidates

def points_to_grid(points, grid_res, min_xy, grid_shape):
    """投影并栅格化"""
    pts_xy = points[:, [0, 1]]
    indices = np.floor((pts_xy - min_xy) / grid_res).astype(np.int32)
    
    # 过滤越界点
    valid_mask = (indices[:, 0] >= 0) & (indices[:, 0] < grid_shape[1]) & \
                 (indices[:, 1] >= 0) & (indices[:, 1] < grid_shape[0])
    indices = indices[valid_mask]
    
    grid = np.zeros(grid_shape, dtype=np.float32)
    # 高级索引填充
    grid[indices[:, 1], indices[:, 0]] = 1.0
    return grid

def compute_grid_iou_and_translation(source_centered, target_centered, R_cand):
    """
    Step 3: AABB 平移修正 + 自适应 Grid IoU 计算
    """
    # 1. 应用候选旋转
    source_rotated = copy.deepcopy(source_centered)
    source_rotated.rotate(R_cand, center=(0,0,0))
    
    # 2. AABB 平移修正
    c_scan = source_rotated.get_axis_aligned_bounding_box().get_center()
    c_bim = target_centered.get_axis_aligned_bounding_box().get_center()
    t_corr = c_bim - c_scan
    
    # 3. 应用平移
    source_final = copy.deepcopy(source_rotated)
    source_final.translate(t_corr)
    
    # 4. [修复核心] 自适应计算 Grid 参数
    #    不再使用固定 grid_res=0.1，而是限制网格最大边长为 512 像素
    #    这能解决单位是 mm 时导致的 MemoryError
    pts_src = np.asarray(source_final.points)
    pts_tgt = np.asarray(target_centered.points)
    all_pts = np.vstack([pts_src, pts_tgt])
    
    min_xy = np.min(all_pts[:, [0, 1]], axis=0)
    max_xy = np.max(all_pts[:, [0, 1]], axis=0)
    span = max_xy - min_xy
    
    # 自适应分辨率：将最大边长限制在 500 像素左右
    # 既保证了 IoU 的形状敏感度，又保证了内存不爆炸
    max_span = max(span[0], span[1])
    if max_span == 0: max_span = 1.0
    grid_res = max_span / 500.0 
    
    # 计算 Grid 尺寸
    grid_w = int(np.ceil((max_xy[0] - min_xy[0]) / grid_res)) + 4
    grid_h = int(np.ceil((max_xy[1] - min_xy[1]) / grid_res)) + 4
    min_xy_padded = min_xy - grid_res * 2
    
    # 5. 栅格化
    mask_scan = points_to_grid(pts_src, grid_res, min_xy_padded, (grid_h, grid_w))
    mask_bim = points_to_grid(pts_tgt, grid_res, min_xy_padded, (grid_h, grid_w))
    
    # 6. 计算 IoU
    intersection = np.sum(np.logical_and(mask_scan > 0, mask_bim > 0))
    union = np.sum(np.logical_or(mask_scan > 0, mask_bim > 0))
    
    iou = 0.0
    if union > 0:
        iou = intersection / union
        
    return t_corr, iou, source_final, mask_scan, mask_bim

# ==============================================================================
# 主流程：6.3.1 初始位姿估计 (修复版)
# ==============================================================================

def initial_alignment_pipeline_v4(scan_pcd, bim_pcd):
    print("="*60)
    print("启动 6.3.1 初始位姿估计 (V4: Memory Fix & Visualization Restore)")
    print("="*60)
    
    # 预处理：染色
    scan_pcd.paint_uniform_color([1, 0.706, 0]) # 黄色 Scan
    bim_pcd.paint_uniform_color([0, 0.651, 0.929]) # 蓝色 BIM
    
    # ------------------------------------------------------------------
    # 可视化 1: 原始状态分别展示 (带主轴)
    # ------------------------------------------------------------------
    print("[可视化 1.1] 展示 Scan 点云及其主轴")
    axes_scan_raw, U_scan_raw = create_pca_axis_lineset(scan_pcd, scale_ratio=0.6)
    visualize_step([scan_pcd, axes_scan_raw], "1.1 Scan Point Cloud & PCA Axes")

    print("[可视化 1.2] 展示 BIM 点云及其主轴")
    axes_bim_raw, U_bim_raw = create_pca_axis_lineset(bim_pcd, scale_ratio=0.6)
    visualize_step([bim_pcd, axes_bim_raw], "1.2 BIM Point Cloud & PCA Axes")

    # ------------------------------------------------------------------
    # Step 1: 质心归化
    # ------------------------------------------------------------------
    print("[Step 1] 执行质心归化...")
    scan_centered, c_scan = normalize_centroid(scan_pcd)
    bim_centered, c_bim = normalize_centroid(bim_pcd)
    
    # ------------------------------------------------------------------
    # 可视化 2: 去中心化后联合展示 (带主轴)
    # ------------------------------------------------------------------
    print("[可视化 2] 质心归化后状态 (同时显示)")
    # 重新计算归化后的主轴 LineSet
    axes_scan, U_scan = create_pca_axis_lineset(scan_centered, scale_ratio=0.6)
    axes_bim, U_bim = create_pca_axis_lineset(bim_centered, scale_ratio=0.6)
    
    # 添加一个原点坐标系辅助观察
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0])
    
    visualize_step([scan_centered, axes_scan, 
                    bim_centered, axes_bim, origin], 
                   "2. Normalized Centroids & Combined Axes")
    
    # ------------------------------------------------------------------
    # Step 2: 构建候选旋转
    # ------------------------------------------------------------------
    candidates = get_candidate_rotations(U_scan, U_bim)
    print(f"[Step 2] 生成 {len(candidates)} 个候选旋转，开始 Grid IoU 验证...")
    
    best_iou = -1.0
    best_T = np.eye(4)
    best_idx = 0
    
    # ------------------------------------------------------------------
    # Step 3: 遍历验证
    # ------------------------------------------------------------------
    for i, R_k in enumerate(candidates):
        # 核心修复：内部使用自适应分辨率，不会再报 MemoryError
        t_corr, iou, scan_final, mask_scan, mask_bim = compute_grid_iou_and_translation(
            scan_centered, bim_centered, R_k
        )
        
        print(f"  > 候选 {i} (Rot {i*90}°): Grid IoU = {iou:.4f}")
        
        # --- 可视化 3: Grid IoU ---
        # 2D Grid
        visualize_grid_overlap(mask_scan, mask_bim, iou, f"Candidate {i} (Rot {i*90} deg)")
        
        # 3D View (可选，为了流畅性可以注释掉)
        # visualize_step([scan_final, bim_centered], f"3D View: Candidate {i} - IoU {iou:.2f}")
        
        if iou > best_iou:
            best_iou = iou
            best_idx = i
            T_local = np.eye(4)
            T_local[:3, :3] = R_k
            T_local[:3, 3] = t_corr
            best_T = T_local

    print("-" * 60)
    print(f"最优候选: {best_idx}, Max Grid IoU: {best_iou:.4f}")
    
    # 结果还原
    T_to_center = np.eye(4); T_to_center[:3, 3] = -c_scan
    T_back_world = np.eye(4); T_back_world[:3, 3] = c_bim
    T_final = T_back_world @ best_T @ T_to_center
    
    # ------------------------------------------------------------------
    # 可视化 4: 最终结果
    # ------------------------------------------------------------------
    print("[可视化 4] 最终配准结果")
    scan_final_global = copy.deepcopy(scan_pcd)
    scan_final_global.transform(T_final)
    visualize_step([scan_final_global, bim_pcd], "4. Final Alignment Result")
    
    return T_final

# ==============================================================================
# 入口
# ==============================================================================
if __name__ == "__main__":
    # 配置
    USE_REAL_DATA = True 
    
    if USE_REAL_DATA:
        # 请替换为你的真实路径
        scan_path = r"D:\Application\PycharmProfessional\pycharm\PointCloud_registration\PCR_test\data\merged_random.ply"
        bim_path = r"D:\Application\PycharmProfessional\pycharm\PointCloud_registration\PCR_test\data\cad_cloud2.ply"
        try:
            pcd_scan = o3d.io.read_point_cloud(scan_path)
            pcd_bim = o3d.io.read_point_cloud(bim_path)
            if not pcd_scan.has_points(): raise ValueError("Scan is empty")
        except Exception as e:
            print(f"加载失败: {e}，切换至模拟模式")
            USE_REAL_DATA = False

    if not USE_REAL_DATA:
        print("生成模拟数据...")
        # 生成毫米级单位的数据 (模拟导致内存溢出的情况)
        mesh_box = o3d.geometry.TriangleMesh.create_box(10000, 3000, 3000) # 10米 x 3米
        pcd_bim = mesh_box.sample_points_poisson_disk(5000)
        
        pcd_scan = copy.deepcopy(pcd_bim)
        pts = np.asarray(pcd_scan.points)
        mask = (pts[:, 1] < 1000) | (pts[:, 2] < 1000) # L型
        pcd_scan = pcd_scan.select_by_index(np.where(mask)[0])
        
        R_gt = pcd_scan.get_rotation_matrix_from_xyz((np.pi/2, 0.1, 0.1))
        pcd_scan.rotate(R_gt, center=(0,0,0))
        pcd_scan.translate((20000, -10000, 5000)) # 大平移

    T_init = initial_alignment_pipeline_v4(pcd_scan, pcd_bim)
    print("\n初始位姿矩阵 T_init:\n", T_init)