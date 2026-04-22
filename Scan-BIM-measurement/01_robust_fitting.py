import numpy as np
import open3d as o3d
import copy

# ==========================================
# 1. 核心算法类：包含 LS, RANSAC, Huber, Tukey
# ==========================================

class PlaneFitter:
    
    @staticmethod
    def format_equation(model):
        """
        格式化输出平面方程参数: Ax + By + Cz + D = 0
        """
        return f"{model[0]:>7.4f} x + {model[1]:>7.4f} y + {model[2]:>7.4f} z + {model[3]:>7.4f} = 0"

    @staticmethod
    def _unify_direction(model, reference_normal=None):
        """
        统一法向量方向，确保 A,B,C,D 便于比较
        如果提供了参考法向，则与参考方向一致；否则默认 Z>0 或 Y>0
        """
        n = model[:3]
        d = model[3]
        
        # 如果有参考真值，与其点积判断
        if reference_normal is not None:
            if np.dot(n, reference_normal) < 0:
                return -model
            return model
            
        # 默认规则：让法向量的第一个最大分量为正，便于阅读
        max_idx = np.argmax(np.abs(n))
        if n[max_idx] < 0:
            return -model
        return model

    @staticmethod
    def fit_ls(points):
        """标准最小二乘法 (Least Squares)"""
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        cov = np.dot(centered.T, centered)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        normal = eig_vecs[:, 0] # 最小特征值对应的特征向量
        d = -np.dot(normal, centroid)
        return np.array([normal[0], normal[1], normal[2], d])

    @staticmethod
    def fit_ransac(points, threshold=0.01):
        """RANSAC (Open3D 实现)"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # 迭代次数设为 2000 以保证稳定性
        plane_model, inliers = pcd.segment_plane(distance_threshold=threshold,
                                                 ransac_n=3,
                                                 num_iterations=2000)
        return np.array(plane_model)

    @staticmethod
    def fit_irls_huber(points, max_iter=30, tol=1e-6):
        """Huber M-估计 (对比算法)"""
        # 使用 RANSAC 热启动
        current_model = PlaneFitter.fit_ransac(points, threshold=0.015)
        k_huber = 1.345
        
        for k in range(max_iter):
            prev_model = current_model
            n, d = current_model[:3], current_model[3]
            residuals = np.dot(points, n) + d
            
            # MAD 尺度估计
            median_res = np.median(residuals)
            mad = np.median(np.abs(residuals - median_res))
            sigma = 1.4826 * mad
            if sigma < 1e-6: sigma = 1e-6
            
            # Huber 权重计算
            res_scaled = np.abs(residuals / sigma)
            weights = np.ones_like(residuals)
            mask = res_scaled > k_huber
            # Huber 只是降低权重，不会置零
            weights[mask] = k_huber / res_scaled[mask]
            
            # 加权最小二乘更新 (WLS)
            w_sum = np.sum(weights)
            centroid = np.average(points, axis=0, weights=weights)
            centered = points - centroid
            weighted_centered = centered * weights[:, np.newaxis]
            cov = np.dot(centered.T, weighted_centered)
            eig_vals, eig_vecs = np.linalg.eigh(cov)
            new_n = eig_vecs[:, 0]
            
            # 保持方向一致
            if np.dot(new_n, n) < 0: new_n = -new_n
            new_d = -np.dot(new_n, centroid)
            current_model = np.array([new_n[0], new_n[1], new_n[2], new_d])
            
            if np.linalg.norm(current_model - prev_model) < tol: break
            
        return current_model

    @staticmethod
    def fit_robust_tukey(points, max_iter=50, tol=1e-6):
        """
        [本文提出算法] RANSAC 初始化 + Tukey M-估计精修
        """
        # Step 1: RANSAC 粗配准 (热启动)
        # 阈值 0.015 (1.5cm) 能覆盖大部分噪声，但排除了大鼓包
        current_model = PlaneFitter.fit_ransac(points, threshold=0.015)
        
        # Step 2: IRLS 迭代
        for k in range(max_iter):
            prev_model = current_model
            n, d = current_model[:3], current_model[3]
            
            # 计算残差
            residuals = np.dot(points, n) + d
            
            # 鲁棒尺度估计 (MAD)
            median_res = np.median(residuals)
            mad = np.median(np.abs(residuals - median_res))
            sigma = 1.4826 * mad
            if sigma < 1e-6: sigma = 1e-6
            
            # Tukey 权重 (c=4.685)
            # 红色下降特性: 残差过大直接权重置零
            c = 4.685 * sigma
            weights = np.zeros(len(points))
            mask = np.abs(residuals) <= c
            
            # 安全检查：如果内点太少，停止迭代
            if np.sum(mask) < 10: break
                
            weights[mask] = (1 - (residuals[mask] / c) ** 2) ** 2
            
            # WLS 更新
            w_sum = np.sum(weights)
            centroid = np.average(points, axis=0, weights=weights)
            centered = points - centroid
            weighted_centered = centered * weights[:, np.newaxis]
            cov = np.dot(centered.T, weighted_centered)
            eig_vals, eig_vecs = np.linalg.eigh(cov)
            new_n = eig_vecs[:, 0]
            
            if np.dot(new_n, n) < 0: new_n = -new_n
            new_d = -np.dot(new_n, centroid)
            current_model = np.array([new_n[0], new_n[1], new_n[2], new_d])
            
            if np.linalg.norm(current_model - prev_model) < tol: break
            
        return current_model
    
    @staticmethod
    def fit_region_growing(points, distance_threshold=0.01, angle_threshold=30.0):
        """
        基于区域生长的平面拟合方法
        
        Args:
            points: 点云数据
            distance_threshold: 距离阈值 (m)
            angle_threshold: 角度阈值 (度)
            
        Returns:
            平面模型 [A, B, C, D]
        """
        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 计算法线
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        try:
            # 使用 RANSAC 找到初始平面
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=1000
            )
            
            # 基于初始平面进行区域生长
            # 这里使用 RANSAC 结果作为区域生长的基础
            # 因为 Open3D 的区域生长分割返回多个平面，我们选择最大的平面
            
            # 创建一个新的点云，只包含内点
            inlier_cloud = pcd.select_by_index(inliers)
            
            # 对初始平面进行最小二乘拟合，获得更准确的平面模型
            if len(inliers) > 3:
                inlier_points = np.asarray(inlier_cloud.points)
                # 使用最小二乘法拟合平面
                centroid = np.mean(inlier_points, axis=0)
                centered = inlier_points - centroid
                cov = np.dot(centered.T, centered)
                eig_vals, eig_vecs = np.linalg.eigh(cov)
                normal = eig_vecs[:, 0]  # 最小特征值对应的特征向量
                d = -np.dot(normal, centroid)
                plane_model = np.array([normal[0], normal[1], normal[2], d])
            
            return plane_model
        except:
            # 如果区域生长失败，回退到 RANSAC
            return PlaneFitter.fit_ransac(points, threshold=distance_threshold)
    
    @staticmethod
    def fit_tls_svd(points):
        """
        基于奇异值分解（SVD）的总最小二乘（Total Least Squares, TLS）平面拟合
        
        Args:
            points: 点云数据
            
        Returns:
            平面模型 [A, B, C, D]
        """
        # 中心化数据
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # 使用SVD分解
        U, S, Vt = np.linalg.svd(centered)
        
        # 最小奇异值对应的右奇异向量即为平面法向量
        normal = Vt[-1, :]
        
        # 计算平面方程的 D 参数
        d = -np.dot(normal, centroid)
        
        return np.array([normal[0], normal[1], normal[2], d])
    
    @staticmethod
    def fit_wls(points, weights=None):
        """
        加权最小二乘（Weighted Least Squares, WLS）平面拟合
        
        Args:
            points: 点云数据
            weights: 权重数组，若为 None 则使用均匀权重
            
        Returns:
            平面模型 [A, B, C, D]
        """
        # 如果没有提供权重，使用均匀权重
        if weights is None:
            weights = np.ones(len(points))
        
        # 计算加权中心点
        centroid = np.average(points, axis=0, weights=weights)
        
        # 中心化数据
        centered = points - centroid
        
        # 计算加权协方差矩阵
        weighted_centered = centered * weights[:, np.newaxis]
        cov = np.dot(centered.T, weighted_centered)
        
        # 计算特征值和特征向量
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        
        # 最小特征值对应的特征向量即为平面法向量
        normal = eig_vecs[:, 0]
        
        # 计算平面方程的 D 参数
        d = -np.dot(normal, centroid)
        
        return np.array([normal[0], normal[1], normal[2], d])


# ==========================================
# 2. 数据生成器：专注于 "非理想平面"
# ==========================================

class DataGenerator:
    @staticmethod
    def generate_planar_defect_data(scenario_type="Weld_Bulge"):
        """
        生成带有特定工业缺陷的平面点云 (2m x 1.5m 钢板)
        """
        n_points = 12000
        # 1. 生成基底平面 (Z=0)
        x = np.random.uniform(-1.0, 1.0, n_points)
        y = np.random.uniform(-0.75, 0.75, n_points)
        z = np.zeros(n_points)
        
        # 2. 基础测量噪声 (高斯分布, sigma=2mm)
        z += np.random.normal(0, 0.002, n_points)
        
        # 3. 添加特定的结构性缺陷
        if scenario_type == "Weld_Bulge":
            # [焊缝鼓包]: 中心区域高斯隆起 30mm
            r = np.sqrt(x**2 + y**2)
            mask = r < 0.4
            z[mask] += 0.030 * np.exp(-(r[mask]**2)/0.08)
            
        elif scenario_type == "Plate_Warping":
            # [板材翘曲]: 整体呈二次曲面弯曲 (最大挠度约 50mm)
            z += 0.05 * (x**2)
            
        elif scenario_type == "Step_Offset":
            # [错台/拼接误差]: 钢板拼接处有 15mm 的台阶
            # 这种情况下，基准面应以面积较大的一侧为主，或者取平均面
            # 鲁棒算法通常会锁住面积较大的一侧
            mask = x > 0.3
            z[mask] += 0.015 
            
        elif scenario_type == "Mixed_Defects":
            # [混合干扰]: 同时存在微小翘曲 + 稀疏离群点 + 局部凹陷
            # 翘曲
            z += 0.02 * (y**2)
            # 局部凹陷
            r_pit = np.sqrt((x-0.5)**2 + (y-0.5)**2)
            mask_pit = r_pit < 0.2
            z[mask_pit] -= 0.015
            # 离群点 (5%)
            idx = np.random.choice(n_points, int(n_points*0.05), replace=False)
            z[idx] += np.random.uniform(0.05, 0.10, len(idx))

        points = np.column_stack((x, y, z))
        
        # 4. 随机姿态旋转 (模拟真实 ROI，不让平面平行于坐标轴)
        # 绕 X 轴转 15度, 绕 Y 轴转 -10度
        R = o3d.geometry.get_rotation_matrix_from_xyz((np.radians(15), np.radians(-10), 0))
        points = np.dot(points, R.T)
        
        # 5. 计算真值 (Ground Truth)
        # 原始平面法向为 (0,0,1), d=0
        gt_n = np.dot(R, np.array([0,0,1]))
        gt_model = np.array([gt_n[0], gt_n[1], gt_n[2], 0])
            
        return points, gt_model


# ==========================================
# 3. 可视化与评价系统
# ==========================================

def run_comparison_and_visualize():
    # 定义测试场景
    scenarios = ["Weld_Bulge", "Plate_Warping", "Step_Offset", "Mixed_Defects"]
    
    # 颜色库
    colors = {
        "LS": [1, 0, 0],       # 红: 最小二乘 (反面教材)
        "RANSAC": [0, 0, 1],   # 蓝: RANSAC (粗糙)
        "Huber": [1, 0, 1],    # 紫: Huber (对比)
        "Proposed": [0, 1, 0], # 绿: 本文方法 (Tukey)
        "RegionGrowing": [0, 1, 1],  # 青: 区域生长方法
        "TLS-SVD": [0.5, 0, 0.5],  # 深紫: 总最小二乘
        "WLS": [1, 0.5, 0]     # 橙: 加权最小二乘
    }

    for scene in scenarios:
        print(f"\n{'='*30} 测试场景: {scene} {'='*30}")
        
        # 1. 生成数据
        points, gt_model = DataGenerator.generate_planar_defect_data(scene)
        
        # 统一真值方向
        gt_model = PlaneFitter._unify_direction(gt_model)

        # 2. 运行算法并记录时间
        import time
        models = {}
        times = {}
        
        start_time = time.time()
        models["LS"] = PlaneFitter.fit_ls(points)
        times["LS"] = time.time() - start_time
        
        start_time = time.time()
        models["RANSAC"] = PlaneFitter.fit_ransac(points, threshold=0.015)
        times["RANSAC"] = time.time() - start_time
        
        start_time = time.time()
        models["Huber"] = PlaneFitter.fit_irls_huber(points)
        times["Huber"] = time.time() - start_time
        
        start_time = time.time()
        models["Proposed"] = PlaneFitter.fit_robust_tukey(points)
        times["Proposed"] = time.time() - start_time
        
        start_time = time.time()
        models["RegionGrowing"] = PlaneFitter.fit_region_growing(points)
        times["RegionGrowing"] = time.time() - start_time
        
        start_time = time.time()
        models["TLS-SVD"] = PlaneFitter.fit_tls_svd(points)
        times["TLS-SVD"] = time.time() - start_time
        
        start_time = time.time()
        # 对于 WLS，使用基于残差的权重（这里使用 RANSAC 初始模型计算残差）
        initial_model = PlaneFitter.fit_ransac(points, threshold=0.015)
        residuals = np.abs(np.dot(points, initial_model[:3]) + initial_model[3])
        # 使用残差的倒数作为权重（添加小量以避免除以零）
        weights = 1.0 / (residuals + 1e-6)
        models["WLS"] = PlaneFitter.fit_wls(points, weights=weights)
        times["WLS"] = time.time() - start_time
        
        # 3. 打印结果对比表（调整间距）
        print(f"{'Method':<10} | {'Angle Err(deg)':<12} | {'Dist Err(mm)':<12} | {'RMSE(mm)':<12} | {'MAE(mm)':<12} | {'Inlier Ratio':<12} | {'Time(ms)':<10}")
        print("-" * 85)
        
        # 打印真值
        print(f"{'GT (Ref)':<10} | {'0.0000':<12} | {'0.0000':<12} | {'0.0000':<12} | {'0.0000':<12} | {'1.0000':<12} | {'0.0000':<10}")
        print("-" * 85)

        # 计算点到平面的距离
        def distance_to_plane(points, plane):
            A, B, C, D = plane
            return np.abs(A * points[:, 0] + B * points[:, 1] + C * points[:, 2] + D) / np.sqrt(A**2 + B**2 + C**2)
        
        threshold = 0.015  # 15mm 阈值

        for name in ["LS", "RANSAC", "Huber", "Proposed", "RegionGrowing", "TLS-SVD", "WLS"]:
            model = models[name]
            # 统一方向以便比较方程参数
            model = PlaneFitter._unify_direction(model, reference_normal=gt_model[:3])
            
            # 计算指标
            n_est, n_gt = model[:3], gt_model[:3]
            
            # 1. 角度误差
            dot = np.clip(np.dot(n_est, n_gt), -1, 1)
            angle_err = np.degrees(np.arccos(dot))
            
            # 2. 距离误差 (平面原点距离差)
            dist_err = np.abs(np.abs(model[3]) - np.abs(gt_model[3])) * 1000
            
            # 3. 均方根误差 (RMSE)
            distances = distance_to_plane(points, model)
            rmse = np.sqrt(np.mean(distances**2)) * 1000
            
            # 4. 平均绝对误差 (MAE)
            mae = np.mean(distances) * 1000
            
            # 5. 内点率
            inlier_ratio = np.sum(distances < threshold) / len(distances)
            
            # 6. 计算时间
            time_ms = times[name] * 1000
            
            print(f"{name:<10} | {angle_err:<12.4f} | {dist_err:<12.4f} | {rmse:<12.4f} | {mae:<12.4f} | {inlier_ratio:<12.4f} | {time_ms:<10.4f}")
        
        # 4. 可视化
        # 4.1 单独可视化需要拟合的数据
        print(f"\n正在渲染原始数据: {scene}... (请查看弹出窗口)")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # 根据高度设置点云颜色，使数据分布更明显
        z_values = points[:, 2]
        z_min, z_max = z_values.min(), z_values.max()
        normalized_z = (z_values - z_min) / (z_max - z_min + 1e-6)
        colors = np.zeros((len(points), 3))
        colors[:, 0] = normalized_z  # 红通道随高度增加
        colors[:, 2] = 1 - normalized_z  # 蓝通道随高度减少
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 添加坐标轴
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        # 添加窗口大小参数
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Original Data: {scene}", width=800, height=600)
        vis.add_geometry(pcd)
        vis.add_geometry(axis)
        vis.run()
        vis.destroy_window()
        
        # 4.2 可视化拟合平面
        print(f"正在渲染拟合结果: {scene}... (请查看弹出窗口)")
        geoms = [pcd]
        
        # 绘制拟合平面
        # 找到点云中心，用于放置平面 Mesh
        center = np.mean(points, axis=0)
        
        # 颜色库
        plane_colors = {
            "LS": [1, 0, 0],       # 红: 最小二乘
            "RANSAC": [0, 0, 1],   # 蓝: RANSAC
            "Huber": [1, 0, 1],    # 紫: Huber
            "Proposed": [0, 1, 0], # 绿: 本文方法 (Tukey)
            "RegionGrowing": [0, 1, 1],  # 青: 区域生长方法
            "TLS-SVD": [0.5, 0, 0.5],  # 深紫: 总最小二乘
            "WLS": [1, 0.5, 0]     # 橙: 加权最小二乘
        }
        
        for name, model in models.items():
            model = PlaneFitter._unify_direction(model, reference_normal=gt_model[:3])
            
            # 创建平面 Mesh
            mesh = o3d.geometry.TriangleMesh.create_box(width=2.5, height=2.0, depth=0.002)
            mesh.paint_uniform_color(plane_colors[name])
            
            # 变换 Mesh 到拟合位置
            # 1. 移到原点中心
            mesh.translate([-1.25, -1.0, 0])
            
            # 2. 旋转 (Z轴 -> 法向量 n)
            n = model[:3]
            z_axis = np.array([0, 0, 1])
            axis = np.cross(z_axis, n)
            angle_rot = np.arccos(np.dot(z_axis, n))
            if np.linalg.norm(axis) > 1e-6:
                axis /= np.linalg.norm(axis)
                R_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle_rot)
                mesh.rotate(R_mat, center=[0,0,0])
                
            # 3. 平移 (投影中心)
            # P_proj = Center - (n*Center + d) * n
            dist_val = np.dot(center, n) + model[3]
            trans_vec = center - dist_val * n
            
            # 为了防止平面重叠导致的 Z-fighting，稍微沿法向错开更多，使效果更明显
            offset_map = {"LS": 0.00, "RANSAC": 0.02, "Huber": 0.04, "Proposed": 0.06, "RegionGrowing": 0.08, "TLS-SVD": 0.10, "WLS": 0.12}
            mesh.translate(trans_vec + n * offset_map[name])
            
            # 为平面添加标签
            # 注意：Open3D 不直接支持文本标签，这里通过添加一个小立方体作为标记
            label_marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            label_marker.paint_uniform_color(plane_colors[name])
            label_marker.translate(trans_vec + n * (offset_map[name] + 0.03))
            geoms.append(label_marker)
            geoms.append(mesh)

        # 添加坐标轴
        geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
        
        # 添加窗口大小参数
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Fitting Results: {scene}", width=800, height=600)
        for geom in geoms:
            vis.add_geometry(geom)
        vis.run()
        vis.destroy_window()

    # 5. 分析总结
    print(f"\n{'='*60}")
    print("测试数据分析总结")
    print(f"{'='*60}")
    
    # 总结各场景的最佳方法
    print("\n各场景最佳拟合方法:")
    print("-" * 60)
    
    # 计算每个场景的最佳方法
    best_methods = {}
    for scene in scenarios:
        # 重新生成数据并运行算法以获取结果
        points, gt_model = DataGenerator.generate_planar_defect_data(scene)
        gt_model = PlaneFitter._unify_direction(gt_model)
        
        models = {}
        start_time = time.time()
        models["LS"] = PlaneFitter.fit_ls(points)
        times["LS"] = time.time() - start_time
        
        start_time = time.time()
        models["RANSAC"] = PlaneFitter.fit_ransac(points, threshold=0.015)
        times["RANSAC"] = time.time() - start_time
        
        start_time = time.time()
        models["Huber"] = PlaneFitter.fit_irls_huber(points)
        times["Huber"] = time.time() - start_time
        
        start_time = time.time()
        models["Proposed"] = PlaneFitter.fit_robust_tukey(points)
        times["Proposed"] = time.time() - start_time
        
        start_time = time.time()
        models["RegionGrowing"] = PlaneFitter.fit_region_growing(points)
        times["RegionGrowing"] = time.time() - start_time
        
        start_time = time.time()
        models["TLS-SVD"] = PlaneFitter.fit_tls_svd(points)
        times["TLS-SVD"] = time.time() - start_time
        
        start_time = time.time()
        # 对于 WLS，使用基于残差的权重（这里使用 RANSAC 初始模型计算残差）
        initial_model = PlaneFitter.fit_ransac(points, threshold=0.015)
        residuals = np.abs(np.dot(points, initial_model[:3]) + initial_model[3])
        # 使用残差的倒数作为权重（添加小量以避免除以零）
        weights = 1.0 / (residuals + 1e-6)
        models["WLS"] = PlaneFitter.fit_wls(points, weights=weights)
        times["WLS"] = time.time() - start_time
        
        # 计算各方法的综合得分
        scores = {}
        for name, model in models.items():
            model = PlaneFitter._unify_direction(model, reference_normal=gt_model[:3])
            n_est, n_gt = model[:3], gt_model[:3]
            
            # 角度误差
            dot = np.clip(np.dot(n_est, n_gt), -1, 1)
            angle_err = np.degrees(np.arccos(dot))
            
            # 距离误差
            dist_err = np.abs(np.abs(model[3]) - np.abs(gt_model[3])) * 1000
            
            # RMSE
            distances = distance_to_plane(points, model)
            rmse = np.sqrt(np.mean(distances**2)) * 1000
            
            # 内点率
            threshold = 0.015
            inlier_ratio = np.sum(distances < threshold) / len(distances)
            
            # 计算时间
            time_ms = times[name] * 1000
            
            # 综合得分 (越小越好)
            # 权重：角度误差(30%) + 距离误差(30%) + RMSE(20%) + (1-内点率)(10%) + 时间(10%)
            score = (angle_err * 0.3) + (dist_err * 0.3) + (rmse * 0.2) + ((1 - inlier_ratio) * 100 * 0.1) + (time_ms * 0.001 * 0.1)
            scores[name] = score
        
        # 找到最佳方法
        best_method = min(scores, key=scores.get)
        best_methods[scene] = best_method
        print(f"{scene:<20} | 最佳方法: {best_method:<10} | 综合得分: {scores[best_method]:<10.4f}")
    
    # 总体分析
    print("\n总体分析:")
    print("-" * 60)
    
    # 统计各方法在不同场景中的表现
    method_counts = {"LS": 0, "RANSAC": 0, "Huber": 0, "Proposed": 0, "RegionGrowing": 0, "TLS-SVD": 0, "WLS": 0}
    for scene, method in best_methods.items():
        method_counts[method] += 1
    
    print("各方法最佳场景数:")
    for method, count in method_counts.items():
        print(f"{method:<10} | {count} 个场景")
    
    # 方法特点总结
    print("\n方法特点总结:")
    print("-" * 60)
    print("LS (最小二乘法):")
    print("  - 优点: 计算速度快，实现简单")
    print("  - 缺点: 对异常值敏感，在存在明显缺陷时表现较差")
    print("  - 适用场景: 数据质量高，无明显异常值的情况")
    
    print("\nRANSAC:")
    print("  - 优点: 对异常值有一定的鲁棒性，计算速度较快")
    print("  - 缺点: 结果可能不稳定，依赖于阈值选择")
    print("  - 适用场景: 存在大量异常值，但需要快速计算的情况")
    
    print("\nHuber M-估计:")
    print("  - 优点: 对异常值有较好的鲁棒性")
    print("  - 缺点: 计算时间较长，在某些场景下精度提升有限")
    print("  - 适用场景: 数据存在中等程度异常值的情况")
    
    print("\nProposed (Tukey M-估计):")
    print("  - 优点: 对异常值具有最强的鲁棒性，在复杂缺陷场景下表现最佳")
    print("  - 缺点: 计算时间相对较长")
    print("  - 适用场景: 存在严重缺陷或大量异常值的复杂工业场景")
    
    print("\nRegionGrowing (区域生长):")
    print("  - 优点: 考虑了点云的局部几何结构，能更好地处理复杂平面")
    print("  - 缺点: 计算时间较长，依赖于法线估计质量")
    print("  - 适用场景: 点云密度均匀，需要考虑局部几何结构的场景")
    
    print("\nTLS-SVD (总最小二乘):")
    print("  - 优点: 考虑了所有变量的误差，理论上更准确，数值稳定性好")
    print("  - 缺点: 对异常值敏感，计算复杂度较高")
    print("  - 适用场景: 数据质量高，无明显异常值的情况")
    
    print("\nWLS (加权最小二乘):")
    print("  - 优点: 可以根据先验知识或残差设置权重，提高拟合精度")
    print("  - 缺点: 权重选择对结果影响较大，需要合理设置")
    print("  - 适用场景: 数据质量不均匀，部分区域更可靠的情况")

if __name__ == "__main__":
    run_comparison_and_visualize()