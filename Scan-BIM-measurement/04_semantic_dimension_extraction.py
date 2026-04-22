import os
import json
import logging
import copy
import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple, Optional, Any

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 尝试导入平面拟合模块
try:
    # 使用动态导入，因为模块名以数字开头
    import importlib.util
    spec = importlib.util.spec_from_file_location("PlaneFitter", "01_robust_fitting.py")
    robust_fitting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(robust_fitting)
    PlaneFitter = robust_fitting.PlaneFitter
except Exception as e:
    logger.error(f"无法导入01_robust_fitting.py: {str(e)}")
    raise

class SemanticDimensionExtractor:
    """
    语义维度提取器，用于从配准后的点云中提取关键尺寸并进行偏差分析
    """
    
    def __init__(self, scan_path: str, roi_path: str, tasks_path: str, visualize: bool = True, window_width: int = 1024, window_height: int = 768, line_color: str = "red", line_width: float = 30.0):
        """
        初始化语义维度提取器
        
        Args:
            scan_path: 原始扫描点云文件路径
            roi_path: ROI定义文件路径
            tasks_path: 测量任务文件路径
            visualize: 是否启用可视化
            window_width: 可视化窗口宽度
            window_height: 可视化窗口高度
            line_color: 平面间垂线颜色，可选值: "red", "green", "blue", "black"
            line_width: 平面间垂线宽度
        """
        self.scan_path = scan_path
        self.roi_path = roi_path
        self.tasks_path = tasks_path
        self.visualize = visualize
        self.window_width = window_width
        self.window_height = window_height
        self.line_width = line_width
        
        # 验证并设置线条颜色
        valid_colors = {"red": [1.0, 0.0, 0.0], "green": [0.0, 1.0, 0.0], "blue": [0.0, 0.0, 1.0], "black": [0.0, 0.0, 0.0]}
        if line_color.lower() in valid_colors:
            self.line_color = valid_colors[line_color.lower()]
        else:
            logger.warning(f"无效的线条颜色: {line_color}，使用默认值黑色")
            self.line_color = valid_colors["black"]
        
        self.scan_pcd: Optional[o3d.geometry.PointCloud] = None
        self.roi_definitions: Optional[Dict[str, Any]] = None
        self.measurement_tasks: Optional[List[Dict[str, Any]]] = None
        
        self.fitted_planes: Dict[str, np.ndarray] = {}
        self.measurement_results: List[Dict[str, Any]] = []
    
    def visualize_roi_and_plane(self, roi_name: str, obb: o3d.geometry.OrientedBoundingBox, 
                               cropped_pcd: o3d.geometry.PointCloud, plane_model: np.ndarray):
        """
        可视化ROI区域、裁剪的点云和拟合的平面
        
        Args:
            roi_name: ROI名称
            obb: 定向包围盒
            cropped_pcd: 裁剪后的点云
            plane_model: 拟合的平面方程参数 [A, B, C, D]
        """
        if not self.visualize:
            return
        
        try:
            # 创建可视化对象
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"ROI Visualization: {roi_name}", 
                            width=self.window_width, height=self.window_height)
            
            # 添加原始点云（半透明）
            original_pcd = copy.deepcopy(self.scan_pcd)
            original_pcd.paint_uniform_color([0.8, 0.8, 0.8])
            original_pcd.voxel_down_sample(voxel_size=0.05)  # 下采样以提高性能
            vis.add_geometry(original_pcd)
            
            # 添加ROI包围盒（使用点和线来创建）
            # 手动创建包围盒的线框
            center = obb.get_center()
            extent = obb.extent
            R = obb.R
            
            # 生成包围盒的8个顶点
            half_extent = extent / 2
            vertices = np.array([
                [-half_extent[0], -half_extent[1], -half_extent[2]],
                [half_extent[0], -half_extent[1], -half_extent[2]],
                [half_extent[0], half_extent[1], -half_extent[2]],
                [-half_extent[0], half_extent[1], -half_extent[2]],
                [-half_extent[0], -half_extent[1], half_extent[2]],
                [half_extent[0], -half_extent[1], half_extent[2]],
                [half_extent[0], half_extent[1], half_extent[2]],
                [-half_extent[0], half_extent[1], half_extent[2]]
            ])
            
            # 应用旋转和平移
            vertices = (R @ vertices.T).T + center
            
            # 创建线框
            lines = [
                [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
                [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
                [0, 4], [1, 5], [2, 6], [3, 7]   # 侧面
            ]
            
            obb_mesh = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(vertices),
                lines=o3d.utility.Vector2iVector(lines)
            )
            obb_mesh.paint_uniform_color([1.0, 0.0, 0.0])
            vis.add_geometry(obb_mesh)
            
            # 添加裁剪后的点云
            if cropped_pcd.has_points():
                vis.add_geometry(cropped_pcd)
            
            # 添加拟合的平面（二维平面）
            if cropped_pcd.has_points():
                # 获取裁剪点云的边界框
                bbox = cropped_pcd.get_axis_aligned_bounding_box()
                center = bbox.get_center()
                
                # 固定平面大小为0.5x0.5（单位：米）
                plane_size = 0.5
                
                # 创建二维平面网格
                # 使用TriangleMesh.create_box创建一个非常薄的平面
                plane = o3d.geometry.TriangleMesh.create_box(width=plane_size, 
                                                            height=plane_size, 
                                                            depth=0.001)  # 非常薄的深度
                plane.translate([center[0] - plane_size/2, 
                                center[1] - plane_size/2, 
                                center[2]])
                
                # 根据平面方程调整平面方向
                A, B, C, D = plane_model
                normal = np.array([A, B, C])
                normal = normal / np.linalg.norm(normal)
                
                # 计算平面的旋转矩阵
                # 默认平面法线是Z轴方向，我们需要旋转到目标法线方向
                z_axis = np.array([0, 0, 1])
                if np.dot(normal, z_axis) < 0.999:
                    # 计算旋转轴和角度
                    axis = np.cross(z_axis, normal)
                    axis = axis / np.linalg.norm(axis)
                    angle = np.arccos(np.dot(z_axis, normal))
                    # 应用旋转
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                    plane.rotate(R, center=center)
                
                # 使用与其他可视化方法一致的蓝色
                plane.paint_uniform_color([0.0, 0.0, 1.0])
                plane.compute_vertex_normals()
                vis.add_geometry(plane)
            
            # 设置视角
            vis.get_view_control().set_front([-0.1, -0.1, -1.0])
            vis.get_view_control().set_lookat(obb.get_center())
            vis.get_view_control().set_up([0.0, 1.0, 0.0])
            
            # 运行可视化
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            logger.error(f"可视化ROI {roi_name} 时出错: {str(e)}")
    
    def visualize_measurement(self, task_id: str, task_name: str, 
                             plane1: np.ndarray, plane2: np.ndarray, 
                             distance: float):
        """
        可视化测量任务，包括拟合平面和距离
        
        Args:
            task_id: 任务ID
            task_name: 任务名称
            plane1: 第一个平面的方程参数 [A, B, C, D]
            plane2: 第二个平面的方程参数 [A, B, C, D]
            distance: 两个平面之间的距离
        """
        if not self.visualize:
            return
        
        try:
            # 创建可视化对象
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=f"Measurement Visualization: {task_name}", 
                            width=self.window_width, height=self.window_height)
            
            # 设置线条宽度
            render_option = vis.get_render_option()
            render_option.line_width = self.line_width
            
            # 添加原始点云（半透明）
            original_pcd = copy.deepcopy(self.scan_pcd)
            original_pcd.paint_uniform_color([0.8, 0.8, 0.8])
            original_pcd.voxel_down_sample(voxel_size=0.05)  # 下采样以提高性能
            vis.add_geometry(original_pcd)
            
            # 为两个平面创建网格
            planes = []
            colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]  # 红色和绿色
            
            # 计算两个平面的中心点，用于设置视角
            plane_centers = []
            
            # 固定平面大小为0.5x0.5（单位：米）
            plane_size = 0.5
            
            for i, (plane_model, color) in enumerate(zip([plane1, plane2], colors)):
                A, B, C, D = plane_model
                normal = np.array([A, B, C])
                norm_normal = np.linalg.norm(normal)
                normal_normalized = normal / norm_normal
                
                # 使用平面方程直接计算平面上的点
                # 平面方程：Ax + By + Cz + D = 0
                # 原点到平面的最近点：p = -D * n / ||n||²
                plane_center = -D * normal / (norm_normal ** 2)
                
                # 创建平面网格
                plane = o3d.geometry.TriangleMesh.create_box(width=plane_size, height=plane_size, depth=0.001)
                plane.translate([-plane_size/2, -plane_size/2, 0])
                
                # 计算平面的旋转矩阵
                z_axis = np.array([0, 0, 1])
                if np.dot(normal_normalized, z_axis) < 0.999:
                    # 计算旋转轴和角度
                    axis = np.cross(z_axis, normal_normalized)
                    axis = axis / np.linalg.norm(axis)
                    angle = np.arccos(np.dot(z_axis, normal_normalized))
                    # 应用旋转
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                    plane.rotate(R, center=[0, 0, 0])
                
                # 将平面移动到平面中心位置
                plane.translate(plane_center)
                
                plane.paint_uniform_color(color)
                plane.compute_vertex_normals()
                vis.add_geometry(plane)
                planes.append(plane)
                plane_centers.append(plane_center)
            
            # 计算两个平面之间的垂线，用于可视化距离
            if len(planes) == 2:
                # 提取两个平面的法向量
                A1, B1, C1, D1 = plane1
                normal1 = np.array([A1, B1, C1])
                norm_normal1 = np.linalg.norm(normal1)
                normal1_normalized = normal1 / norm_normal1
                
                A2, B2, C2, D2 = plane2
                normal2 = np.array([A2, B2, C2])
                norm_normal2 = np.linalg.norm(normal2)
                normal2_normalized = normal2 / norm_normal2
                
                # 计算两个平面的中心点
                center1 = -D1 * normal1 / (norm_normal1 ** 2)
                center2 = -D2 * normal2 / (norm_normal2 ** 2)
                
                # 计算垂线方向（使用法向量方向）
                # 对于平行平面，垂线方向就是法向量方向
                # 修正垂线方向，确保从第一个平面指向第二个平面
                line_direction = center2 - center1
                line_direction = line_direction / np.linalg.norm(line_direction)
                
                # 计算垂线上的两个点，确保它们分别在两个平面上
                point1 = center1
                point2 = center2
                
                # 创建垂线
                line_points = [point1, point2]
                line = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(line_points),
                    lines=o3d.utility.Vector2iVector([[0, 1]])
                )
                line.paint_uniform_color(self.line_color)  # 使用配置的线条颜色
                vis.add_geometry(line)
                
                # 添加距离文本（在Open3D中直接添加文本比较复杂，这里我们只是显示在控制台）
                logger.info(f"测量任务 {task_id} ({task_name}) 的距离: {distance:.4f} m")
            
            # 设置视角
            if plane_centers:
                # 计算中心点
                view_center = np.mean(plane_centers, axis=0)
                vis.get_view_control().set_front([-0.1, -0.1, -1.0])
                vis.get_view_control().set_lookat(view_center)
                vis.get_view_control().set_up([0.0, 1.0, 0.0])
            else:
                vis.get_view_control().set_front([-0.1, -0.1, -1.0])
                vis.get_view_control().set_lookat([0, 0, 0])
                vis.get_view_control().set_up([0.0, 1.0, 0.0])
            
            # 运行可视化
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            logger.error(f"可视化测量任务 {task_id} 时出错: {str(e)}")
    
    def load_data(self) -> bool:
        """
        加载所有输入数据
        
        Returns:
            bool: 数据加载是否成功
        """
        try:
            # 加载原始扫描点云
            logger.info(f"加载原始扫描点云: {self.scan_path}")
            self.scan_pcd = o3d.io.read_point_cloud(self.scan_path)
            if not self.scan_pcd.has_points():
                logger.error("扫描点云为空")
                return False
            logger.info(f"成功加载点云，包含 {len(self.scan_pcd.points)} 个点")
            
            # 显示点云边界框信息
            self.show_point_cloud_bounds()
            
            # 加载ROI定义
            logger.info(f"加载ROI定义: {self.roi_path}")
            with open(self.roi_path, 'r', encoding='utf-8') as f:
                roi_data = json.load(f)
                self.roi_definitions = roi_data.get('rois', {})
            logger.info(f"成功加载 {len(self.roi_definitions)} 个ROI定义")
            
            # 加载测量任务
            logger.info(f"加载测量任务: {self.tasks_path}")
            with open(self.tasks_path, 'r', encoding='utf-8') as f:
                tasks_data = json.load(f)
                self.measurement_tasks = tasks_data.get('tasks', [])
            logger.info(f"成功加载 {len(self.measurement_tasks)} 个测量任务")
            
            return True
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            return False
    
    def show_point_cloud_bounds(self):
        """
        显示点云的边界框信息
        """
        if self.scan_pcd is None or not self.scan_pcd.has_points():
            logger.error("点云为空，无法显示边界框")
            return
        
        # 计算点云的边界框
        bbox = self.scan_pcd.get_axis_aligned_bounding_box()
        min_bound = bbox.min_bound
        max_bound = bbox.max_bound
        center = bbox.get_center()
        extent = bbox.get_extent()
        
        logger.info("点云边界框信息:")
        logger.info(f"最小坐标: {min_bound}")
        logger.info(f"最大坐标: {max_bound}")
        logger.info(f"中心点: {center}")
        logger.info(f"尺寸: {extent}")
        
        # 可视化点云和边界框
        if self.visualize:
            try:
                # 创建可视化对象
                vis = o3d.visualization.VisualizerWithEditing()
                vis.create_window(window_name="Point Cloud Bounds", 
                                width=self.window_width, height=self.window_height)
                
                # 添加点云
                vis.add_geometry(self.scan_pcd)
                
                # 添加边界框
                bbox.color = (1, 0, 0)  # 红色边界框
                vis.add_geometry(bbox)
                
                # 设置视角
                vis.get_view_control().set_front([-0.1, -0.1, -1.0])
                vis.get_view_control().set_lookat(center)
                vis.get_view_control().set_up([0.0, 1.0, 0.0])
                
                # 运行可视化，支持点选择
                logger.info("正在显示点云边界框...")
                logger.info("提示: 点击点云可选择点，按 'Q' 键退出")
                vis.run()
                
                # 获取选择的点
                selected_ids = vis.get_picked_points()
                if selected_ids:
                    logger.info(f"选择了 {len(selected_ids)} 个点:")
                    points = np.asarray(self.scan_pcd.points)
                    for i, idx in enumerate(selected_ids[:5]):  # 只显示前5个点
                        logger.info(f"点 {idx}: {points[idx]}")
                    if len(selected_ids) > 5:
                        logger.info(f"... 还有 {len(selected_ids) - 5} 个点未显示")
                
                vis.destroy_window()
                
            except Exception as e:
                logger.error(f"可视化点云边界框时出错: {str(e)}")
    
    def transform_roi_to_scan(self, roi_def: Dict[str, Any]) -> o3d.geometry.OrientedBoundingBox:
        """
        直接使用ROI定义创建定向包围盒
        
        Args:
            roi_def: ROI定义字典
            
        Returns:
            o3d.geometry.OrientedBoundingBox: 定向包围盒
        """
        # 提取ROI参数
        center = np.array(roi_def['obb_center'])
        extent = np.array(roi_def['obb_extent'])
        rotation = np.array(roi_def['obb_rotation'])
        
        # 直接创建定向包围盒
        # 注意：Open3D的OrientedBoundingBox构造函数期望旋转矩阵是列主序的
        obb_scan = o3d.geometry.OrientedBoundingBox(
            center=center,
            R=rotation,
            extent=extent
        )
        
        return obb_scan
    
    def crop_roi_points(self, obb: o3d.geometry.OrientedBoundingBox) -> o3d.geometry.PointCloud:
        """
        裁剪包围盒内部的点云
        
        Args:
            obb: 定向包围盒
            
        Returns:
            o3d.geometry.PointCloud: 裁剪后的点云
        """
        # 使用Open3D的crop方法裁剪点云
        cropped_pcd = self.scan_pcd.crop(obb)
        logger.info(f"裁剪后点云包含 {len(cropped_pcd.points)} 个点")
        return cropped_pcd
    
    def fit_plane_in_roi(self, roi_name: str, roi_def: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        在ROI内拟合平面
        
        Args:
            roi_name: ROI名称
            roi_def: ROI定义字典
            
        Returns:
            Optional[np.ndarray]: 拟合的平面方程参数 [A, B, C, D]
        """
        try:
            # 转换ROI到Scan坐标系
            obb_scan = self.transform_roi_to_scan(roi_def)
            
            # 裁剪点云
            cropped_pcd = self.crop_roi_points(obb_scan)
            
            if not cropped_pcd.has_points():
                logger.warning(f"ROI {roi_name} 内没有点云，无法拟合平面")
                return None
            
            # 提取点云数据
            points = np.asarray(cropped_pcd.points)
            
            # 使用稳健的平面拟合方法
            # 这里使用01_robust_fitting.py中的Tukey M-估计方法
            plane_model = PlaneFitter.fit_robust_tukey(points)
            
            # 统一法向量方向，使其与预期法向量一致
            expected_normal = np.array(roi_def.get('expected_normal', [0, 0, 1]))
            plane_model = PlaneFitter._unify_direction(plane_model, expected_normal)
            
            # 优化平面拟合：约束法向量为坐标轴方向
            # 检测平面是否接近平行于坐标轴
            A, B, C, D = plane_model
            normal = np.array([A, B, C])
            normal = normal / np.linalg.norm(normal)
            
            # 计算与坐标轴的夹角
            angles = []
            axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            for axis in axes:
                angle = np.arccos(np.abs(np.dot(normal, axis)))
                angles.append(angle)
            
            # 如果与某个坐标轴的夹角小于10度，则约束为该坐标轴方向
            threshold = np.deg2rad(10)  # 10度阈值
            if min(angles) < threshold:
                axis_idx = np.argmin(angles)
                constrained_normal = np.zeros(3)
                constrained_normal[axis_idx] = 1.0
                
                # 重新计算平面方程
                # 使用最小二乘法，约束法向量为坐标轴方向
                if axis_idx == 0:  # X轴方向
                    # 平面方程: x = d
                    d = np.mean(points[:, 0])
                    plane_model = np.array([1.0, 0.0, 0.0, -d])
                elif axis_idx == 1:  # Y轴方向
                    # 平面方程: y = d
                    d = np.mean(points[:, 1])
                    plane_model = np.array([0.0, 1.0, 0.0, -d])
                else:  # Z轴方向
                    # 平面方程: z = d
                    d = np.mean(points[:, 2])
                    plane_model = np.array([0.0, 0.0, 1.0, -d])
                
                logger.info(f"ROI {roi_name} 平面拟合优化: 约束为{['X', 'Y', 'Z'][axis_idx]}轴方向")
            
            logger.info(f"ROI {roi_name} 平面拟合成功: {PlaneFitter.format_equation(plane_model)}")
            
            # 可视化ROI、裁剪点云和拟合平面
            self.visualize_roi_and_plane(roi_name, obb_scan, cropped_pcd, plane_model)
            
            return plane_model
        except Exception as e:
            logger.error(f"ROI {roi_name} 平面拟合失败: {str(e)}")
            return None
    
    def calculate_distance_plane_to_plane(self, plane1: np.ndarray, plane2: np.ndarray) -> float:
        """
        计算两个平面之间的距离
        
        数学原理：对于两个平面 A1x + B1y + C1z + D1 = 0 和 A2x + B2y + C2z + D2 = 0，
        如果它们平行（法向量相同或相反），则距离为 |D1 - D2| / sqrt(A² + B² + C²)
        
        Args:
            plane1: 第一个平面的方程参数 [A1, B1, C1, D1]
            plane2: 第二个平面的方程参数 [A2, B2, C2, D2]
            
        Returns:
            float: 两个平面之间的距离
        """
        # 提取法向量和常数项
        n1 = plane1[:3]
        d1 = plane1[3]
        n2 = plane2[:3]
        d2 = plane2[3]
        
        # 确保法向量已归一化
        norm_n1 = np.linalg.norm(n1)
        norm_n2 = np.linalg.norm(n2)
        
        if norm_n1 < 1e-6 or norm_n2 < 1e-6:
            logger.error("平面法向量无效")
            return 0.0
        
        # 归一化法向量
        n1_normalized = n1 / norm_n1
        n2_normalized = n2 / norm_n2
        
        # 检查两个平面是否平行
        dot_product = np.dot(n1_normalized, n2_normalized)
        if abs(abs(dot_product) - 1.0) > 1e-6:
            logger.warning("两个平面不平行，计算的是最短距离")
        
        # 计算距离
        # 注意：由于平面方程可能已经归一化，这里需要重新计算
        # 正确的距离公式是 |D1 - D2| / sqrt(A² + B² + C²)，其中D是归一化后的常数项
        distance = abs((d1 / norm_n1) - (d2 / norm_n2))
        
        return distance
    
    def process_measurement_tasks(self) -> None:
        """
        处理测量任务，计算尺寸并分析偏差
        """
        for task in self.measurement_tasks:
            try:
                task_id = task['task_id']
                task_name = task['name']
                task_type = task['type']
                source_rois = task['source_rois']
                nominal_value = task['nominal_value']
                tolerance = task['tolerance']
                
                logger.info(f"处理测量任务: {task_name} ({task_id})")
                
                # 检查所有源ROI是否都已拟合
                missing_rois = [roi for roi in source_rois if roi not in self.fitted_planes]
                if missing_rois:
                    logger.warning(f"任务 {task_id} 缺少ROI: {missing_rois}，跳过此任务")
                    continue
                
                # 根据任务类型计算尺寸
                measured_value = 0.0
                
                if task_type == 'distance_plane_to_plane':
                    if len(source_rois) != 2:
                        logger.error(f"任务 {task_id} 需要两个ROI来计算平面间距离")
                        continue
                    
                    plane1 = self.fitted_planes[source_rois[0]]
                    plane2 = self.fitted_planes[source_rois[1]]
                    measured_value = self.calculate_distance_plane_to_plane(plane1, plane2)
                    
                    # 可视化测量任务
                    self.visualize_measurement(task_id, task_name, plane1, plane2, measured_value)
                
                else:
                    logger.warning(f"不支持的任务类型: {task_type}")
                    continue
                
                # 计算偏差
                deviation = measured_value - nominal_value
                within_tolerance = abs(deviation) <= tolerance
                
                # 记录结果
                result = {
                    'task_id': task_id,
                    'name': task_name,
                    'type': task_type,
                    'source_rois': source_rois,
                    'measured_value': measured_value,
                    'nominal_value': nominal_value,
                    'deviation': deviation,
                    'tolerance': tolerance,
                    'within_tolerance': bool(within_tolerance)  # 转换为Python原生布尔类型
                }
                
                self.measurement_results.append(result)
                logger.info(f"任务 {task_id} 完成: 测量值 = {measured_value:.4f} m, 偏差 = {deviation:.4f} m, "
                         f"公差 = ±{tolerance} m, {'在公差范围内' if within_tolerance else '超出公差范围'}")
                
            except Exception as e:
                logger.error(f"处理任务 {task.get('task_id', '未知')} 时出错: {str(e)}")
                continue
    
    def visualize_tasks_roi(self):
        """
        按照测量任务分类可视化ROI区域和点云
        """
        if not self.visualize:
            return
        
        try:
            for task in self.measurement_tasks:
                task_id = task['task_id']
                task_name = task['name']
                source_rois = task['source_rois']
                
                logger.info(f"可视化任务 {task_id} ({task_name}) 的ROI区域")
                
                # 创建可视化对象
                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name=f"Task ROI Visualization: {task_name}", 
                                width=self.window_width, height=self.window_height)
                
                # 添加原始点云（半透明）
                original_pcd = copy.deepcopy(self.scan_pcd)
                original_pcd.paint_uniform_color([0.8, 0.8, 0.8])
                original_pcd.voxel_down_sample(voxel_size=0.05)  # 下采样以提高性能
                vis.add_geometry(original_pcd)
                
                # 添加任务相关的ROI包围盒
                for roi_name in source_rois:
                    if roi_name in self.roi_definitions:
                        roi_def = self.roi_definitions[roi_name]
                        obb = self.transform_roi_to_scan(roi_def)
                        
                        # 手动创建包围盒的线框
                        center = obb.get_center()
                        extent = obb.extent
                        R = obb.R
                        
                        # 生成包围盒的8个顶点
                        half_extent = extent / 2
                        vertices = np.array([
                            [-half_extent[0], -half_extent[1], -half_extent[2]],
                            [half_extent[0], -half_extent[1], -half_extent[2]],
                            [half_extent[0], half_extent[1], -half_extent[2]],
                            [-half_extent[0], half_extent[1], -half_extent[2]],
                            [-half_extent[0], -half_extent[1], half_extent[2]],
                            [half_extent[0], -half_extent[1], half_extent[2]],
                            [half_extent[0], half_extent[1], half_extent[2]],
                            [-half_extent[0], half_extent[1], half_extent[2]]
                        ])
                        
                        # 应用旋转和平移
                        vertices = (R @ vertices.T).T + center
                        
                        # 创建线框
                        lines = [
                            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
                            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
                            [0, 4], [1, 5], [2, 6], [3, 7]   # 侧面
                        ]
                        
                        obb_mesh = o3d.geometry.LineSet(
                            points=o3d.utility.Vector3dVector(vertices),
                            lines=o3d.utility.Vector2iVector(lines)
                        )
                        # 为不同的ROI使用不同的颜色
                        colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
                        color_idx = source_rois.index(roi_name) % len(colors)
                        obb_mesh.paint_uniform_color(colors[color_idx])
                        vis.add_geometry(obb_mesh)
                
                # 设置视角
                vis.get_view_control().set_front([-0.1, -0.1, -1.0])
                vis.get_view_control().set_lookat([0, 0, 0])
                vis.get_view_control().set_up([0.0, 1.0, 0.0])
                
                # 运行可视化
                vis.run()
                vis.destroy_window()
                
        except Exception as e:
            logger.error(f"可视化任务ROI时出错: {str(e)}")
    
    def visualize_all_dimensions(self):
        """
        可视化所有计算的尺寸线与点云
        """
        if not self.visualize or not self.measurement_results:
            return
        
        try:
            # 创建可视化对象
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="All Dimensions Visualization", 
                            width=self.window_width, height=self.window_height)
            
            # 设置线条宽度
            render_option = vis.get_render_option()
            render_option.line_width = self.line_width
            
            # 添加原始点云（半透明）
            original_pcd = copy.deepcopy(self.scan_pcd)
            original_pcd.paint_uniform_color([0.8, 0.8, 0.8])
            original_pcd.voxel_down_sample(voxel_size=0.05)  # 下采样以提高性能
            vis.add_geometry(original_pcd)
            
            # 添加所有拟合的平面
            # 固定平面大小为0.5x0.5（单位：米）
            plane_size = 0.5
            
            for roi_name, plane_model in self.fitted_planes.items():
                A, B, C, D = plane_model
                normal = np.array([A, B, C])
                norm_normal = np.linalg.norm(normal)
                normal_normalized = normal / norm_normal
                
                # 使用平面方程直接计算平面上的点
                plane_center = -D * normal / (norm_normal ** 2)
                
                # 创建平面网格
                plane = o3d.geometry.TriangleMesh.create_box(width=plane_size, height=plane_size, depth=0.001)
                plane.translate([-plane_size/2, -plane_size/2, 0])
                
                # 计算平面的旋转矩阵
                z_axis = np.array([0, 0, 1])
                if np.dot(normal_normalized, z_axis) < 0.999:
                    # 计算旋转轴和角度
                    axis = np.cross(z_axis, normal_normalized)
                    axis = axis / np.linalg.norm(axis)
                    angle = np.arccos(np.dot(z_axis, normal_normalized))
                    # 应用旋转
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                    plane.rotate(R, center=[0, 0, 0])
                
                # 将平面移动到平面中心位置
                plane.translate(plane_center)
                
                plane.paint_uniform_color([0.0, 0.0, 1.0])
                plane.compute_vertex_normals()
                vis.add_geometry(plane)
            
            # 添加所有测量任务的尺寸线
            for result in self.measurement_results:
                task_id = result['task_id']
                task_name = result['name']
                source_rois = result['source_rois']
                
                if len(source_rois) == 2 and all(roi in self.fitted_planes for roi in source_rois):
                    plane1 = self.fitted_planes[source_rois[0]]
                    plane2 = self.fitted_planes[source_rois[1]]
                    
                    # 计算两个平面的中心点
                    A1, B1, C1, D1 = plane1
                    normal1 = np.array([A1, B1, C1])
                    norm_normal1 = np.linalg.norm(normal1)
                    center1 = -D1 * normal1 / (norm_normal1 ** 2)
                    
                    A2, B2, C2, D2 = plane2
                    normal2 = np.array([A2, B2, C2])
                    norm_normal2 = np.linalg.norm(normal2)
                    center2 = -D2 * normal2 / (norm_normal2 ** 2)
                    
                    # 创建尺寸线
                    line_points = [center1, center2]
                    line = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(line_points),
                        lines=o3d.utility.Vector2iVector([[0, 1]])
                    )
                    line.paint_uniform_color(self.line_color)  # 使用配置的线条颜色
                    vis.add_geometry(line)
            
            # 设置视角
            vis.get_view_control().set_front([-0.1, -0.1, -1.0])
            vis.get_view_control().set_lookat([0, 0, 0])
            vis.get_view_control().set_up([0.0, 1.0, 0.0])
            
            # 运行可视化
            vis.run()
            vis.destroy_window()
            
        except Exception as e:
            logger.error(f"可视化所有尺寸时出错: {str(e)}")
    
    def run(self) -> bool:
        """
        运行完整的语义维度提取流程
        
        Returns:
            bool: 流程是否成功完成
        """
        try:
            # 加载数据
            if not self.load_data():
                return False
            
            # 按照任务分类可视化ROI区域和点云
            logger.info("可视化任务ROI区域")
            self.visualize_tasks_roi()
            
            # 拟合每个ROI内的平面
            logger.info("开始拟合ROI内的平面")
            for roi_name, roi_def in self.roi_definitions.items():
                plane_model = self.fit_plane_in_roi(roi_name, roi_def)
                if plane_model is not None:
                    self.fitted_planes[roi_name] = plane_model
            
            # 处理测量任务
            logger.info("开始处理测量任务")
            self.process_measurement_tasks()
            
            # 可视化所有计算的尺寸线与点云
            logger.info("可视化所有尺寸")
            self.visualize_all_dimensions()
            
            # 生成报告
            self.generate_report()
            
            return True
        except Exception as e:
            logger.error(f"运行失败: {str(e)}")
            return False
    
    def generate_report(self) -> None:
        """
        生成测量报告
        """
        logger.info("生成测量报告")
        
        # 统计信息
        total_tasks = len(self.measurement_tasks)
        completed_tasks = len(self.measurement_results)
        within_tolerance_tasks = sum(1 for r in self.measurement_results if r['within_tolerance'])
        
        # 打印报告
        logger.info("=" * 80)
        logger.info("语义维度提取报告")
        logger.info("=" * 80)
        logger.info(f"总任务数: {total_tasks}")
        logger.info(f"完成任务数: {completed_tasks}")
        logger.info(f"在公差范围内的任务数: {within_tolerance_tasks}")
        logger.info("-" * 80)
        
        for result in self.measurement_results:
            logger.info(f"任务ID: {result['task_id']}")
            logger.info(f"任务名称: {result['name']}")
            logger.info(f"测量值: {result['measured_value']:.4f} m")
            logger.info(f"标准值: {result['nominal_value']:.4f} m")
            logger.info(f"偏差: {result['deviation']:.4f} m")
            logger.info(f"公差: ±{result['tolerance']} m")
            logger.info(f"状态: {'在公差范围内' if result['within_tolerance'] else '超出公差范围'}")
            logger.info("-" * 80)
        
        # 从输入文件名生成输出文件前缀
        filename = os.path.basename(self.scan_path)
        prefix = os.path.splitext(filename)[0]
        output_file = f'{prefix}_measurement_results.json'
        
        # 保存结果到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_tasks': total_tasks,
                    'completed_tasks': completed_tasks,
                    'within_tolerance_tasks': within_tolerance_tasks
                },
                'results': self.measurement_results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"测量结果已保存到 {output_file}")

if __name__ == "__main__":
    # 默认文件路径
    default_scan_path = r"scan_path"
    default_roi_path = r"roi_path"
    default_tasks_path = r"tasks_path"
    
    # 检查文件是否存在
    for file_path in [default_scan_path, default_roi_path, default_tasks_path]:
        if not os.path.exists(file_path):
            logger.warning(f"文件 {file_path} 不存在，使用默认路径")
    
    # 创建提取器实例
    extractor = SemanticDimensionExtractor(
        scan_path=default_scan_path,
        roi_path=default_roi_path,
        tasks_path=default_tasks_path,
        visualize=True,  # 启用可视化
        line_width=5.0   # 设置线条宽度为5.0，使其更明显
    )
    
    # 运行提取流程
    success = extractor.run()
    
    if success:
        logger.info("语义维度提取完成")
    else:
        logger.error("语义维度提取失败")
