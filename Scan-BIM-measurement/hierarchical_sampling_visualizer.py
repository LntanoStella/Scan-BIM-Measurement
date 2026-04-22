import os
import argparse
import numpy as np
import open3d as o3d
from sample import HierarchicalStructurePreservingSampling

def read_point_cloud(file_path):
    """
    读取点云文件，支持多种格式
    
    Args:
        file_path: 点云文件路径
    
    Returns:
        open3d.geometry.PointCloud: 读取的点云
    """
    # 获取文件扩展名
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.txt':
        # 读取 TXT 格式点云
        points = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 3:
                            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                            points.append([x, y, z])
            
            if not points:
                raise ValueError("TXT 文件中没有点数据")
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            return pcd
        except Exception as e:
            print(f"读取 TXT 文件失败: {e}")
            return o3d.geometry.PointCloud()
    else:
        # 使用 Open3D 内置函数读取其他格式
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            return pcd
        except Exception as e:
            print(f"读取点云文件失败: {e}")
            return o3d.geometry.PointCloud()

def visualize_hierarchical_sampling(pcd, sampled_points, point_types, save_path=None, visualize=True, window_size=(800, 600)):
    """
    可视化分层采样结果
    
    Args:
        pcd: 原始点云
        sampled_points: 采样后的点云
        point_types: 点类型 (0=平面, 1=边缘, 2=角点)
        save_path: 保存路径，如果为None则不保存
        visualize: 是否开启可视化
        window_size: 窗口大小，格式为 (width, height)
    """
    # 创建不同类型点的点云
    plane_points = sampled_points[point_types == 0]
    edge_points = sampled_points[point_types == 1]
    corner_points = sampled_points[point_types == 2]
    
    # 创建点云对象
    plane_pcd = o3d.geometry.PointCloud()
    edge_pcd = o3d.geometry.PointCloud()
    corner_pcd = o3d.geometry.PointCloud()
    
    # 添加点
    if len(plane_points) > 0:
        plane_pcd.points = o3d.utility.Vector3dVector(plane_points)
        plane_pcd.paint_uniform_color([0, 0, 1])  # 蓝色 - 平面点
    
    if len(edge_points) > 0:
        edge_pcd.points = o3d.utility.Vector3dVector(edge_points)
        edge_pcd.paint_uniform_color([0, 1, 0])  # 绿色 - 边缘点
    
    if len(corner_points) > 0:
        corner_pcd.points = o3d.utility.Vector3dVector(corner_points)
        corner_pcd.paint_uniform_color([1, 0, 0])  # 红色 - 角点
    
    print("可视化分层采样结果...")
    print(f"平面点数量: {len(plane_points)}")
    print(f"边缘点数量: {len(edge_points)}")
    print(f"角点数量: {len(corner_points)}")
    
    # 可视化
    if visualize:
        # 平面点可视化窗口
        if len(plane_points) > 0:
            plane_geometries = []
            # 只添加平面点云
            plane_geometries.append(plane_pcd)
            
            # 创建可视化窗口
            vis_plane = o3d.visualization.Visualizer()
            vis_plane.create_window(window_name="平面点采样结果", width=window_size[0], height=window_size[1])
            for geometry in plane_geometries:
                vis_plane.add_geometry(geometry)
            # 设置渲染选项
            vis_plane.get_render_option().point_size = 3.0
            vis_plane.get_render_option().background_color = np.asarray([1, 1, 1])
            vis_plane.run()
            vis_plane.destroy_window()
        
        # 边缘点可视化窗口
        if len(edge_points) > 0:
            edge_geometries = []
            # 只添加边缘点云
            edge_geometries.append(edge_pcd)
            
            # 创建可视化窗口
            vis_edge = o3d.visualization.Visualizer()
            vis_edge.create_window(window_name="边缘点采样结果", width=window_size[0], height=window_size[1])
            for geometry in edge_geometries:
                vis_edge.add_geometry(geometry)
            # 设置渲染选项
            vis_edge.get_render_option().point_size = 3.0
            vis_edge.get_render_option().background_color = np.asarray([1, 1, 1])
            vis_edge.run()
            vis_edge.destroy_window()
        
        # 角点可视化窗口
        if len(corner_points) > 0:
            corner_geometries = []
            # 只添加角点云
            corner_geometries.append(corner_pcd)
            
            # 创建可视化窗口
            vis_corner = o3d.visualization.Visualizer()
            vis_corner.create_window(window_name="角点采样结果", width=window_size[0], height=window_size[1])
            for geometry in corner_geometries:
                vis_corner.add_geometry(geometry)
            # 设置渲染选项
            vis_corner.get_render_option().point_size = 3.0
            vis_corner.get_render_option().background_color = np.asarray([1, 1, 1])
            vis_corner.run()
            vis_corner.destroy_window()
    
    # 保存采样后的点云
    if save_path:
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存整体采样点云
        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
        o3d.io.write_point_cloud(save_path, sampled_pcd)
        print(f"采样点云已保存到: {save_path}")
        
        # 分别保存不同类型的点云
        base_name = os.path.splitext(save_path)[0]
        
        if len(plane_points) > 0:
            plane_path = f"{base_name}_plane.ply"
            o3d.io.write_point_cloud(plane_path, plane_pcd)
            print(f"平面点云已保存到: {plane_path}")
        
        if len(edge_points) > 0:
            edge_path = f"{base_name}_edge.ply"
            o3d.io.write_point_cloud(edge_path, edge_pcd)
            print(f"边缘点云已保存到: {edge_path}")
        
        if len(corner_points) > 0:
            corner_path = f"{base_name}_corner.ply"
            o3d.io.write_point_cloud(corner_path, corner_pcd)
            print(f"角点云已保存到: {corner_path}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="基于几何分层的点云采样可视化工具")
    # parser.add_argument("--point_cloud_path", type=str, default=r'D:\Application\PycharmProfessional\pycharm\PointCloud_registration\GeoTransformer-main\data\custom\RawData_world\test\merged_points.txt',
    # help="点云文件路径 (支持PLY、PCD等格式)")
    parser.add_argument("--point_cloud_path", type=str, default=r"D:\Application\PycharmProfessional\pycharm\PointCloud_registration\PCR_test\data\MiC-9_uniform_50000.ply",
    help="点云文件路径 (支持PLY、PCD等格式)")
    parser.add_argument("--target_points", type=int, default=30000, help="目标采样点数")
    parser.add_argument("--corner_ratio", type=float, default=0.30, help="角点比例")
    parser.add_argument("--edge_ratio", type=float, default=0.30, help="边缘点比例")
    parser.add_argument("--plane_ratio", type=float, default=0.40, help="平面点比例")
    parser.add_argument("--save", type=str, default=None, help="保存采样点云的路径")
    parser.add_argument("--visualize", type=bool, default=True, help="是否开启可视化")
    parser.add_argument("--window_width", type=int, default=800, help="可视化窗口宽度")
    parser.add_argument("--window_height", type=int, default=600, help="可视化窗口高度")
    
    args = parser.parse_args()
    
    # 加载点云
    print(f"加载点云: {args.point_cloud_path}")
    pcd = read_point_cloud(args.point_cloud_path)
    
    if not pcd.has_points():
        print("错误: 点云为空")
        return
    
    print(f"原始点云点数: {len(pcd.points)}")
    
    # 初始化分层采样器
    sampler = HierarchicalStructurePreservingSampling(
        target_num_points=args.target_points,
        corner_ratio=args.corner_ratio,
        edge_ratio=args.edge_ratio,
        plane_ratio=args.plane_ratio
    )
    
    # 执行采样
    print("执行分层采样...")
    points = np.asarray(pcd.points)
    
    # 检查点云大小，限制最大点数以避免内存问题
    max_points = 50000
    if len(points) > max_points:
        print(f"点云过大 ({len(points)} 点)，进行均匀下采样...")
        # 对原始点云进行均匀随机采样
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        print(f"下采样后点数: {len(points)}")
    
    # 执行分层采样
    try:
        sampled_points, indices, point_types = sampler.sample(points)
        print(f"采样后点数: {len(sampled_points)}")
    except Exception as e:
        print(f"采样过程出错: {e}")
        return
    
    # 可视化结果
    visualize_hierarchical_sampling(
        pcd, 
        sampled_points, 
        point_types, 
        args.save,
        visualize=args.visualize,
        window_size=(args.window_width, args.window_height)
    )

if __name__ == "__main__":
    main()
