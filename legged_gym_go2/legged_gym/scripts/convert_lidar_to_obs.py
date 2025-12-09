#!/usr/bin/env python3
"""
激光雷达数据转换工具
将真实激光雷达传感器传回的数据转换为强化学习算法所需的观测值(obs)格式
"""
import torch
import numpy as np
import math
import argparse

class LidarToObsConverter:
    """
    激光雷达数据到观测值的转换器
    按照HighLevelNavigationEnv中的处理方式进行转换
    """
    def __init__(self, config=None):
        """
        初始化转换器
        
        Args:
            config: 配置对象，包含雷达参数设置
        """
        # 默认配置
        self.config = {
            'lidar_max_range': 10.0,  # 雷达最大范围
            'lidar_num_bins': 16,      # 雷达分箱数量
            'unsafe_sphere_radius': 0.3,  # 障碍物半径
            'target_radius': 0.4,      # 目标半径
            'target_lidar_max_range': 10.0,  # 目标雷达最大范围
            'target_lidar_num_bins': 16,    # 目标雷达分箱数量
            'boundary_half_extents': (3.0, 3.0),  # 边界半长
        }
        
        # 更新配置
        if config:
            self.config.update(config)
            
        # 计算雷达方向向量（机体坐标系）
        bin_size = 2 * math.pi / float(self.config['lidar_num_bins'])
        angles = np.arange(self.config['lidar_num_bins'], dtype=np.float32) * bin_size - math.pi + 0.5 * bin_size
        self._lidar_dir_body = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        
        # 观测维度计算
        self._base_obs_dim = 8  # 基础特征维度
        self.lidar_feature_dim = self.config['lidar_num_bins']
        self.target_feature_dim = self.config['target_lidar_num_bins']
        self.num_high_level_obs = self._base_obs_dim + self.target_feature_dim + self.lidar_feature_dim
        
        print(f"雷达转换器初始化完成:")
        print(f"  雷达分箱数: {self.config['lidar_num_bins']}")
        print(f"  雷达最大范围: {self.config['lidar_max_range']}m")
        print(f"  目标雷达分箱数: {self.config['target_lidar_num_bins']}")
        print(f"  输出观测维度: {self.num_high_level_obs}")
    
    def world_to_body_coords(self, points_world, robot_pos, heading_cos, heading_sin):
        """
        将世界坐标系下的点转换为机器人坐标系
        
        Args:
            points_world: 世界坐标系下的点 [num_points, 2]
            robot_pos: 机器人位置 [2]
            heading_cos: 机器人朝向的cos值
            heading_sin: 机器人朝向的sin值
        
        Returns:
            points_body: 机器人坐标系下的点 [num_points, 2]
        """
        # 平移到机器人中心
        points_centered = points_world - robot_pos
        
        # 旋转到机器人坐标系
        # [x', y'] = [x*cosθ + y*sinθ, -x*sinθ + y*cosθ]
        points_body_x = points_centered[:, 0] * heading_cos + points_centered[:, 1] * heading_sin
        points_body_y = -points_centered[:, 0] * heading_sin + points_centered[:, 1] * heading_cos
        
        return np.stack((points_body_x, points_body_y), axis=-1)
    
    def convert_obstacle_lidar(self, obstacle_points, robot_pos, heading_cos, heading_sin):
        """
        转换障碍物点云到雷达特征
        
        Args:
            obstacle_points: 障碍物点云 [num_points, 2]
            robot_pos: 机器人位置 [2]
            heading_cos: 机器人朝向的cos值
            heading_sin: 机器人朝向的sin值
            
        Returns:
            lidar_features: 雷达特征向量 [lidar_num_bins]
        """
        if len(obstacle_points) == 0:
            return np.zeros(self.config['lidar_num_bins'], dtype=np.float32)
        
        # 转换到机器人坐标系
        points_body = self.world_to_body_coords(obstacle_points, robot_pos, heading_cos, heading_sin)
        
        # 计算距离
        planar_dist = np.sqrt(points_body[:, 0]**2 + points_body[:, 1]** 2 + 1e-9)
        
        # 计算表面距离（减去障碍物半径）
        surface_dist = np.maximum(planar_dist - self.config['unsafe_sphere_radius'], 0.0)
        
        # 归一化距离
        normalized_dist = np.clip(surface_dist / self.config['lidar_max_range'], 0.0, 1.0)
        
        # 计算强度（距离越近，强度越大）
        intensity = 1.0 - normalized_dist
        
        # 计算角度并分配到分箱
        bin_size = 2 * math.pi / float(self.config['lidar_num_bins'])
        angles = np.arctan2(points_body[:, 1], points_body[:, 0])
        bin_indices = np.floor((angles + math.pi) / bin_size).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, self.config['lidar_num_bins'] - 1)
        
        # 创建雷达特征向量
        lidar_features = np.zeros(self.config['lidar_num_bins'], dtype=np.float32)
        for i, bin_idx in enumerate(bin_indices):
            # 在每个分箱中保留最大强度
            if intensity[i] > lidar_features[bin_idx]:
                lidar_features[bin_idx] = intensity[i]
        
        return lidar_features
    
    def convert_target_lidar(self, target_pos, robot_pos, heading_cos, heading_sin):
        """
        转换目标位置到目标雷达特征
        
        Args:
            target_pos: 目标位置 [2]
            robot_pos: 机器人位置 [2]
            heading_cos: 机器人朝向的cos值
            heading_sin: 机器人朝向的sin值
            
        Returns:
            target_features: 目标雷达特征向量 [target_lidar_num_bins]
        """
        # 计算目标相对位置
        target_rel = target_pos - robot_pos
        
        # 计算距离
        target_distance = np.linalg.norm(target_rel)
        
        # 计算表面距离（减去目标半径）
        target_surface_dist = np.maximum(target_distance - self.config['target_radius'], 0.0)
        
        # 计算强度
        target_intensity = 1.0 - np.clip(target_surface_dist / self.config['target_lidar_max_range'], 0.0, 1.0)
        
        # 转换到机器人坐标系
        target_rel_body = np.array([
            target_rel[0] * heading_cos + target_rel[1] * heading_sin,
            -target_rel[0] * heading_sin + target_rel[1] * heading_cos
        ])
        
        # 计算角度
        target_angle = np.arctan2(target_rel_body[1], target_rel_body[0])
        
        # 平滑编码到分箱中
        bin_size = 2 * math.pi / float(self.config['target_lidar_num_bins'])
        normalized_angle = (target_angle + math.pi) / (2 * math.pi)
        scaled_bins = normalized_angle * self.config['target_lidar_num_bins']
        
        floored = np.floor(scaled_bins)
        frac = np.clip(scaled_bins - floored, 0.0, 1.0)
        
        lower_bin = int(np.remainder(floored, self.config['target_lidar_num_bins']))
        upper_bin = (lower_bin + 1) % self.config['target_lidar_num_bins']
        
        lower_weight = (1.0 - frac) * target_intensity
        upper_weight = frac * target_intensity
        
        # 创建目标特征向量
        target_features = np.zeros(self.config['target_lidar_num_bins'], dtype=np.float32)
        target_features[lower_bin] = lower_weight
        target_features[upper_bin] = upper_weight
        
        return target_features
    
    def compute_base_observations(self, robot_state):
        """
        计算基础观测值
        
        Args:
            robot_state: 包含机器人状态的字典，应包含:
                - heading_cos: 朝向的cos值
                - heading_sin: 朝向的sin值
                - body_vel_x: 身体坐标系x方向速度
                - body_vel_y: 身体坐标系y方向速度
                - yaw_rate: 偏航角速度
                - target_distance: 到目标的距离
                - target_dir_x: 机器人坐标系中目标方向x分量
                - target_dir_y: 机器人坐标系中目标方向y分量
        
        Returns:
            base_obs: 基础观测值 [_base_obs_dim]
        """
        base_obs = np.zeros(self._base_obs_dim, dtype=np.float32)
        
        # 填充基础观测值
        base_obs[0] = robot_state['heading_cos']        # cos(heading)
        base_obs[1] = robot_state['heading_sin']        # sin(heading)
        
        # 线速度（缩放和裁剪）
        lin_vel_scale = 2.0
        base_obs[2] = np.clip(robot_state['body_vel_x'] * lin_vel_scale, -1.0, 1.0)
        base_obs[3] = np.clip(robot_state['body_vel_y'] * lin_vel_scale, -1.0, 1.0)
        
        # 角速度（缩放和裁剪）
        ang_vel_scale = 0.25
        base_obs[4] = np.clip(robot_state['yaw_rate'] * ang_vel_scale, -1.0, 1.0)
        
        # 目标距离（归一化）
        lidar_max_range = self.config['lidar_max_range']
        base_obs[5] = np.clip(robot_state['target_distance'] / lidar_max_range, 0.0, 1.0)
        
        # 目标方向
        base_obs[6] = robot_state['target_dir_x']
        base_obs[7] = robot_state['target_dir_y']
        
        return base_obs
    
    def convert_to_observation(self, robot_state, obstacle_points, target_pos=None):
        """
        将雷达数据和机器人状态转换为完整观测值
        
        Args:
            robot_state: 机器人状态字典
            obstacle_points: 障碍物点云 [num_points, 2]
            target_pos: 目标位置 [2]
        
        Returns:
            observation: 完整观测向量 [num_high_level_obs]
        """
        # 创建观测向量
        obs = np.zeros(self.num_high_level_obs, dtype=np.float32)
        
        # 计算基础观测
        base_obs = self.compute_base_observations(robot_state)
        obs[:self._base_obs_dim] = base_obs
        
        # 计算目标雷达特征
        if target_pos is not None:
            target_features = self.convert_target_lidar(
                target_pos, 
                robot_state['position'], 
                robot_state['heading_cos'], 
                robot_state['heading_sin']
            )
            obs[self._base_obs_dim:self._base_obs_dim + self.target_feature_dim] = target_features
        
        # 计算障碍物雷达特征
        lidar_features = self.convert_obstacle_lidar(
            obstacle_points, 
            robot_state['position'], 
            robot_state['heading_cos'], 
            robot_state['heading_sin']
        )
        lidar_start_idx = self._base_obs_dim + self.target_feature_dim
        obs[lidar_start_idx:lidar_start_idx + self.lidar_feature_dim] = lidar_features
        
        return obs
    
    def create_example_robot_state(self):
        """
        创建示例机器人状态用于演示
        
        Returns:
            robot_state: 示例机器人状态字典
        """
        return {
            'position': np.array([0.0, 0.0], dtype=np.float32),  # 机器人位置
            'heading_cos': 1.0,     # 朝向的cos值（面向x轴）
            'heading_sin': 0.0,     # 朝向的sin值
            'body_vel_x': 0.1,      # 身体坐标系x方向速度
            'body_vel_y': 0.0,      # 身体坐标系y方向速度
            'yaw_rate': 0.0,        # 偏航角速度
            'target_distance': 4.0, # 到目标的距离
            'target_dir_x': 1.0,    # 目标方向x分量
            'target_dir_y': 0.0     # 目标方向y分量
        }
    
    def create_example_lidar_data(self):
        """
        创建示例激光雷达数据用于演示
        
        Returns:
            obstacle_points: 示例障碍物点云
            target_pos: 示例目标位置
        """
        # 创建几个示例障碍物点
        obstacle_points = np.array([
            [2.0, 0.5],   # 右侧障碍物
            [2.0, -0.5],  # 左侧障碍物
            [3.0, 1.0]    # 右前方障碍物
        ], dtype=np.float32)
        
        # 示例目标位置
        target_pos = np.array([4.0, 0.0], dtype=np.float32)
        
        return obstacle_points, target_pos
    
    def demonstrate_conversion(self):
        """
        演示数据转换过程
        """
        print("\n执行转换演示...")
        
        # 创建示例数据
        robot_state = self.create_example_robot_state()
        obstacle_points, target_pos = self.create_example_lidar_data()
        
        print("示例机器人状态:")
        for key, value in robot_state.items():
            print(f"  {key}: {value}")
        
        print(f"\n示例障碍物点 ({len(obstacle_points)}个点):")
        for i, point in enumerate(obstacle_points):
            print(f"  点{i+1}: {point}")
        
        print(f"\n示例目标位置: {target_pos}")
        
        # 执行转换
        obs = self.convert_to_observation(robot_state, obstacle_points, target_pos)
        
        print(f"\n转换后的观测值 (维度: {len(obs)}):")
        print(f"基础观测值 (前8维): {obs[:8]}")
        print(f"目标雷达特征 ({self.target_feature_dim}维): {obs[8:8+self.target_feature_dim]}")
        print(f"障碍物雷达特征 ({self.lidar_feature_dim}维): {obs[8+self.target_feature_dim:]}")
        
        return obs

def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description='激光雷达数据到观测值转换工具')
    parser.add_argument('--demo', action='store_true', help='运行演示')
    parser.add_argument('--lidar-bins', type=int, default=16, help='雷达分箱数量')
    parser.add_argument('--max-range', type=float, default=10.0, help='雷达最大范围')
    args = parser.parse_args()
    
    # 创建配置
    config = {
        'lidar_num_bins': args.lidar_bins,
        'lidar_max_range': args.max_range
    }
    
    # 初始化转换器
    converter = LidarToObsConverter(config)
    
    # 运行演示
    if args.demo:
        converter.demonstrate_conversion()
    else:
        print("\n提示: 使用 --demo 参数运行演示")
        print("\n如何在代码中使用:")
        print('''
        # 1. 初始化转换器
        converter = LidarToObsConverter()
        
        # 2. 准备数据
        robot_state = {
            'position': np.array([x, y]),
            'heading_cos': heading_cos,
            'heading_sin': heading_sin,
            'body_vel_x': body_vel_x,
            'body_vel_y': body_vel_y,
            'yaw_rate': yaw_rate,
            'target_distance': target_dist,
            'target_dir_x': target_dir_x,
            'target_dir_y': target_dir_y
        }
        obstacle_points = np.array([[x1, y1], [x2, y2], ...])  # 激光雷达点云
        target_pos = np.array([tx, ty])  # 目标位置
        
        # 3. 转换为观测值
        obs = converter.convert_to_observation(robot_state, obstacle_points, target_pos)
        
        # 4. 转换为PyTorch张量（如果需要）
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # 添加批次维度
        ''')

if __name__ == '__main__':
    main()
