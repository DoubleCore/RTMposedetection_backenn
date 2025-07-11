"""
速度计算模块
"""
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from utils.data_structures import SpeedMetrics, TrackingResult
from config.model_configs import ANALYSIS_CONFIG

logger = logging.getLogger(__name__)

class SpeedCalculator:
    """
    速度计算器
    """
    
    def __init__(self):
        """
        初始化速度计算器
        """
        self.config = ANALYSIS_CONFIG["speed_calculation"]
        self.fps = self.config["fps"]
        self.smoothing_window = self.config["smoothing_window"]
        self.units = self.config["units"]
        
        logger.info("速度计算器初始化完成")
    
    async def calculate_speed_sequence(
        self, 
        person_sequence: List[Dict[str, Any]]
    ) -> List[Optional[SpeedMetrics]]:
        """
        计算人物序列的速度
        
        Args:
            person_sequence: 人物序列数据
            
        Returns:
            速度指标序列
        """
        try:
            if len(person_sequence) < 2:
                logger.warning("序列长度不足，无法计算速度")
                return [None] * len(person_sequence)
            
            # 提取位置序列
            positions = self._extract_positions(person_sequence)
            
            # 计算速度
            velocities = self._calculate_velocities(positions, person_sequence)
            
            # 计算加速度
            accelerations = self._calculate_accelerations(velocities)
            
            # 平滑处理
            if len(velocities) > self.smoothing_window:
                velocities = self._smooth_data(velocities)
                accelerations = self._smooth_data(accelerations)
            
            # 创建速度指标
            speed_metrics = []
            for i, frame_data in enumerate(person_sequence):
                if i < len(velocities) and velocities[i] is not None:
                    velocity_x, velocity_y = velocities[i]
                    speed_magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
                    acceleration = accelerations[i] if i < len(accelerations) else 0.0
                    
                    metric = SpeedMetrics(
                        velocity_x=velocity_x,
                        velocity_y=velocity_y,
                        speed_magnitude=speed_magnitude,
                        acceleration=acceleration,
                        frame_id=frame_data["frame_id"]
                    )
                    speed_metrics.append(metric)
                else:
                    speed_metrics.append(None)
            
            return speed_metrics
            
        except Exception as e:
            logger.error(f"速度计算失败: {str(e)}")
            return [None] * len(person_sequence)
    
    def _extract_positions(
        self, 
        person_sequence: List[Dict[str, Any]]
    ) -> List[Optional[tuple]]:
        """
        提取位置序列
        
        Args:
            person_sequence: 人物序列
            
        Returns:
            位置序列 [(x, y), ...]
        """
        positions = []
        
        for frame_data in person_sequence:
            tracking_data = frame_data["tracking_data"]
            
            if hasattr(tracking_data, 'bbox') and tracking_data.bbox:
                # 使用边界框中心作为位置
                bbox = tracking_data.bbox
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                positions.append((center_x, center_y))
            elif hasattr(tracking_data, 'pose_data') and tracking_data.pose_data:
                # 使用姿态重心作为位置
                center = self._calculate_pose_center(tracking_data.pose_data)
                if center:
                    positions.append(center)
                else:
                    positions.append(None)
            else:
                positions.append(None)
        
        return positions
    
    def _calculate_pose_center(self, pose_data) -> Optional[tuple]:
        """
        计算姿态重心
        
        Args:
            pose_data: 姿态数据
            
        Returns:
            重心坐标或None
        """
        if not hasattr(pose_data, 'keypoints') or not pose_data.keypoints:
            return None
        
        valid_points = [kpt for kpt in pose_data.keypoints if kpt.confidence > 0.3]
        
        if not valid_points:
            return None
        
        center_x = sum(kpt.x for kpt in valid_points) / len(valid_points)
        center_y = sum(kpt.y for kpt in valid_points) / len(valid_points)
        
        return (center_x, center_y)
    
    def _calculate_velocities(
        self, 
        positions: List[Optional[tuple]], 
        person_sequence: List[Dict[str, Any]]
    ) -> List[Optional[tuple]]:
        """
        计算速度序列
        
        Args:
            positions: 位置序列
            person_sequence: 人物序列（用于获取时间信息）
            
        Returns:
            速度序列 [(vx, vy), ...]
        """
        velocities = [None]  # 第一帧没有速度
        
        for i in range(1, len(positions)):
            if positions[i] is None or positions[i-1] is None:
                velocities.append(None)
                continue
            
            # 计算位置差
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            
            # 计算时间差
            dt = self._calculate_time_delta(person_sequence, i)
            
            if dt > 0:
                # 像素/秒转换为实际单位（这里需要根据实际情况调整转换因子）
                # 假设1像素 = 0.01米（需要根据实际场景标定）
                pixel_to_meter = 0.01
                
                velocity_x = (dx * pixel_to_meter) / dt
                velocity_y = (dy * pixel_to_meter) / dt
                
                velocities.append((velocity_x, velocity_y))
            else:
                velocities.append(None)
        
        return velocities
    
    def _calculate_time_delta(
        self, 
        person_sequence: List[Dict[str, Any]], 
        index: int
    ) -> float:
        """
        计算时间差
        
        Args:
            person_sequence: 人物序列
            index: 当前帧索引
            
        Returns:
            时间差（秒）
        """
        if index == 0:
            return 0.0
        
        # 如果有时间戳，使用时间戳
        if "timestamp" in person_sequence[index] and "timestamp" in person_sequence[index-1]:
            return person_sequence[index]["timestamp"] - person_sequence[index-1]["timestamp"]
        
        # 否则基于帧率计算
        return 1.0 / self.fps
    
    def _calculate_accelerations(
        self, 
        velocities: List[Optional[tuple]]
    ) -> List[float]:
        """
        计算加速度序列
        
        Args:
            velocities: 速度序列
            
        Returns:
            加速度序列（标量）
        """
        accelerations = [0.0]  # 第一帧加速度为0
        
        for i in range(1, len(velocities)):
            if velocities[i] is None or velocities[i-1] is None:
                accelerations.append(0.0)
                continue
            
            # 计算速度变化
            v1_mag = np.sqrt(velocities[i-1][0]**2 + velocities[i-1][1]**2)
            v2_mag = np.sqrt(velocities[i][0]**2 + velocities[i][1]**2)
            
            # 加速度 = 速度变化 / 时间变化
            dv = v2_mag - v1_mag
            dt = 1.0 / self.fps  # 假设固定帧率
            
            acceleration = dv / dt if dt > 0 else 0.0
            accelerations.append(acceleration)
        
        return accelerations
    
    def _smooth_data(self, data: List[Optional[tuple]]) -> List[Optional[tuple]]:
        """
        平滑数据
        
        Args:
            data: 原始数据
            
        Returns:
            平滑后的数据
        """
        if len(data) < self.smoothing_window:
            return data
        
        smoothed = []
        half_window = self.smoothing_window // 2
        
        for i in range(len(data)):
            # 计算窗口范围
            start_idx = max(0, i - half_window)
            end_idx = min(len(data), i + half_window + 1)
            
            # 收集有效数据点
            window_data = [d for d in data[start_idx:end_idx] if d is not None]
            
            if window_data:
                if isinstance(window_data[0], tuple):
                    # 元组数据（速度）
                    avg_x = np.mean([d[0] for d in window_data])
                    avg_y = np.mean([d[1] for d in window_data])
                    smoothed.append((avg_x, avg_y))
                else:
                    # 标量数据（加速度）
                    avg_val = np.mean(window_data)
                    smoothed.append(avg_val)
            else:
                smoothed.append(data[i])
        
        return smoothed
    
    def calculate_movement_metrics(
        self, 
        speed_sequence: List[Optional[SpeedMetrics]]
    ) -> Dict[str, Any]:
        """
        计算运动指标
        
        Args:
            speed_sequence: 速度序列
            
        Returns:
            运动指标
        """
        valid_metrics = [m for m in speed_sequence if m is not None]
        
        if not valid_metrics:
            return {}
        
        speeds = [m.speed_magnitude for m in valid_metrics]
        accelerations = [m.acceleration for m in valid_metrics]
        
        metrics = {
            "max_speed": max(speeds),
            "min_speed": min(speeds),
            "avg_speed": np.mean(speeds),
            "speed_std": np.std(speeds),
            "max_acceleration": max(accelerations),
            "min_acceleration": min(accelerations),
            "avg_acceleration": np.mean(accelerations),
            "total_distance": self._calculate_total_distance(valid_metrics),
            "movement_smoothness": self._calculate_smoothness(speeds)
        }
        
        return metrics
    
    def _calculate_total_distance(
        self, 
        speed_metrics: List[SpeedMetrics]
    ) -> float:
        """
        计算总移动距离
        
        Args:
            speed_metrics: 速度指标列表
            
        Returns:
            总距离
        """
        total_distance = 0.0
        dt = 1.0 / self.fps
        
        for metric in speed_metrics:
            distance = metric.speed_magnitude * dt
            total_distance += distance
        
        return total_distance
    
    def _calculate_smoothness(self, speeds: List[float]) -> float:
        """
        计算运动平滑度
        
        Args:
            speeds: 速度列表
            
        Returns:
            平滑度评分（0-1，越高越平滑）
        """
        if len(speeds) < 3:
            return 1.0
        
        # 计算速度变化的标准差
        speed_changes = []
        for i in range(1, len(speeds)):
            change = abs(speeds[i] - speeds[i-1])
            speed_changes.append(change)
        
        if not speed_changes:
            return 1.0
        
        # 标准化平滑度评分
        change_std = np.std(speed_changes)
        avg_speed = np.mean(speeds)
        
        # 避免除零
        if avg_speed == 0:
            return 1.0
        
        # 平滑度评分（变化越小越平滑）
        smoothness = 1.0 / (1.0 + change_std / avg_speed)
        
        return min(1.0, max(0.0, smoothness)) 