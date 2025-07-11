"""
角度计算模块
"""
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from utils.data_structures import AngleMetrics, TrackingResult
from config.model_configs import ANALYSIS_CONFIG

logger = logging.getLogger(__name__)

class AngleCalculator:
    """
    角度计算器
    """
    
    def __init__(self):
        """
        初始化角度计算器
        """
        self.config = ANALYSIS_CONFIG["angle_calculation"]
        self.joint_pairs = self.config["joint_pairs"]
        self.smoothing_window = self.config["smoothing_window"]
        
        # COCO 17个关键点索引映射
        self.keypoint_indices = {
            "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
            "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
            "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
            "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
        }
        
        logger.info("角度计算器初始化完成")
    
    async def calculate_angle_sequence(
        self, 
        person_sequence: List[Dict[str, Any]]
    ) -> List[AngleMetrics]:
        """
        计算人物序列的角度
        
        Args:
            person_sequence: 人物序列数据
            
        Returns:
            角度指标序列
        """
        try:
            all_angle_metrics = []
            
            for frame_data in person_sequence:
                frame_id = frame_data["frame_id"]
                tracking_data = frame_data["tracking_data"]
                
                # 提取姿态数据
                if hasattr(tracking_data, 'pose_data') and tracking_data.pose_data:
                    pose_data = tracking_data.pose_data
                    frame_angles = self._calculate_frame_angles(pose_data, frame_id)
                    all_angle_metrics.extend(frame_angles)
            
            # 平滑处理
            if len(all_angle_metrics) > self.smoothing_window * len(self.joint_pairs):
                all_angle_metrics = self._smooth_angles(all_angle_metrics)
            
            return all_angle_metrics
            
        except Exception as e:
            logger.error(f"角度计算失败: {str(e)}")
            return []
    
    def _calculate_frame_angles(
        self, 
        pose_data, 
        frame_id: int
    ) -> List[AngleMetrics]:
        """
        计算单帧的角度
        
        Args:
            pose_data: 姿态数据
            frame_id: 帧ID
            
        Returns:
            单帧角度指标列表
        """
        frame_angles = []
        
        if not hasattr(pose_data, 'keypoints') or not pose_data.keypoints:
            return frame_angles
        
        keypoints = pose_data.keypoints
        
        # 计算预定义的关节角度
        for joint_triplet in self.joint_pairs:
            if len(joint_triplet) == 3:
                joint1, joint2, joint3 = joint_triplet
                angle = self._calculate_joint_angle(keypoints, joint1, joint2, joint3)
                
                if angle is not None:
                    angle_metric = AngleMetrics(
                        joint_name=f"{joint1}_{joint2}_{joint3}",
                        angle=angle,
                        frame_id=frame_id
                    )
                    frame_angles.append(angle_metric)
        
        # 计算额外的重要角度
        additional_angles = self._calculate_additional_angles(keypoints, frame_id)
        frame_angles.extend(additional_angles)
        
        return frame_angles
    
    def _calculate_joint_angle(
        self, 
        keypoints: List, 
        joint1: str, 
        joint2: str, 
        joint3: str
    ) -> Optional[float]:
        """
        计算三个关键点形成的角度
        
        Args:
            keypoints: 关键点列表
            joint1: 第一个关节名
            joint2: 顶点关节名
            joint3: 第三个关节名
            
        Returns:
            角度值（度）或None
        """
        try:
            # 获取关键点索引
            idx1 = self.keypoint_indices.get(joint1)
            idx2 = self.keypoint_indices.get(joint2)
            idx3 = self.keypoint_indices.get(joint3)
            
            if idx1 is None or idx2 is None or idx3 is None:
                return None
            
            if (idx1 >= len(keypoints) or idx2 >= len(keypoints) or 
                idx3 >= len(keypoints)):
                return None
            
            # 获取关键点
            kpt1 = keypoints[idx1]
            kpt2 = keypoints[idx2]  # 顶点
            kpt3 = keypoints[idx3]
            
            # 检查置信度
            if (kpt1.confidence < 0.3 or kpt2.confidence < 0.3 or 
                kpt3.confidence < 0.3):
                return None
            
            # 计算角度
            angle = self._calculate_angle_between_points(
                (kpt1.x, kpt1.y),
                (kpt2.x, kpt2.y),
                (kpt3.x, kpt3.y)
            )
            
            return angle
            
        except Exception as e:
            logger.error(f"关节角度计算失败: {str(e)}")
            return None
    
    def _calculate_angle_between_points(
        self, 
        point1: tuple, 
        point2: tuple, 
        point3: tuple
    ) -> float:
        """
        计算三点之间的角度（point2为顶点）
        
        Args:
            point1: 第一个点
            point2: 顶点
            point3: 第三个点
            
        Returns:
            角度（度）
        """
        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = point3
        
        # 计算向量
        vec1 = np.array([x1 - x2, y1 - y2])
        vec2 = np.array([x3 - x2, y3 - y2])
        
        # 计算向量长度
        len1 = np.linalg.norm(vec1)
        len2 = np.linalg.norm(vec2)
        
        # 避免除零
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # 计算夹角
        cos_angle = np.dot(vec1, vec2) / (len1 * len2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 确保在有效范围内
        
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def _calculate_additional_angles(
        self, 
        keypoints: List, 
        frame_id: int
    ) -> List[AngleMetrics]:
        """
        计算额外的重要角度
        
        Args:
            keypoints: 关键点列表
            frame_id: 帧ID
            
        Returns:
            额外角度指标列表
        """
        additional_angles = []
        
        # 身体倾斜角度（躯干与垂直方向的夹角）
        trunk_angle = self._calculate_trunk_angle(keypoints)
        if trunk_angle is not None:
            additional_angles.append(AngleMetrics(
                joint_name="trunk_tilt",
                angle=trunk_angle,
                frame_id=frame_id
            ))
        
        # 头部角度
        head_angle = self._calculate_head_angle(keypoints)
        if head_angle is not None:
            additional_angles.append(AngleMetrics(
                joint_name="head_tilt",
                angle=head_angle,
                frame_id=frame_id
            ))
        
        # 脊柱弯曲角度
        spine_curve = self._calculate_spine_curvature(keypoints)
        if spine_curve is not None:
            additional_angles.append(AngleMetrics(
                joint_name="spine_curvature",
                angle=spine_curve,
                frame_id=frame_id
            ))
        
        return additional_angles
    
    def _calculate_trunk_angle(self, keypoints: List) -> Optional[float]:
        """
        计算躯干倾斜角度
        
        Args:
            keypoints: 关键点列表
            
        Returns:
            躯干角度或None
        """
        try:
            # 使用肩膀中点和髋部中点计算躯干方向
            left_shoulder_idx = self.keypoint_indices["left_shoulder"]
            right_shoulder_idx = self.keypoint_indices["right_shoulder"]
            left_hip_idx = self.keypoint_indices["left_hip"]
            right_hip_idx = self.keypoint_indices["right_hip"]
            
            if (left_shoulder_idx >= len(keypoints) or right_shoulder_idx >= len(keypoints) or
                left_hip_idx >= len(keypoints) or right_hip_idx >= len(keypoints)):
                return None
            
            left_shoulder = keypoints[left_shoulder_idx]
            right_shoulder = keypoints[right_shoulder_idx]
            left_hip = keypoints[left_hip_idx]
            right_hip = keypoints[right_hip_idx]
            
            # 检查置信度
            if (left_shoulder.confidence < 0.3 or right_shoulder.confidence < 0.3 or
                left_hip.confidence < 0.3 or right_hip.confidence < 0.3):
                return None
            
            # 计算肩膀和髋部中点
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_center_x = (left_hip.x + right_hip.x) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            
            # 计算躯干向量与垂直方向的夹角
            trunk_vector = np.array([hip_center_x - shoulder_center_x, 
                                   hip_center_y - shoulder_center_y])
            vertical_vector = np.array([0, 1])  # 垂直向下
            
            # 计算角度
            cos_angle = np.dot(trunk_vector, vertical_vector) / (
                np.linalg.norm(trunk_vector) * np.linalg.norm(vertical_vector)
            )
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            return angle_deg
            
        except Exception as e:
            logger.error(f"躯干角度计算失败: {str(e)}")
            return None
    
    def _calculate_head_angle(self, keypoints: List) -> Optional[float]:
        """
        计算头部角度
        
        Args:
            keypoints: 关键点列表
            
        Returns:
            头部角度或None
        """
        try:
            nose_idx = self.keypoint_indices["nose"]
            left_shoulder_idx = self.keypoint_indices["left_shoulder"]
            right_shoulder_idx = self.keypoint_indices["right_shoulder"]
            
            if (nose_idx >= len(keypoints) or left_shoulder_idx >= len(keypoints) or
                right_shoulder_idx >= len(keypoints)):
                return None
            
            nose = keypoints[nose_idx]
            left_shoulder = keypoints[left_shoulder_idx]
            right_shoulder = keypoints[right_shoulder_idx]
            
            if (nose.confidence < 0.3 or left_shoulder.confidence < 0.3 or
                right_shoulder.confidence < 0.3):
                return None
            
            # 计算肩膀中点
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # 计算头部向量与垂直方向的夹角
            head_vector = np.array([nose.x - shoulder_center_x, 
                                  nose.y - shoulder_center_y])
            vertical_vector = np.array([0, -1])  # 垂直向上
            
            # 计算角度
            if np.linalg.norm(head_vector) == 0:
                return None
            
            cos_angle = np.dot(head_vector, vertical_vector) / (
                np.linalg.norm(head_vector) * np.linalg.norm(vertical_vector)
            )
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            return angle_deg
            
        except Exception as e:
            logger.error(f"头部角度计算失败: {str(e)}")
            return None
    
    def _calculate_spine_curvature(self, keypoints: List) -> Optional[float]:
        """
        计算脊柱弯曲度
        
        Args:
            keypoints: 关键点列表
            
        Returns:
            脊柱弯曲度或None
        """
        try:
            # 使用鼻子、肩膀中点、髋部中点来计算脊柱弯曲
            nose_idx = self.keypoint_indices["nose"]
            left_shoulder_idx = self.keypoint_indices["left_shoulder"]
            right_shoulder_idx = self.keypoint_indices["right_shoulder"]
            left_hip_idx = self.keypoint_indices["left_hip"]
            right_hip_idx = self.keypoint_indices["right_hip"]
            
            if (nose_idx >= len(keypoints) or left_shoulder_idx >= len(keypoints) or
                right_shoulder_idx >= len(keypoints) or left_hip_idx >= len(keypoints) or
                right_hip_idx >= len(keypoints)):
                return None
            
            nose = keypoints[nose_idx]
            left_shoulder = keypoints[left_shoulder_idx]
            right_shoulder = keypoints[right_shoulder_idx]
            left_hip = keypoints[left_hip_idx]
            right_hip = keypoints[right_hip_idx]
            
            # 检查置信度
            if (nose.confidence < 0.3 or left_shoulder.confidence < 0.3 or
                right_shoulder.confidence < 0.3 or left_hip.confidence < 0.3 or
                right_hip.confidence < 0.3):
                return None
            
            # 计算关键点
            shoulder_center = ((left_shoulder.x + right_shoulder.x) / 2,
                             (left_shoulder.y + right_shoulder.y) / 2)
            hip_center = ((left_hip.x + right_hip.x) / 2,
                         (left_hip.y + right_hip.y) / 2)
            
            # 计算脊柱弯曲角度（三点角度）
            curvature_angle = self._calculate_angle_between_points(
                (nose.x, nose.y),
                shoulder_center,
                hip_center
            )
            
            # 将角度转换为弯曲度（180度为直线，偏差越大弯曲越明显）
            curvature = abs(180.0 - curvature_angle)
            
            return curvature
            
        except Exception as e:
            logger.error(f"脊柱弯曲度计算失败: {str(e)}")
            return None
    
    def _smooth_angles(
        self, 
        angle_metrics: List[AngleMetrics]
    ) -> List[AngleMetrics]:
        """
        平滑角度序列
        
        Args:
            angle_metrics: 角度指标列表
            
        Returns:
            平滑后的角度指标列表
        """
        # 按关节名称分组
        angle_groups = {}
        for metric in angle_metrics:
            joint_name = metric.joint_name
            if joint_name not in angle_groups:
                angle_groups[joint_name] = []
            angle_groups[joint_name].append(metric)
        
        # 对每组分别平滑
        smoothed_metrics = []
        for joint_name, group in angle_groups.items():
            # 按帧ID排序
            group.sort(key=lambda x: x.frame_id)
            
            # 提取角度值
            angles = [metric.angle for metric in group]
            
            # 平滑处理
            smoothed_angles = self._apply_moving_average(angles)
            
            # 创建新的指标对象
            for i, metric in enumerate(group):
                if i < len(smoothed_angles):
                    smoothed_metric = AngleMetrics(
                        joint_name=metric.joint_name,
                        angle=smoothed_angles[i],
                        frame_id=metric.frame_id
                    )
                    smoothed_metrics.append(smoothed_metric)
        
        return smoothed_metrics
    
    def _apply_moving_average(self, values: List[float]) -> List[float]:
        """
        应用移动平均平滑
        
        Args:
            values: 原始值列表
            
        Returns:
            平滑后的值列表
        """
        if len(values) < self.smoothing_window:
            return values
        
        smoothed = []
        half_window = self.smoothing_window // 2
        
        for i in range(len(values)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(values), i + half_window + 1)
            
            window_values = values[start_idx:end_idx]
            avg_value = np.mean(window_values)
            smoothed.append(avg_value)
        
        return smoothed 