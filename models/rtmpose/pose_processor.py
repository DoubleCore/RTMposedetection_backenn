"""
RTMPose姿态数据处理器
"""
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from utils.data_structures import PoseDetectionResult, Keypoint
from config.model_configs import RTMPOSE_CONFIG

logger = logging.getLogger(__name__)

class PoseProcessor:
    """
    姿态数据处理器
    """
    
    def __init__(self):
        """
        初始化姿态处理器
        """
        self.config = RTMPOSE_CONFIG
        self.keypoint_names = self._get_keypoint_names()
        
    def _get_keypoint_names(self) -> List[str]:
        """
        获取关键点名称列表（COCO 17个关键点）
        """
        return [
            "nose",           # 0
            "left_eye",       # 1
            "right_eye",      # 2
            "left_ear",       # 3
            "right_ear",      # 4
            "left_shoulder",  # 5
            "right_shoulder", # 6
            "left_elbow",     # 7
            "right_elbow",    # 8
            "left_wrist",     # 9
            "right_wrist",    # 10
            "left_hip",       # 11
            "right_hip",      # 12
            "left_knee",      # 13
            "right_knee",     # 14
            "left_ankle",     # 15
            "right_ankle"     # 16
        ]
    
    def filter_low_confidence_poses(
        self, 
        pose_results: List[PoseDetectionResult],
        min_confidence: float = None
    ) -> List[PoseDetectionResult]:
        """
        过滤低置信度的姿态
        
        Args:
            pose_results: 姿态检测结果列表
            min_confidence: 最小置信度阈值
            
        Returns:
            过滤后的姿态结果
        """
        if min_confidence is None:
            min_confidence = self.config["confidence_threshold"]
        
        filtered_results = []
        
        for pose_result in pose_results:
            if pose_result.confidence >= min_confidence:
                filtered_results.append(pose_result)
        
        logger.info(f"姿态过滤: {len(pose_results)} -> {len(filtered_results)}")
        return filtered_results
    
    def smooth_keypoints(
        self,
        pose_sequence: List[List[PoseDetectionResult]],
        window_size: int = 5
    ) -> List[List[PoseDetectionResult]]:
        """
        对关键点序列进行平滑处理
        
        Args:
            pose_sequence: 姿态序列（每帧的姿态列表）
            window_size: 平滑窗口大小
            
        Returns:
            平滑后的姿态序列
        """
        if len(pose_sequence) < window_size:
            return pose_sequence
        
        smoothed_sequence = []
        
        for frame_idx in range(len(pose_sequence)):
            frame_poses = pose_sequence[frame_idx]
            smoothed_poses = []
            
            for pose in frame_poses:
                smoothed_pose = self._smooth_single_pose(
                    pose, pose_sequence, frame_idx, window_size
                )
                smoothed_poses.append(smoothed_pose)
            
            smoothed_sequence.append(smoothed_poses)
        
        return smoothed_sequence
    
    def _smooth_single_pose(
        self,
        current_pose: PoseDetectionResult,
        pose_sequence: List[List[PoseDetectionResult]],
        frame_idx: int,
        window_size: int
    ) -> PoseDetectionResult:
        """
        平滑单个姿态的关键点
        """
        # 简化实现：仅对当前姿态进行处理
        # 实际应用中需要找到对应的人物轨迹进行时序平滑
        return current_pose
    
    def calculate_bone_lengths(self, pose_result: PoseDetectionResult) -> Dict[str, float]:
        """
        计算骨骼长度
        
        Args:
            pose_result: 姿态检测结果
            
        Returns:
            骨骼长度字典
        """
        bone_connections = [
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
            ("left_shoulder", "right_shoulder"),
            ("left_hip", "right_hip")
        ]
        
        bone_lengths = {}
        keypoints = pose_result.keypoints
        
        for bone_name, (start_joint, end_joint) in enumerate(bone_connections):
            try:
                start_idx = self.keypoint_names.index(start_joint)
                end_idx = self.keypoint_names.index(end_joint)
                
                start_kpt = keypoints[start_idx]
                end_kpt = keypoints[end_idx]
                
                if start_kpt.confidence > 0 and end_kpt.confidence > 0:
                    length = np.sqrt(
                        (end_kpt.x - start_kpt.x) ** 2 + 
                        (end_kpt.y - start_kpt.y) ** 2
                    )
                    bone_name = f"{start_joint}-{end_joint}"
                    bone_lengths[bone_name] = length
                    
            except (ValueError, IndexError):
                continue
        
        return bone_lengths
    
    def calculate_joint_angles(self, pose_result: PoseDetectionResult) -> Dict[str, float]:
        """
        计算关节角度
        
        Args:
            pose_result: 姿态检测结果
            
        Returns:
            关节角度字典
        """
        angle_definitions = [
            ("left_elbow", ["left_shoulder", "left_elbow", "left_wrist"]),
            ("right_elbow", ["right_shoulder", "right_elbow", "right_wrist"]),
            ("left_knee", ["left_hip", "left_knee", "left_ankle"]),
            ("right_knee", ["right_hip", "right_knee", "right_ankle"]),
            ("left_shoulder", ["left_elbow", "left_shoulder", "left_hip"]),
            ("right_shoulder", ["right_elbow", "right_shoulder", "right_hip"])
        ]
        
        joint_angles = {}
        keypoints = pose_result.keypoints
        
        for joint_name, joint_points in angle_definitions:
            try:
                indices = [self.keypoint_names.index(name) for name in joint_points]
                kpts = [keypoints[idx] for idx in indices]
                
                # 检查所有关键点是否有效
                if all(kpt.confidence > 0 for kpt in kpts):
                    angle = self._calculate_angle_between_points(
                        (kpts[0].x, kpts[0].y),
                        (kpts[1].x, kpts[1].y),
                        (kpts[2].x, kpts[2].y)
                    )
                    joint_angles[joint_name] = angle
                    
            except (ValueError, IndexError):
                continue
        
        return joint_angles
    
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
        vec1 = (x1 - x2, y1 - y2)
        vec2 = (x3 - x2, y3 - y2)
        
        # 计算点积和向量长度
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        len1 = np.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
        len2 = np.sqrt(vec2[0] ** 2 + vec2[1] ** 2)
        
        # 避免除零
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # 计算角度
        cos_angle = dot_product / (len1 * len2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 确保在有效范围内
        
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def extract_pose_features(self, pose_result: PoseDetectionResult) -> Dict[str, Any]:
        """
        提取姿态特征
        
        Args:
            pose_result: 姿态检测结果
            
        Returns:
            姿态特征字典
        """
        features = {
            "bone_lengths": self.calculate_bone_lengths(pose_result),
            "joint_angles": self.calculate_joint_angles(pose_result),
            "center_of_mass": self._calculate_center_of_mass(pose_result),
            "pose_confidence": pose_result.confidence,
            "visible_keypoints": sum(1 for kpt in pose_result.keypoints if kpt.confidence > 0)
        }
        
        return features
    
    def _calculate_center_of_mass(self, pose_result: PoseDetectionResult) -> tuple:
        """
        计算姿态重心
        
        Args:
            pose_result: 姿态检测结果
            
        Returns:
            重心坐标 (x, y)
        """
        valid_points = [kpt for kpt in pose_result.keypoints if kpt.confidence > 0]
        
        if not valid_points:
            return (0.0, 0.0)
        
        center_x = sum(kpt.x for kpt in valid_points) / len(valid_points)
        center_y = sum(kpt.y for kpt in valid_points) / len(valid_points)
        
        return (center_x, center_y) 