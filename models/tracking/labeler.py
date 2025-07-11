"""
人物标签化模块
"""
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from utils.data_structures import TrackingResult, PoseDetectionResult

logger = logging.getLogger(__name__)

class PersonLabeler:
    """
    人物标签化器
    """
    
    def __init__(self):
        """
        初始化人物标签化器
        """
        self.person_labels = {}  # person_id -> label_info
        self.label_colors = self._generate_label_colors()
        
    def _generate_label_colors(self) -> List[tuple]:
        """
        生成标签颜色
        
        Returns:
            颜色列表
        """
        colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 黄色
            (255, 0, 255),  # 洋红色
            (0, 255, 255),  # 青色
            (255, 128, 0),  # 橙色
            (128, 0, 255),  # 紫色
            (255, 192, 203), # 粉色
            (0, 128, 128),  # 青绿色
        ]
        return colors
    
    def assign_labels(
        self, 
        tracking_results: List[TrackingResult]
    ) -> List[Dict[str, Any]]:
        """
        为追踪到的人物分配标签
        
        Args:
            tracking_results: 追踪结果列表
            
        Returns:
            带标签的结果列表
        """
        labeled_results = []
        
        for track_result in tracking_results:
            person_id = track_result.person_id
            
            # 如果是新人物，分配新标签
            if person_id not in self.person_labels:
                self._create_new_label(person_id, track_result)
            
            # 更新标签信息
            self._update_label_info(person_id, track_result)
            
            # 创建带标签的结果
            labeled_result = {
                "person_id": person_id,
                "label": self.person_labels[person_id]["label"],
                "color": self.person_labels[person_id]["color"],
                "tracking_data": track_result,
                "label_confidence": self._calculate_label_confidence(person_id)
            }
            
            labeled_results.append(labeled_result)
        
        return labeled_results
    
    def _create_new_label(
        self, 
        person_id: int, 
        track_result: TrackingResult
    ):
        """
        为新人物创建标签
        
        Args:
            person_id: 人物ID
            track_result: 追踪结果
        """
        # 分析人物特征来决定标签
        label = self._analyze_person_characteristics(track_result)
        
        # 分配颜色
        color_idx = (person_id - 1) % len(self.label_colors)
        color = self.label_colors[color_idx]
        
        self.person_labels[person_id] = {
            "label": label,
            "color": color,
            "first_seen": track_result,
            "characteristics": self._extract_characteristics(track_result),
            "frame_count": 1
        }
        
        logger.info(f"为人物 {person_id} 分配标签: {label}")
    
    def _analyze_person_characteristics(
        self, 
        track_result: TrackingResult
    ) -> str:
        """
        分析人物特征
        
        Args:
            track_result: 追踪结果
            
        Returns:
            人物标签
        """
        # 简化的特征分析
        # 实际应用中可以基于姿态、位置、大小等特征进行更复杂的分析
        
        pose_data = track_result.pose_data
        bbox = track_result.bbox
        
        # 基于边界框大小判断
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        if bbox_area > 50000:  # 大尺寸
            size_label = "Large"
        elif bbox_area > 20000:  # 中等尺寸
            size_label = "Medium"
        else:  # 小尺寸
            size_label = "Small"
        
        # 基于位置判断（假设跳水台在图像上方）
        center_y = (bbox[1] + bbox[3]) / 2
        if center_y < 200:  # 图像上方
            position_label = "High"
        elif center_y < 400:  # 图像中部
            position_label = "Mid"
        else:  # 图像下方
            position_label = "Low"
        
        # 组合标签
        label = f"Person_{size_label}_{position_label}"
        return label
    
    def _extract_characteristics(
        self, 
        track_result: TrackingResult
    ) -> Dict[str, Any]:
        """
        提取人物特征
        
        Args:
            track_result: 追踪结果
            
        Returns:
            特征字典
        """
        characteristics = {}
        
        if track_result.pose_data:
            pose_data = track_result.pose_data
            
            # 身体比例特征
            bbox = pose_data.bbox
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            aspect_ratio = height / width if width > 0 else 0
            
            characteristics.update({
                "bbox_width": width,
                "bbox_height": height,
                "aspect_ratio": aspect_ratio,
                "bbox_area": width * height,
                "pose_confidence": pose_data.confidence
            })
            
            # 关键点特征
            valid_keypoints = sum(1 for kpt in pose_data.keypoints if kpt.confidence > 0)
            characteristics["visible_keypoints"] = valid_keypoints
        
        if track_result.segmentation_data:
            seg_data = track_result.segmentation_data
            characteristics.update({
                "has_segmentation": True,
                "segmentation_confidence": seg_data.confidence
            })
        else:
            characteristics["has_segmentation"] = False
        
        return characteristics
    
    def _update_label_info(
        self, 
        person_id: int, 
        track_result: TrackingResult
    ):
        """
        更新标签信息
        
        Args:
            person_id: 人物ID
            track_result: 追踪结果
        """
        if person_id in self.person_labels:
            self.person_labels[person_id]["frame_count"] += 1
            
            # 更新特征（可以实现特征的平滑更新）
            new_characteristics = self._extract_characteristics(track_result)
            self._smooth_characteristics(person_id, new_characteristics)
    
    def _smooth_characteristics(
        self, 
        person_id: int, 
        new_characteristics: Dict[str, Any]
    ):
        """
        平滑更新特征
        
        Args:
            person_id: 人物ID
            new_characteristics: 新特征
        """
        old_characteristics = self.person_labels[person_id]["characteristics"]
        frame_count = self.person_labels[person_id]["frame_count"]
        
        # 使用移动平均平滑数值特征
        alpha = 0.1  # 平滑系数
        
        for key, new_value in new_characteristics.items():
            if key in old_characteristics and isinstance(new_value, (int, float)):
                old_value = old_characteristics[key]
                smoothed_value = old_value * (1 - alpha) + new_value * alpha
                old_characteristics[key] = smoothed_value
            else:
                old_characteristics[key] = new_value
    
    def _calculate_label_confidence(self, person_id: int) -> float:
        """
        计算标签置信度
        
        Args:
            person_id: 人物ID
            
        Returns:
            标签置信度
        """
        if person_id not in self.person_labels:
            return 0.0
        
        frame_count = self.person_labels[person_id]["frame_count"]
        
        # 基于帧数计算置信度（帧数越多，置信度越高）
        confidence = min(1.0, frame_count / 10.0)  # 10帧后达到最大置信度
        
        return confidence
    
    def get_person_summary(self, person_id: int) -> Optional[Dict[str, Any]]:
        """
        获取人物摘要信息
        
        Args:
            person_id: 人物ID
            
        Returns:
            人物摘要
        """
        if person_id not in self.person_labels:
            return None
        
        person_info = self.person_labels[person_id]
        
        summary = {
            "person_id": person_id,
            "label": person_info["label"],
            "color": person_info["color"],
            "frame_count": person_info["frame_count"],
            "characteristics": person_info["characteristics"],
            "confidence": self._calculate_label_confidence(person_id)
        }
        
        return summary
    
    def get_all_persons_summary(self) -> List[Dict[str, Any]]:
        """
        获取所有人物的摘要信息
        
        Returns:
            所有人物摘要列表
        """
        summaries = []
        
        for person_id in self.person_labels.keys():
            summary = self.get_person_summary(person_id)
            if summary:
                summaries.append(summary)
        
        return summaries
    
    def relabel_person(self, person_id: int, new_label: str):
        """
        重新标记人物
        
        Args:
            person_id: 人物ID
            new_label: 新标签
        """
        if person_id in self.person_labels:
            old_label = self.person_labels[person_id]["label"]
            self.person_labels[person_id]["label"] = new_label
            logger.info(f"人物 {person_id} 标签从 '{old_label}' 更改为 '{new_label}'")
    
    def merge_persons(self, person_id1: int, person_id2: int) -> int:
        """
        合并两个人物（如果发现是同一个人）
        
        Args:
            person_id1: 人物ID1
            person_id2: 人物ID2
            
        Returns:
            合并后的人物ID
        """
        if person_id1 in self.person_labels and person_id2 in self.person_labels:
            # 保留帧数更多的那个
            if (self.person_labels[person_id1]["frame_count"] >= 
                self.person_labels[person_id2]["frame_count"]):
                keep_id, remove_id = person_id1, person_id2
            else:
                keep_id, remove_id = person_id2, person_id1
            
            # 合并特征
            self.person_labels[keep_id]["frame_count"] += self.person_labels[remove_id]["frame_count"]
            
            # 删除被合并的人物
            del self.person_labels[remove_id]
            
            logger.info(f"合并人物 {remove_id} 到 {keep_id}")
            return keep_id
        
        return person_id1 