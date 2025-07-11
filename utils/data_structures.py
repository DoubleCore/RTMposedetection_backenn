"""
数据结构定义
"""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class Keypoint:
    """关键点数据结构"""
    x: float
    y: float
    confidence: float

@dataclass
class PoseDetectionResult:
    """姿态检测结果"""
    keypoints: List[Keypoint]
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    person_id: Optional[int] = None

@dataclass
class SegmentationResult:
    """分割结果"""
    mask: np.ndarray
    bbox: List[float]
    confidence: float
    person_id: Optional[int] = None

@dataclass
class TrackingResult:
    """追踪结果"""
    person_id: int
    bbox: List[float]
    confidence: float
    pose_data: Optional[PoseDetectionResult] = None
    segmentation_data: Optional[SegmentationResult] = None

@dataclass
class SpeedMetrics:
    """速度指标"""
    velocity_x: float
    velocity_y: float
    speed_magnitude: float
    acceleration: float
    frame_id: int

@dataclass
class AngleMetrics:
    """角度指标"""
    joint_name: str
    angle: float
    frame_id: int

@dataclass
class AnalysisMetrics:
    """综合分析指标"""
    person_id: int
    frame_id: int
    speed_metrics: Optional[SpeedMetrics] = None
    angle_metrics: List[AngleMetrics] = None
    
    def __post_init__(self):
        if self.angle_metrics is None:
            self.angle_metrics = [] 