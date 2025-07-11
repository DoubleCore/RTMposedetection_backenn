"""
RTMPose姿态检测器
基于rtmlib实现，自动下载和管理模型
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import json
import os

from utils.exceptions import ModelError
from utils.data_structures import PoseDetectionResult, Keypoint

logger = logging.getLogger(__name__)

# 导入rtmlib
try:
    from rtmlib import Body, draw_skeleton
    RTMLIB_AVAILABLE = True
    print("✅ rtmlib导入成功")
except ImportError as e:
    RTMLIB_AVAILABLE = False
    print(f"❌ rtmlib导入失败: {e}")
    print("请安装rtmlib: pip install rtmlib")

class RTMPoseDetector:
    """
    RTMPose姿态检测器（基于rtmlib）
    """
    
    def __init__(self, model_name: str = 'rtmo', mode: str = 'balanced'):
        """
        初始化RTMPose检测器
        
        Args:
            model_name: 模型名称 ('rtmo', 'rtmpose' 等)
            mode: 模式 ('performance', 'lightweight', 'balanced')
        """
        if not RTMLIB_AVAILABLE:
            raise ModelError("rtmlib不可用，请先安装: pip install rtmlib")
        
        self.model_name = model_name
        self.mode = mode
        self.confidence_threshold = 0.3
        self.body_model = None
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        self._load_model()
        logger.info("RTMPose检测器初始化完成")
    
    def _load_model(self):
        """
        加载RTMPose模型（rtmlib会自动下载）
        """
        try:
            print(f"🔄 正在初始化RTMPose模型: {self.model_name} ({self.mode})")
            print("📥 首次运行会自动下载模型文件，请耐心等待...")
            
            # 使用rtmlib的Body类，会自动下载模型
            self.body_model = Body(
                pose=self.model_name,
                to_openpose=False,  # 使用COCO17格式
                mode=self.mode,
                backend='onnxruntime',
                device='cpu'  # 可以改为 'cuda' 如果有GPU
            )
            
            print(f"✅ RTMPose模型加载成功: {self.model_name}")
            logger.info(f"RTMPose模型加载成功: {self.model_name}")
            
        except Exception as e:
            print(f"❌ RTMPose模型加载失败: {str(e)}")
            logger.error(f"RTMPose模型加载失败: {str(e)}")
            raise ModelError(f"RTMPose模型加载失败: {str(e)}")
    
    async def detect_pose(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        检测单帧的人体姿态
        
        Args:
            frame_data: 帧数据字典，包含 'image' 和其他信息
            
        Returns:
            姿态检测结果
        """
        try:
            image = frame_data["image"]
            frame_id = frame_data.get("frame_id", 0)
            timestamp = frame_data.get("timestamp", 0)
            
            # 使用rtmlib进行姿态检测
            if self.body_model is None:
                raise ModelError("RTMPose模型未初始化")
            keypoints, scores = self.body_model(image)
            
            # 转换为我们的数据格式
            pose_results = self._convert_to_pose_results(keypoints, scores, image.shape[:2])
            
            return {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "pose_results": pose_results,
                "metadata": {
                    "model": f"RTMPose-{self.model_name}",
                    "confidence_threshold": self.confidence_threshold,
                    "original_size": image.shape[:2],
                    "num_persons": len(pose_results)
                }
            }
            
        except Exception as e:
            logger.error(f"姿态检测失败: {str(e)}")
            raise ModelError(f"姿态检测失败: {str(e)}")
    
    def detect_pose_simple(self, image: np.ndarray) -> tuple:
        """
        简单的姿态检测接口，直接返回关键点和分数
        
        Args:
            image: 输入图像
            
        Returns:
            (keypoints, scores) 元组
        """
        try:
            if self.body_model is None:
                raise ModelError("RTMPose模型未初始化")
            return self.body_model(image)
        except Exception as e:
            logger.error(f"姿态检测失败: {str(e)}")
            raise ModelError(f"姿态检测失败: {str(e)}")
    
    def _convert_to_pose_results(
        self, 
        keypoints: np.ndarray, 
        scores: np.ndarray,
        image_size: tuple
    ) -> List[PoseDetectionResult]:
        """
        将rtmlib的输出转换为我们的数据结构
        
        Args:
            keypoints: 关键点数组，形状为 [num_persons, num_keypoints, 2]
            scores: 置信度数组，形状为 [num_persons, num_keypoints]
            image_size: 图像尺寸 (height, width)
            
        Returns:
            姿态检测结果列表
        """
        pose_results = []
        
        for person_idx in range(len(keypoints)):
            person_keypoints = keypoints[person_idx]
            person_scores = scores[person_idx]
            
            # 转换关键点
            converted_keypoints = []
            valid_points = []
            
            for kpt_idx in range(len(person_keypoints)):
                x, y = person_keypoints[kpt_idx]
                conf = person_scores[kpt_idx]
                
                keypoint = Keypoint(x=float(x), y=float(y), confidence=float(conf))
                converted_keypoints.append(keypoint)
                
                if conf > self.confidence_threshold:
                    valid_points.append((float(x), float(y)))
            
            # 如果有效关键点太少，跳过这个人
            if len(valid_points) < 5:
                continue
            
            # 计算边界框
            if valid_points:
                x_coords = [pt[0] for pt in valid_points]
                y_coords = [pt[1] for pt in valid_points]
                
                # 添加一些边距
                margin = 20
                bbox = [
                    max(0, min(x_coords) - margin),  # x1
                    max(0, min(y_coords) - margin),  # y1
                    min(image_size[1], max(x_coords) + margin),  # x2
                    min(image_size[0], max(y_coords) + margin)   # y2
                ]
                
                # 计算平均置信度
                valid_scores = [person_scores[i] for i in range(len(person_scores)) 
                              if person_scores[i] > self.confidence_threshold]
                avg_confidence = float(np.mean(valid_scores)) if valid_scores else 0.0
                
                pose_result = PoseDetectionResult(
                    keypoints=converted_keypoints,
                    bbox=bbox,
                    confidence=avg_confidence
                )
                
                pose_results.append(pose_result)
        
        return pose_results
    
    def draw_pose_on_image(self, image: np.ndarray, keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """
        在图像上绘制姿态关键点
        
        Args:
            image: 输入图像
            keypoints: 关键点数组
            scores: 置信度数组
            
        Returns:
            绘制后的图像
        """
        try:
            return draw_skeleton(
                image.copy(),
                keypoints,
                scores,
                openpose_skeleton=False,  # 使用COCO17格式
                kpt_thr=self.confidence_threshold,
                line_width=2
            )
        except Exception as e:
            logger.error(f"绘制姿态失败: {str(e)}")
            return image.copy()
    
    def export_to_json(self, keypoints: np.ndarray, scores: np.ndarray, image_info: dict) -> dict:
        """
        将检测结果导出为JSON格式
        
        Args:
            keypoints: 关键点数组
            scores: 置信度数组
            image_info: 图像信息字典
            
        Returns:
            JSON格式的结果
        """
        json_data = {
            "image_name": image_info.get("name", "unknown"),
            "image_size": {
                "width": int(image_info.get("width", 0)),
                "height": int(image_info.get("height", 0))
            },
            "model_info": {
                "model_name": self.model_name,
                "mode": self.mode,
                "confidence_threshold": self.confidence_threshold
            },
            "num_persons": len(keypoints),
            "persons": []
        }
        
        # 为每个人添加数据
        for person_idx in range(len(keypoints)):
            person_data = {
                "person_id": person_idx,
                "keypoints": keypoints[person_idx].tolist(),
                "scores": scores[person_idx].tolist(),
                "skeleton_format": "COCO17",
                "keypoint_names": self.keypoint_names
            }
            json_data["persons"].append(person_data)
        
        return json_data
    
    async def cleanup(self):
        """
        清理资源
        """
        try:
            if self.body_model:
                self.body_model = None
            logger.info("RTMPose检测器资源清理完成")
        except Exception as e:
            logger.error(f"RTMPose资源清理失败: {str(e)}")

# 全局检测器实例（单例模式）
_detector_instance = None

def get_rtmpose_detector(model_name: str = 'rtmo', mode: str = 'balanced') -> RTMPoseDetector:
    """
    获取RTMPose检测器实例（单例）
    
    Args:
        model_name: 模型名称
        mode: 模式
        
    Returns:
        RTMPose检测器实例
    """
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = RTMPoseDetector(model_name, mode)
    
    return _detector_instance 