"""
FastSAM分割器
基于Ultralytics FastSAM实现，轻量化快速分割
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
import json

logger = logging.getLogger(__name__)

# 导入FastSAM
try:
    from ultralytics import FastSAM
    FASTSAM_AVAILABLE = True
    print("✅ FastSAM导入成功")
except ImportError as e:
    FASTSAM_AVAILABLE = False
    print(f"❌ FastSAM导入失败: {e}")
    print("请安装ultralytics: pip install ultralytics")

class FastSAMSegmenter:
    """
    FastSAM分割器（基于Ultralytics）
    """
    
    def __init__(self, model_name: str = 'FastSAM-s.pt'):
        """
        初始化FastSAM分割器
        
        Args:
            model_name: 模型名称 ('FastSAM-s.pt', 'FastSAM-x.pt')
        """
        if not FASTSAM_AVAILABLE:
            raise Exception("FastSAM不可用，请先安装: pip install ultralytics")
        
        self.model_name = model_name
        self.model = None
        self.confidence_threshold = 0.4
        self.iou_threshold = 0.9
        
        self._load_model()
        logger.info("FastSAM分割器初始化完成")
    
    def _load_model(self):
        """
        加载FastSAM模型（会自动下载）
        """
        try:
            print(f"🔄 正在初始化FastSAM模型: {self.model_name}")
            print("📥 首次运行会自动下载模型文件，请耐心等待...")
            
            # 使用FastSAM模型，会自动下载
            self.model = FastSAM(self.model_name)
            
            print(f"✅ FastSAM模型加载成功: {self.model_name}")
            logger.info(f"FastSAM模型加载成功: {self.model_name}")
            
        except Exception as e:
            print(f"❌ FastSAM模型加载失败: {str(e)}")
            logger.error(f"FastSAM模型加载失败: {str(e)}")
            raise Exception(f"FastSAM模型加载失败: {str(e)}")
    
    def segment_with_bbox(self, image: np.ndarray, bboxes: List[List[float]]) -> Dict[str, Any]:
        """
        使用边界框进行分割
        
        Args:
            image: 输入图像
            bboxes: 边界框列表 [[x1, y1, x2, y2], ...]
            
        Returns:
            分割结果
        """
        try:
            if self.model is None:
                raise Exception("FastSAM模型未初始化")
            
            # 使用FastSAM进行分割，提供边界框作为提示
            results = self.model.predict(
                image,
                bboxes=bboxes,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            return self._process_results(results, image, bboxes)
            
        except Exception as e:
            logger.error(f"FastSAM分割失败: {str(e)}")
            raise Exception(f"FastSAM分割失败: {str(e)}")
    
    def segment_everything(self, image: np.ndarray) -> Dict[str, Any]:
        """
        分割图像中的所有对象
        
        Args:
            image: 输入图像
            
        Returns:
            分割结果
        """
        try:
            if self.model is None:
                raise Exception("FastSAM模型未初始化")
            
            # 使用FastSAM进行全图分割
            results = self.model.predict(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            return self._process_results(results, image)
            
        except Exception as e:
            logger.error(f"FastSAM分割失败: {str(e)}")
            raise Exception(f"FastSAM分割失败: {str(e)}")
    
    def _process_results(self, results, image: np.ndarray, bboxes: Optional[List] = None) -> Dict[str, Any]:
        """
        处理FastSAM的输出结果
        
        Args:
            results: FastSAM输出结果
            image: 原始图像
            bboxes: 输入的边界框（可选）
            
        Returns:
            处理后的结果
        """
        try:
            segments = []
            masks = []
            
            if len(results) > 0:
                result = results[0]  # 获取第一个结果
                
                # 获取分割mask
                if hasattr(result, 'masks') and result.masks is not None:
                    for i, mask in enumerate(result.masks.data):
                        # 转换mask为numpy数组
                        mask_np = mask.cpu().numpy().astype(np.uint8)
                        
                        # 计算mask的边界框
                        y_indices, x_indices = np.where(mask_np > 0)
                        if len(x_indices) > 0 and len(y_indices) > 0:
                            x1, y1 = int(np.min(x_indices)), int(np.min(y_indices))
                            x2, y2 = int(np.max(x_indices)), int(np.max(y_indices))
                            
                            # 计算mask面积和置信度
                            area = int(np.sum(mask_np))
                            confidence = float(result.boxes.conf[i]) if hasattr(result, 'boxes') and result.boxes is not None else 0.9
                            
                            segment_info = {
                                "segment_id": i,
                                "bbox": [x1, y1, x2, y2],
                                "area": area,
                                "confidence": confidence,
                                "mask_shape": mask_np.shape
                            }
                            
                            segments.append(segment_info)
                            masks.append(mask_np)
            
            return {
                "num_segments": len(segments),
                "segments": segments,
                "masks": masks,
                "image_size": {
                    "width": image.shape[1],
                    "height": image.shape[0]
                },
                "model_info": {
                    "model_name": self.model_name,
                    "confidence_threshold": self.confidence_threshold,
                    "iou_threshold": self.iou_threshold
                },
                "input_bboxes": bboxes if bboxes else []
            }
            
        except Exception as e:
            logger.error(f"处理FastSAM结果失败: {str(e)}")
            return {
                "num_segments": 0,
                "segments": [],
                "masks": [],
                "image_size": {"width": image.shape[1], "height": image.shape[0]},
                "error": str(e)
            }
    
    def create_masked_image(self, image: np.ndarray, masks: List[np.ndarray], 
                           background_mode: str = "blur") -> np.ndarray:
        """
        创建带mask的图像
        
        Args:
            image: 原始图像
            masks: mask列表
            background_mode: 背景处理模式 ("blur", "black", "white", "transparent")
            
        Returns:
            处理后的图像
        """
        try:
            if not masks:
                return image.copy()
            
            # 合并所有mask
            combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            for mask in masks:
                if mask.shape[:2] == image.shape[:2]:
                    combined_mask = np.logical_or(combined_mask, mask > 0)
            
            # 创建输出图像
            output_image = image.copy()
            
            if background_mode == "blur":
                # 模糊背景
                blurred = cv2.GaussianBlur(image, (21, 21), 0)
                output_image[~combined_mask] = blurred[~combined_mask]
            elif background_mode == "black":
                # 黑色背景
                output_image[~combined_mask] = [0, 0, 0]
            elif background_mode == "white":
                # 白色背景
                output_image[~combined_mask] = [255, 255, 255]
            elif background_mode == "transparent":
                # 透明背景（需要RGBA）
                output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2BGRA)
                output_image[~combined_mask, 3] = 0  # 设置alpha通道
            
            return output_image
            
        except Exception as e:
            logger.error(f"创建masked图像失败: {str(e)}")
            return image.copy()
    
    def visualize_segments(self, image: np.ndarray, segments: List[Dict], 
                          show_bbox: bool = True, show_mask: bool = True) -> np.ndarray:
        """
        可视化分割结果
        
        Args:
            image: 原始图像
            segments: 分割结果
            show_bbox: 是否显示边界框
            show_mask: 是否显示mask轮廓
            
        Returns:
            可视化图像
        """
        try:
            viz_image = image.copy()
            
            # 生成颜色
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
            ]
            
            for i, segment in enumerate(segments):
                color = colors[i % len(colors)]
                
                if show_bbox and "bbox" in segment:
                    x1, y1, x2, y2 = segment["bbox"]
                    cv2.rectangle(viz_image, (x1, y1), (x2, y2), color, 2)
                    
                    # 添加标签
                    label = f"Seg {segment['segment_id']}: {segment.get('confidence', 0):.2f}"
                    cv2.putText(viz_image, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            return viz_image
            
        except Exception as e:
            logger.error(f"可视化分割结果失败: {str(e)}")
            return image.copy()

# 全局分割器实例（单例模式）
_segmenter_instance = None

def get_fast_sam_segmenter(model_name: str = 'FastSAM-s.pt') -> FastSAMSegmenter:
    """
    获取FastSAM分割器实例（单例）
    
    Args:
        model_name: 模型名称
        
    Returns:
        FastSAM分割器实例
    """
    global _segmenter_instance
    
    if _segmenter_instance is None:
        _segmenter_instance = FastSAMSegmenter(model_name)
    
    return _segmenter_instance 