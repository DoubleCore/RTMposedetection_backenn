"""
图像处理工具函数
"""
import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """图像处理器"""
    
    def __init__(self):
        """初始化图像处理器"""
        pass
    
    async def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        预处理单帧图像
        
        Args:
            frame: 输入图像
            
        Returns:
            预处理后的图像
        """
        try:
            # 基本预处理
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # BGR转RGB（如果需要）
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pass
            
            # 可以添加其他预处理步骤，如降噪、增强等
            return frame
            
        except Exception as e:
            logger.error(f"图像预处理失败: {str(e)}")
            raise
    
    def resize_with_aspect_ratio(
        self, 
        image: np.ndarray, 
        target_size: Tuple[int, int],
        padding_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        保持宽高比的图像缩放
        
        Args:
            image: 输入图像
            target_size: 目标尺寸 (width, height)
            padding_color: 填充颜色
            
        Returns:
            (调整后的图像, 缩放因子, 填充偏移)
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 调整大小
        resized = cv2.resize(image, (new_w, new_h))
        
        # 创建目标尺寸的图像
        result = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
        
        # 计算填充位置
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        # 将调整后的图像放置在中心
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return result, scale, (x_offset, y_offset)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像归一化
        
        Args:
            image: 输入图像
            
        Returns:
            归一化后的图像
        """
        return image.astype(np.float32) / 255.0
    
    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像反归一化
        
        Args:
            image: 归一化的图像
            
        Returns:
            反归一化后的图像
        """
        return (image * 255.0).astype(np.uint8)
    
    def draw_keypoints(
        self, 
        image: np.ndarray, 
        keypoints: list,
        color: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 3
    ) -> np.ndarray:
        """
        在图像上绘制关键点
        
        Args:
            image: 输入图像
            keypoints: 关键点列表
            color: 绘制颜色
            radius: 关键点半径
            
        Returns:
            绘制后的图像
        """
        result = image.copy()
        
        for kpt in keypoints:
            if hasattr(kpt, 'confidence') and kpt.confidence > 0:
                cv2.circle(result, (int(kpt.x), int(kpt.y)), radius, color, -1)
        
        return result
    
    def draw_bbox(
        self,
        image: np.ndarray,
        bbox: list,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        在图像上绘制边界框
        
        Args:
            image: 输入图像
            bbox: 边界框 [x1, y1, x2, y2]
            color: 绘制颜色
            thickness: 线条粗细
            
        Returns:
            绘制后的图像
        """
        result = image.copy()
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        return result 