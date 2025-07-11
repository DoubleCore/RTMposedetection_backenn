"""
背景去除逻辑
"""
import cv2
import numpy as np
from typing import Optional, Tuple
import logging

from utils.data_structures import SegmentationResult

logger = logging.getLogger(__name__)

class BackgroundRemover:
    """
    背景去除器
    """
    
    def __init__(self):
        """
        初始化背景去除器
        """
        pass
    
    def remove_person_keep_pool(
        self, 
        image: np.ndarray, 
        person_mask: np.ndarray,
        pool_region: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        去除人物但保留游泳池背景
        
        Args:
            image: 原始图像
            person_mask: 人物分割mask
            pool_region: 游泳池区域mask（可选）
            
        Returns:
            处理后的图像
        """
        try:
            result = image.copy()
            
            if pool_region is None:
                # 如果没有提供游泳池区域，尝试自动检测
                pool_region = self._detect_pool_region(image)
            
            # 在人物区域填充游泳池背景色
            person_pixels = person_mask > 0
            
            if pool_region is not None:
                # 使用游泳池区域的平均颜色填充
                pool_pixels = pool_region > 0
                if np.any(pool_pixels):
                    pool_color = np.mean(image[pool_pixels], axis=0)
                    result[person_pixels] = pool_color
                else:
                    # 如果没有检测到游泳池，使用蓝色
                    result[person_pixels] = [255, 150, 100]  # 游泳池蓝色
            else:
                # 使用默认的游泳池颜色
                result[person_pixels] = [255, 150, 100]
            
            return result
            
        except Exception as e:
            logger.error(f"背景去除失败: {str(e)}")
            return image
    
    def _detect_pool_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        自动检测游泳池区域
        
        Args:
            image: 输入图像
            
        Returns:
            游泳池区域mask
        """
        try:
            # 转换到HSV颜色空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 定义蓝色范围（游泳池水的颜色）
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            
            # 创建蓝色mask
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # 形态学操作去除噪声
            kernel = np.ones((5, 5), np.uint8)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
            blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
            
            # 找到最大连通区域
            contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # 选择最大的轮廓作为游泳池
                largest_contour = max(contours, key=cv2.contourArea)
                pool_mask = np.zeros_like(blue_mask)
                cv2.fillPoly(pool_mask, [largest_contour], 255)
                return pool_mask
            
            return None
            
        except Exception as e:
            logger.error(f"游泳池检测失败: {str(e)}")
            return None
    
    def inpaint_person_region(
        self, 
        image: np.ndarray, 
        person_mask: np.ndarray,
        method: str = "telea"
    ) -> np.ndarray:
        """
        使用图像修复技术填充人物区域
        
        Args:
            image: 原始图像
            person_mask: 人物mask
            method: 修复方法 ("telea" 或 "ns")
            
        Returns:
            修复后的图像
        """
        try:
            # 确保mask是单通道
            if len(person_mask.shape) > 2:
                person_mask = cv2.cvtColor(person_mask, cv2.COLOR_BGR2GRAY)
            
            # 选择修复算法
            if method == "telea":
                inpaint_method = cv2.INPAINT_TELEA
            else:
                inpaint_method = cv2.INPAINT_NS
            
            # 执行图像修复
            result = cv2.inpaint(image, person_mask, 3, inpaint_method)
            
            return result
            
        except Exception as e:
            logger.error(f"图像修复失败: {str(e)}")
            return image
    
    def blend_with_background(
        self, 
        image: np.ndarray, 
        background: np.ndarray,
        person_mask: np.ndarray,
        blend_radius: int = 5
    ) -> np.ndarray:
        """
        将人物区域与背景平滑混合
        
        Args:
            image: 原始图像
            background: 背景图像
            person_mask: 人物mask
            blend_radius: 混合半径
            
        Returns:
            混合后的图像
        """
        try:
            # 创建混合mask
            blend_mask = cv2.GaussianBlur(person_mask.astype(np.float32), 
                                        (blend_radius*2+1, blend_radius*2+1), 0)
            blend_mask = blend_mask / 255.0
            
            # 确保维度匹配
            if len(blend_mask.shape) == 2:
                blend_mask = np.expand_dims(blend_mask, axis=2)
            
            # 混合图像
            result = image * (1 - blend_mask) + background * blend_mask
            result = result.astype(np.uint8)
            
            return result
            
        except Exception as e:
            logger.error(f"图像混合失败: {str(e)}")
            return image
    
    def create_pool_background(
        self, 
        image_shape: Tuple[int, int, int],
        pool_color: Tuple[int, int, int] = (255, 150, 100)
    ) -> np.ndarray:
        """
        创建游泳池背景
        
        Args:
            image_shape: 图像尺寸 (height, width, channels)
            pool_color: 游泳池颜色 (B, G, R)
            
        Returns:
            游泳池背景图像
        """
        try:
            h, w, c = image_shape
            background = np.full((h, w, c), pool_color, dtype=np.uint8)
            
            # 添加一些纹理效果
            noise = np.random.normal(0, 10, (h, w, c))
            background = np.clip(background + noise, 0, 255).astype(np.uint8)
            
            return background
            
        except Exception as e:
            logger.error(f"创建背景失败: {str(e)}")
            return np.zeros(image_shape, dtype=np.uint8)
    
    def enhance_pool_features(self, image: np.ndarray) -> np.ndarray:
        """
        增强游泳池特征
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像
        """
        try:
            # 转换到LAB颜色空间
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 增强蓝色通道
            b_enhanced = cv2.add(b, 20)
            b_enhanced = np.clip(b_enhanced, 0, 255)
            
            # 合并通道
            enhanced_lab = cv2.merge([l, a, b_enhanced])
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            return enhanced_bgr
            
        except Exception as e:
            logger.error(f"游泳池特征增强失败: {str(e)}")
            return image 