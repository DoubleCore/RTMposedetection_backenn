"""
输入处理模块
负责处理图像和视频输入，统一输出为帧数据格式
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Union
from pathlib import Path
import logging

from utils.video_utils import VideoProcessor
from utils.image_utils import ImageProcessor
from utils.exceptions import InputError
from config.settings import settings

logger = logging.getLogger(__name__)

class InputHandler:
    """
    输入处理器类
    """
    
    def __init__(self):
        """
        初始化输入处理器
        """
        self.video_processor = VideoProcessor()
        self.image_processor = ImageProcessor()
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv'}
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        logger.info("输入处理器初始化完成")
    
    async def process_input(self, file_path: str) -> List[Dict[str, Any]]:
        """
        处理输入文件，返回统一的帧数据格式
        
        Args:
            file_path: 输入文件路径
            
        Returns:
            帧数据列表，每个元素包含：
            {
                "frame_id": int,
                "image": np.ndarray,
                "timestamp": float,
                "metadata": dict
            }
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise InputError(f"文件不存在: {file_path}")
            
            file_extension = file_path.suffix.lower()
            
            if file_extension in self.supported_video_formats:
                return await self._process_video(file_path)
            elif file_extension in self.supported_image_formats:
                return await self._process_image(file_path)
            else:
                raise InputError(f"不支持的文件格式: {file_extension}")
                
        except Exception as e:
            logger.error(f"输入处理失败: {file_path}, 错误: {str(e)}")
            raise InputError(f"输入处理失败: {str(e)}")
    
    async def _process_video(self, video_path: Path) -> List[Dict[str, Any]]:
        """
        处理视频文件
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频帧数据列表
        """
        try:
            logger.info(f"开始处理视频: {video_path}")
            
            # 打开视频文件
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise InputError(f"无法打开视频文件: {video_path}")
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
            
            frames_data = []
            frame_id = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 预处理帧
                processed_frame = await self.image_processor.preprocess_frame(frame)
                
                # 计算时间戳
                timestamp = frame_id / fps
                
                frame_data = {
                    "frame_id": frame_id,
                    "image": processed_frame,
                    "timestamp": timestamp,
                    "metadata": {
                        "source_type": "video",
                        "source_path": str(video_path),
                        "original_size": (width, height),
                        "fps": fps,
                        "total_frames": total_frames
                    }
                }
                
                frames_data.append(frame_data)
                frame_id += 1
            
            cap.release()
            logger.info(f"视频处理完成，共{len(frames_data)}帧")
            
            return frames_data
            
        except Exception as e:
            logger.error(f"视频处理失败: {video_path}, 错误: {str(e)}")
            raise InputError(f"视频处理失败: {str(e)}")
    
    async def _process_image(self, image_path: Path) -> List[Dict[str, Any]]:
        """
        处理图像文件
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            单帧数据列表
        """
        try:
            logger.info(f"开始处理图像: {image_path}")
            
            # 读取图像
            image = cv2.imread(str(image_path))
            if image is None:
                raise InputError(f"无法读取图像文件: {image_path}")
            
            height, width = image.shape[:2]
            
            # 预处理图像
            processed_image = await self.image_processor.preprocess_frame(image)
            
            frame_data = {
                "frame_id": 0,
                "image": processed_image,
                "timestamp": 0.0,
                "metadata": {
                    "source_type": "image",
                    "source_path": str(image_path),
                    "original_size": (width, height),
                    "fps": None,
                    "total_frames": 1
                }
            }
            
            logger.info(f"图像处理完成: {width}x{height}")
            
            return [frame_data]
            
        except Exception as e:
            logger.error(f"图像处理失败: {image_path}, 错误: {str(e)}")
            raise InputError(f"图像处理失败: {str(e)}")
    
    def validate_input(self, file_path: str) -> bool:
        """
        验证输入文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否为有效输入
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return False
            
            file_extension = file_path.suffix.lower()
            
            return (file_extension in self.supported_video_formats or 
                    file_extension in self.supported_image_formats)
                    
        except Exception:
            return False
    
    def get_input_info(self, file_path: str) -> Dict[str, Any]:
        """
        获取输入文件信息
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件信息字典
        """
        try:
            file_path = Path(file_path)
            file_extension = file_path.suffix.lower()
            
            info = {
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "file_extension": file_extension,
                "file_type": "video" if file_extension in self.supported_video_formats else "image"
            }
            
            if file_extension in self.supported_video_formats:
                # 获取视频信息
                cap = cv2.VideoCapture(str(file_path))
                if cap.isOpened():
                    info.update({
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "fps": cap.get(cv2.CAP_PROP_FPS),
                        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
                    })
                cap.release()
            
            elif file_extension in self.supported_image_formats:
                # 获取图像信息
                image = cv2.imread(str(file_path))
                if image is not None:
                    height, width = image.shape[:2]
                    info.update({
                        "width": width,
                        "height": height,
                        "total_frames": 1,
                        "duration": 0
                    })
            
            return info
            
        except Exception as e:
            logger.error(f"获取文件信息失败: {file_path}, 错误: {str(e)}")
            return {"error": str(e)} 