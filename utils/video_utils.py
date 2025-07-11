"""
视频处理工具函数
"""
import cv2
import numpy as np
from typing import Generator, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VideoProcessor:
    """视频处理器"""
    
    def __init__(self):
        """初始化视频处理器"""
        pass
    
    def read_video_frames(self, video_path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        读取视频帧的生成器
        
        Args:
            video_path: 视频文件路径
            
        Yields:
            (帧索引, 帧图像)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                yield frame_idx, frame
                frame_idx += 1
        finally:
            cap.release()
    
    def get_video_info(self, video_path: str) -> dict:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            视频信息字典
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        try:
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
            }
            return info
        finally:
            cap.release()
    
    def extract_frames(
        self, 
        video_path: str, 
        output_dir: str,
        frame_interval: int = 1,
        max_frames: Optional[int] = None
    ) -> list:
        """
        提取视频帧并保存
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
            frame_interval: 帧间隔
            max_frames: 最大帧数
            
        Returns:
            保存的帧文件路径列表
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        saved_frames = []
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        frame_idx = 0
        saved_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    if max_frames is not None and saved_count >= max_frames:
                        break
                    
                    frame_filename = f"frame_{frame_idx:06d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    
                    cv2.imwrite(frame_path, frame)
                    saved_frames.append(frame_path)
                    saved_count += 1
                
                frame_idx += 1
                
        finally:
            cap.release()
        
        logger.info(f"提取了 {len(saved_frames)} 帧到 {output_dir}")
        return saved_frames
    
    def create_video_from_frames(
        self,
        frame_paths: list,
        output_path: str,
        fps: float = 30.0,
        codec: str = 'mp4v'
    ):
        """
        从帧序列创建视频
        
        Args:
            frame_paths: 帧文件路径列表
            output_path: 输出视频路径
            fps: 帧率
            codec: 编码器
        """
        if not frame_paths:
            raise ValueError("帧路径列表为空")
        
        # 读取第一帧获取尺寸
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            raise ValueError(f"无法读取第一帧: {frame_paths[0]}")
        
        height, width = first_frame.shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    out.write(frame)
                else:
                    logger.warning(f"无法读取帧: {frame_path}")
        finally:
            out.release()
        
        logger.info(f"视频已保存到: {output_path}")
    
    def resize_video(
        self,
        input_path: str,
        output_path: str,
        target_size: Tuple[int, int],
        maintain_aspect_ratio: bool = True
    ):
        """
        调整视频尺寸
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            target_size: 目标尺寸 (width, height)
            maintain_aspect_ratio: 是否保持宽高比
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_path}")
        
        # 获取原始视频信息
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, original_fps, target_size)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if maintain_aspect_ratio:
                    # 保持宽高比调整
                    h, w = frame.shape[:2]
                    target_w, target_h = target_size
                    
                    scale = min(target_w / w, target_h / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    
                    resized_frame = cv2.resize(frame, (new_w, new_h))
                    
                    # 创建目标尺寸的黑色背景
                    result_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    
                    # 计算填充位置
                    y_offset = (target_h - new_h) // 2
                    x_offset = (target_w - new_w) // 2
                    
                    result_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
                else:
                    result_frame = cv2.resize(frame, target_size)
                
                out.write(result_frame)
                
        finally:
            cap.release()
            out.release()
        
        logger.info(f"调整后的视频已保存到: {output_path}") 