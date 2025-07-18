"""
视频处理工具
负责将视频拆分成帧图像
"""

import cv2
import os
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class VideoFrameExtractor:
    """
    视频帧提取器
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化视频帧提取器
        
        Args:
            output_dir: 输出根目录
        """
        self.output_dir = output_dir
        self.origin_dir = os.path.join(output_dir, "origin")
        
        # 创建原始帧目录
        os.makedirs(self.origin_dir, exist_ok=True)
        
        logger.info(f"视频帧提取器初始化完成，原始帧目录: {self.origin_dir}")
    
    def extract_frames(
        self, 
        video_path: str, 
        task_id: str, 
        max_frames: Optional[int] = None,
        sample_method: str = "all"
    ) -> Dict[str, Any]:
        """
        从视频中提取帧图像
        
        Args:
            video_path: 视频文件路径
            task_id: 任务ID (用于创建子目录)
            max_frames: 最大提取帧数 (None表示提取所有帧)
            sample_method: 采样方法 ("all", "uniform", "interval")
            
        Returns:
            提取结果字典
        """
        try:
            logger.info(f"🎬 开始提取视频帧: {video_path}")
            
            # 为每个任务创建独立的子目录
            task_origin_dir = os.path.join(self.origin_dir, task_id)
            os.makedirs(task_origin_dir, exist_ok=True)
            
            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("无法打开视频文件")
            
            # 获取视频信息
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            logger.info(f"📊 视频信息: {frame_count}帧, {fps}FPS, {width}x{height}, {duration:.2f}秒")
            
            # 计算采样策略
            if sample_method == "all":
                # 提取所有帧
                sample_indices = list(range(frame_count))
                logger.info(f"📋 采样策略: 提取所有帧 ({frame_count} 帧)")
            elif sample_method == "uniform":
                # 均匀采样
                if max_frames is None:
                    max_frames = min(50, frame_count)  # 默认最多50帧
                sample_indices = self._calculate_uniform_sampling(frame_count, max_frames)
                logger.info(f"📋 采样策略: 均匀采样 {len(sample_indices)} 帧")
            else:
                # 间隔采样
                if max_frames is None:
                    max_frames = min(50, frame_count)  # 默认最多50帧
                sample_interval = max(1, frame_count // max_frames)
                sample_indices = list(range(0, frame_count, sample_interval))[:max_frames]
                logger.info(f"📋 采样策略: 间隔采样 {len(sample_indices)} 帧")
            
            # 提取帧图像
            extracted_frames = []
            frame_idx = 0
            saved_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 检查是否为采样帧
                if frame_idx in sample_indices:
                    saved_count += 1
                    
                    # 新的命名格式：frame_n.jpg (简洁命名)
                    frame_filename = f"frame_{saved_count}.jpg"
                    frame_path = os.path.join(task_origin_dir, frame_filename)
                    
                    # 保存图片
                    success = cv2.imwrite(frame_path, frame)
                    if success:
                        frame_info = {
                            "frame_number": saved_count,
                            "original_frame_index": frame_idx,
                            "timestamp": frame_idx / fps if fps > 0 else 0,
                            "filename": frame_filename,
                            "file_path": frame_path,
                            "resolution": {
                                "width": width,
                                "height": height
                            }
                        }
                        extracted_frames.append(frame_info)
                        
                        # 进度日志：每100帧打印一次，或前10帧每帧都打印
                        if saved_count % 100 == 0:
                            logger.info(f"📸 已保存 {saved_count} / {len(sample_indices)} 帧...")
                        elif saved_count <= 10:
                            logger.info(f"📸 保存帧 {saved_count}: {frame_filename} (原始帧号: {frame_idx})")
                    else:
                        logger.error(f"❌ 保存帧失败: {frame_filename}")
                
                frame_idx += 1
            
            cap.release()
            
            # 创建提取结果
            extraction_result = {
                "task_id": task_id,
                "video_info": {
                    "total_frames": frame_count,
                    "fps": fps,
                    "duration": duration,
                    "resolution": {
                        "width": width,
                        "height": height
                    }
                },
                "extraction_info": {
                    "sample_method": sample_method,
                    "max_frames": max_frames,
                    "extracted_count": len(extracted_frames),
                    "task_origin_dir": task_origin_dir
                },
                "extracted_frames": extracted_frames
            }
            
            logger.info(f"✅ 视频帧提取完成: 提取了 {len(extracted_frames)} 帧图像到 {task_origin_dir}")
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"视频帧提取失败: {str(e)}")
            raise
    
    def _calculate_uniform_sampling(self, total_frames: int, target_frames: int) -> List[int]:
        """
        计算均匀采样的帧索引
        
        Args:
            total_frames: 总帧数
            target_frames: 目标帧数
            
        Returns:
            采样帧索引列表
        """
        if target_frames >= total_frames:
            return list(range(total_frames))
        
        # 均匀分布采样
        indices = []
        step = total_frames / target_frames
        
        for i in range(target_frames):
            index = int(i * step)
            indices.append(index)
        
        return indices
    
    def get_extracted_frames(self, task_id: str) -> List[Dict[str, Any]]:
        """
        获取已提取的帧信息
        
        Args:
            task_id: 任务ID
            
        Returns:
            帧信息列表
        """
        frames = []
        task_origin_dir = os.path.join(self.origin_dir, task_id)
        
        if not os.path.exists(task_origin_dir):
            return frames
        
        frame_files = []
        
        # 查找帧文件
        for filename in os.listdir(task_origin_dir):
            if filename.startswith("frame_") and filename.endswith('.jpg'):
                frame_files.append(filename)
        
        # 按帧号排序
        frame_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        for i, filename in enumerate(frame_files):
            frame_path = os.path.join(task_origin_dir, filename)
            if os.path.exists(frame_path):
                # 读取图片获取尺寸信息
                try:
                    img = cv2.imread(frame_path)
                    if img is not None:
                        height, width = img.shape[:2]
                    else:
                        height, width = 0, 0
                    
                    frame_info = {
                        "frame_number": i + 1,
                        "filename": filename,
                        "file_path": frame_path,
                        "resolution": {
                            "width": width,
                            "height": height
                        }
                    }
                    frames.append(frame_info)
                except Exception as e:
                    logger.error(f"读取帧文件失败: {filename}, {str(e)}")
        
        return frames
    
    def cleanup_frames(self, task_id: str):
        """
        清理指定任务的帧文件
        
        Args:
            task_id: 任务ID
        """
        try:
            task_origin_dir = os.path.join(self.origin_dir, task_id)
            if os.path.exists(task_origin_dir):
                # 删除目录下的所有文件
                import shutil
                shutil.rmtree(task_origin_dir)
                logger.info(f"🗑️ 清理任务目录完成: {task_origin_dir}")
        except Exception as e:
            logger.error(f"清理帧文件失败: {str(e)}")

# 全局提取器实例
_extractor_instance = None

def get_frame_extractor(output_dir: str = "output") -> VideoFrameExtractor:
    """
    获取视频帧提取器实例（单例）
    
    Args:
        output_dir: 输出目录
        
    Returns:
        视频帧提取器实例
    """
    global _extractor_instance
    
    if _extractor_instance is None:
        _extractor_instance = VideoFrameExtractor(output_dir)
    
    return _extractor_instance 