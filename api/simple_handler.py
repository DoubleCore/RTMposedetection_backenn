"""
简化的文件处理接口
专门处理RTMPose检测和结果输出
"""

import cv2
import numpy as np
import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional
import logging

from models.rtmpose.detector import get_rtmpose_detector
from utils.exceptions import ProcessingError
from utils.video_processor import get_frame_extractor

logger = logging.getLogger(__name__)

class SimpleFileHandler:
    """
    简化的文件处理器
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化处理器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.json_dir = os.path.join(output_dir, "json")
        self.images_dir = os.path.join(output_dir, "images")
        self.origin_dir = os.path.join(output_dir, "origin")
        self.temp_dir = os.path.join(output_dir, "temp")
        
        # 创建输出目录
        for dir_path in [self.output_dir, self.json_dir, self.images_dir, self.origin_dir, self.temp_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 获取RTMPose检测器和视频帧提取器
        self.detector = get_rtmpose_detector()
        self.frame_extractor = get_frame_extractor(output_dir)
        
        logger.info(f"简化文件处理器初始化完成，输出目录: {output_dir}")
    
    async def process_image_file(self, file_path: str, task_id: str, filename: str) -> Dict[str, Any]:
        """
        处理图片文件
        
        Args:
            file_path: 图片文件路径
            task_id: 任务ID
            filename: 原始文件名
            
        Returns:
            处理结果
        """
        try:
            logger.info(f"🖼️ 开始处理图片: {filename}")
            
            # 读取图片
            image = cv2.imread(file_path)
            if image is None:
                raise ProcessingError("无法读取图片文件")
            
            # RTMPose姿态检测
            keypoints, scores = self.detector.detect_pose_simple(image)
            num_persons = len(keypoints)
            
            logger.info(f"👥 检测到 {num_persons} 个人")
            
            # 创建JSON结果
            json_result = {
                "task_id": task_id,
                "file_name": filename,
                "file_type": "image",
                "image_size": {
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0])
                },
                "analysis_time": datetime.now().isoformat(),
                "num_persons": num_persons,
                "model_info": {
                    "model_name": self.detector.model_name,
                    "mode": self.detector.mode,
                    "confidence_threshold": self.detector.confidence_threshold
                },
                "persons": []
            }
            
            # 添加每个人的关键点数据
            for person_idx in range(num_persons):
                person_data = {
                    "person_id": person_idx,
                    "keypoints": keypoints[person_idx].tolist(),
                    "scores": scores[person_idx].tolist(),
                    "skeleton_format": "COCO17",
                    "keypoint_names": self.detector.keypoint_names,
                    "avg_confidence": float(np.mean(scores[person_idx]))
                }
                json_result["persons"].append(person_data)
            
            # 保存JSON结果
            json_path = os.path.join(self.json_dir, f"{task_id}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_result, f, indent=2, ensure_ascii=False)
            
            # 绘制并保存标注图片
            annotated_image = self.detector.draw_pose_on_image(image, keypoints, scores)
            annotated_filename = f"{task_id}_frame1.jpg"
            annotated_path = os.path.join(self.images_dir, annotated_filename)
            cv2.imwrite(annotated_path, annotated_image)
            
            logger.info(f"✅ 图片处理完成: JSON={json_path}, 图片={annotated_path}")
            
            return {
                "type": "image",
                "num_persons": num_persons,
                "json_file": f"{task_id}.json",
                "annotated_image": annotated_filename,
                "processing_time": "immediate",
                "results": json_result
            }
            
        except Exception as e:
            logger.error(f"图片处理失败: {str(e)}")
            raise ProcessingError(f"图片处理失败: {str(e)}")
    
    async def process_video_file(self, file_path: str, task_id: str, filename: str) -> Dict[str, Any]:
        """
        处理视频文件（新流程：先提取帧→再进行姿态检测）
        
        Args:
            file_path: 视频文件路径
            task_id: 任务ID
            filename: 原始文件名
            
        Returns:
            处理结果
        """
        try:
            logger.info(f"🎥 开始处理视频: {filename}")
            
            # 第一步：提取视频帧到 origin 目录
            logger.info("📸 第一步：提取视频帧到原始目录...")
            extraction_result = self.frame_extractor.extract_frames(
                video_path=file_path,
                task_id=task_id,
                max_frames=20,
                sample_method="uniform"
            )
            
            extracted_frames = extraction_result["extracted_frames"]
            video_info = extraction_result["video_info"]
            
            logger.info(f"✅ 帧提取完成: 提取了 {len(extracted_frames)} 帧到 origin 目录")
            
            # 第二步：对提取的帧进行姿态检测
            logger.info("🤖 第二步：对原始帧进行姿态检测...")
            
            processed_frames = []
            annotated_images = []
            total_persons = 0
            
            for frame_info in extracted_frames:
                frame_path = frame_info["file_path"]
                frame_number = frame_info["frame_number"]
                original_frame_index = frame_info["original_frame_index"]
                timestamp = frame_info["timestamp"]
                
                try:
                    # 读取原始帧图像
                    image = cv2.imread(frame_path)
                    if image is None:
                        logger.warning(f"⚠️ 无法读取帧图像: {frame_path}")
                        continue
                    
                    # RTMPose姿态检测
                    keypoints, scores = self.detector.detect_pose_simple(image)
                    num_persons = len(keypoints)
                    total_persons += num_persons
                    
                    # 记录帧数据
                    frame_data = {
                        "frame_number": frame_number,
                        "original_frame_index": original_frame_index,
                        "timestamp": timestamp,
                        "num_persons": num_persons,
                        "origin_image": frame_info["filename"],
                        "persons": []
                    }
                    
                    # 添加人员数据
                    for person_idx in range(num_persons):
                        person_data = {
                            "person_id": person_idx,
                            "keypoints": keypoints[person_idx].tolist(),
                            "scores": scores[person_idx].tolist(),
                            "skeleton_format": "COCO17",
                            "keypoint_names": self.detector.keypoint_names,
                            "avg_confidence": float(np.mean(scores[person_idx]))
                        }
                        frame_data["persons"].append(person_data)
                    
                    processed_frames.append(frame_data)
                    
                    # 保存标注图片
                    annotated_frame = self.detector.draw_pose_on_image(image, keypoints, scores)
                    annotated_filename = f"{task_id}_frame{frame_number}.jpg"
                    annotated_path = os.path.join(self.images_dir, annotated_filename)
                    cv2.imwrite(annotated_path, annotated_frame)
                    annotated_images.append(annotated_filename)
                    
                    logger.info(f"✅ 处理帧 {frame_number}: 检测到 {num_persons} 个人 (原始帧号: {original_frame_index})")
                    
                except Exception as e:
                    logger.error(f"❌ 处理帧 {frame_number} 失败: {str(e)}")
                    continue
            
            # 创建JSON结果
            json_result = {
                "task_id": task_id,
                "file_name": filename,
                "file_type": "video",
                "video_info": video_info,
                "extraction_info": extraction_result["extraction_info"],
                "analysis_time": datetime.now().isoformat(),
                "processed_frames": len(processed_frames),
                "total_persons_detected": total_persons,
                "model_info": {
                    "model_name": self.detector.model_name,
                    "mode": self.detector.mode,
                    "confidence_threshold": self.detector.confidence_threshold
                },
                "frames": processed_frames,
                "file_structure": {
                    "origin_frames": [f["filename"] for f in extracted_frames],
                    "annotated_frames": annotated_images
                }
            }
            
            # 保存JSON结果
            json_path = os.path.join(self.json_dir, f"{task_id}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 视频处理完成: JSON={json_path}")
            logger.info(f"📁 原始帧数: {len(extracted_frames)}, 标注帧数: {len(annotated_images)}")
            
            return {
                "type": "video",
                "video_info": video_info,
                "extraction_info": extraction_result["extraction_info"],
                "processed_frames": len(processed_frames),
                "total_persons_detected": total_persons,
                "json_file": f"{task_id}.json",
                "annotated_images": annotated_images,
                "origin_images": [f["filename"] for f in extracted_frames],
                "processing_time": f"{len(processed_frames)} frames processed",
                "results": json_result
            }
            
        except Exception as e:
            logger.error(f"视频处理失败: {str(e)}")
            raise ProcessingError(f"视频处理失败: {str(e)}")
    
    def get_result_files(self, task_id: str) -> Dict[str, Any]:
        """
        获取任务的结果文件路径
        
        Args:
            task_id: 任务ID
            
        Returns:
            文件路径字典
        """
        json_path = os.path.join(self.json_dir, f"{task_id}.json")
        
        # 查找标注图片文件
        annotated_files = []
        for filename in os.listdir(self.images_dir):
            if filename.startswith(task_id):
                annotated_files.append(filename)
        
        # 查找原始帧文件
        origin_files = []
        for filename in os.listdir(self.origin_dir):
            if filename.startswith(task_id):
                origin_files.append(filename)
        
        # 按帧号排序
        annotated_files.sort(key=lambda x: int(x.split('frame')[1].split('.')[0]) if 'frame' in x else 0)
        origin_files.sort(key=lambda x: int(x.split('frame')[1].split('.')[0]) if 'frame' in x else 0)
        
        return {
            "json_file": json_path if os.path.exists(json_path) else None,
            "annotated_images": annotated_files,
            "origin_images": origin_files,
            "total_files": len(annotated_files) + len(origin_files) + (1 if os.path.exists(json_path) else 0)
        }
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        清理临时文件
        
        Args:
            max_age_hours: 最大保留时间（小时）
        """
        import time
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for filename in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, filename)
            if os.path.isfile(file_path):
                file_age = current_time - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    try:
                        os.remove(file_path)
                        logger.info(f"清理临时文件: {filename}")
                    except Exception as e:
                        logger.error(f"清理文件失败: {filename}, {str(e)}")

# 全局处理器实例
_handler_instance = None

def get_file_handler(output_dir: str = "output") -> SimpleFileHandler:
    """
    获取文件处理器实例（单例）
    
    Args:
        output_dir: 输出目录
        
    Returns:
        文件处理器实例
    """
    global _handler_instance
    
    if _handler_instance is None:
        _handler_instance = SimpleFileHandler(output_dir)
    
    return _handler_instance 