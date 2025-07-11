"""
主处理管道协调器
负责协调整个处理流程：RTMPose -> SAM -> 追踪 -> 分析
"""
import asyncio
from typing import List, Dict, Any, Callable, Optional
import logging

from core.input_handler import InputHandler
from models.rtmpose.detector import get_rtmpose_detector
from models.sam.segmenter import SAMSegmenter
from models.tracking.tracker import PersonTracker
from analysis.pose_analyzer import PoseAnalyzer
from utils.exceptions import ProcessingError

logger = logging.getLogger(__name__)

class ProcessingPipeline:
    """
    主处理管道类
    """
    
    def __init__(self):
        """
        初始化处理管道
        """
        self.input_handler = InputHandler()
        self.rtmpose_detector = get_rtmpose_detector()
        self.sam_segmenter = SAMSegmenter()
        self.person_tracker = PersonTracker()
        self.pose_analyzer = PoseAnalyzer()
        
        logger.info("处理管道初始化完成")
    
    async def process_file(
        self, 
        file_path: str, 
        analysis_type: str = "full",
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """
        处理单个文件的主方法
        
        Args:
            file_path: 输入文件路径
            analysis_type: 分析类型 ("pose_only", "speed_only", "angle_only", "full")
            progress_callback: 进度回调函数
            
        Returns:
            处理结果字典
        """
        try:
            logger.info(f"开始处理文件: {file_path}")
            
            # 阶段1: 输入处理 (10%)
            frames_data = await self.input_handler.process_input(file_path)
            total_frames = len(frames_data)
            if progress_callback:
                progress_callback(10.0)
            
            # 阶段2: RTMPose姿态检测 (40%)
            pose_results = []
            for i, frame_data in enumerate(frames_data):
                pose_result = await self.rtmpose_detector.detect_pose(frame_data)
                pose_results.append(pose_result)
                
                if progress_callback:
                    progress = 10.0 + (i + 1) / total_frames * 30.0
                    progress_callback(progress)
            
            # 阶段3: SAM分割处理 (60%)
            if analysis_type in ["full"]:
                segmented_results = []
                for i, (frame_data, pose_result) in enumerate(zip(frames_data, pose_results)):
                    segmented_result = await self.sam_segmenter.segment_persons(
                        frame_data, pose_result
                    )
                    segmented_results.append(segmented_result)
                    
                    if progress_callback:
                        progress = 40.0 + (i + 1) / total_frames * 20.0
                        progress_callback(progress)
            else:
                segmented_results = []
            
            # 阶段4: 人物追踪 (75%)
            if analysis_type in ["full"] and segmented_results:
                tracking_results = await self.person_tracker.track_persons(
                    pose_results, segmented_results
                )
                if progress_callback:
                    progress_callback(75.0)
            else:
                tracking_results = pose_results
            
            # 阶段5: 姿态分析 (100%)
            analysis_results = await self.pose_analyzer.analyze_sequence(
                tracking_results, analysis_type
            )
            if progress_callback:
                progress_callback(100.0)
            
            logger.info(f"文件处理完成: {file_path}")
            
            return {
                "total_frames": total_frames,
                "pose_results": pose_results,
                "segmented_results": segmented_results,
                "tracking_results": tracking_results,
                "analysis_results": analysis_results,
                "metadata": {
                    "file_path": file_path,
                    "analysis_type": analysis_type,
                    "processing_stages": self._get_processing_stages(analysis_type)
                }
            }
            
        except Exception as e:
            logger.error(f"处理文件失败: {file_path}, 错误: {str(e)}")
            raise ProcessingError(f"处理失败: {str(e)}")
    
    async def process_batch(
        self, 
        file_paths: List[str], 
        analysis_type: str = "full",
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        批量处理文件
        
        Args:
            file_paths: 文件路径列表
            analysis_type: 分析类型
            progress_callback: 进度回调函数
            
        Returns:
            处理结果列表
        """
        results = []
        total_files = len(file_paths)
        
        for i, file_path in enumerate(file_paths):
            try:
                result = await self.process_file(file_path, analysis_type)
                results.append(result)
                
                if progress_callback:
                    progress = (i + 1) / total_files * 100.0
                    progress_callback(progress)
                    
            except Exception as e:
                logger.error(f"批量处理中文件失败: {file_path}, 错误: {str(e)}")
                results.append({
                    "error": str(e),
                    "file_path": file_path
                })
        
        return results
    
    def _get_processing_stages(self, analysis_type: str) -> List[str]:
        """
        根据分析类型返回处理阶段
        """
        base_stages = ["input_processing", "pose_detection"]
        
        if analysis_type == "full":
            return base_stages + ["segmentation", "tracking", "analysis"]
        elif analysis_type == "pose_only":
            return base_stages
        else:
            return base_stages + ["analysis"]
    
    async def cleanup(self):
        """
        清理资源
        """
        try:
            await self.rtmpose_detector.cleanup()
            await self.sam_segmenter.cleanup()
            await self.person_tracker.cleanup()
            logger.info("处理管道资源清理完成")
        except Exception as e:
            logger.error(f"资源清理失败: {str(e)}") 