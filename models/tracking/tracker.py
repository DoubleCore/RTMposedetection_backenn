"""
人物追踪器
"""
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from config.model_configs import TRACKING_CONFIG
from utils.exceptions import ModelError
from utils.data_structures import TrackingResult, PoseDetectionResult, SegmentationResult

logger = logging.getLogger(__name__)

class PersonTracker:
    """
    人物追踪器
    """
    
    def __init__(self):
        """
        初始化人物追踪器
        """
        self.config = TRACKING_CONFIG
        self.active_tracks = {}
        self.next_person_id = 1
        self.max_age = self.config.get("max_age", 30)
        self.min_hits = self.config.get("n_init", 3)
        
        logger.info("人物追踪器初始化完成")
    
    async def track_persons(
        self, 
        pose_results_sequence: List[Dict[str, Any]],
        segmentation_results_sequence: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        对人物序列进行追踪
        
        Args:
            pose_results_sequence: 姿态检测结果序列
            segmentation_results_sequence: 分割结果序列
            
        Returns:
            追踪结果序列
        """
        try:
            tracking_results = []
            
            for frame_idx, (pose_frame, seg_frame) in enumerate(
                zip(pose_results_sequence, segmentation_results_sequence)
            ):
                frame_tracking = await self._track_single_frame(
                    pose_frame, seg_frame, frame_idx
                )
                tracking_results.append(frame_tracking)
            
            logger.info(f"追踪完成，共处理 {len(tracking_results)} 帧")
            return tracking_results
            
        except Exception as e:
            logger.error(f"人物追踪失败: {str(e)}")
            raise ModelError(f"人物追踪失败: {str(e)}")
    
    async def _track_single_frame(
        self,
        pose_frame: Dict[str, Any],
        seg_frame: Optional[Dict[str, Any]],
        frame_idx: int
    ) -> Dict[str, Any]:
        """
        追踪单帧中的人物
        
        Args:
            pose_frame: 姿态检测结果
            seg_frame: 分割结果
            frame_idx: 帧索引
            
        Returns:
            单帧追踪结果
        """
        try:
            pose_results = pose_frame.get("pose_results", [])
            seg_results = seg_frame.get("segmentation_results", []) if seg_frame else []
            
            # 匹配姿态和分割结果
            matched_detections = self._match_pose_and_segmentation(
                pose_results, seg_results
            )
            
            # 更新追踪状态
            tracking_results = self._update_tracking(matched_detections, frame_idx)
            
            return {
                "frame_id": pose_frame["frame_id"],
                "timestamp": pose_frame["timestamp"],
                "tracking_results": tracking_results,
                "metadata": {
                    "active_tracks": len(self.active_tracks),
                    "new_detections": len(matched_detections)
                }
            }
            
        except Exception as e:
            logger.error(f"单帧追踪失败: {str(e)}")
            raise
    
    def _match_pose_and_segmentation(
        self,
        pose_results: List[PoseDetectionResult],
        seg_results: List[SegmentationResult]
    ) -> List[Dict[str, Any]]:
        """
        匹配姿态检测和分割结果
        
        Args:
            pose_results: 姿态检测结果
            seg_results: 分割结果
            
        Returns:
            匹配后的检测结果
        """
        matched_detections = []
        
        for pose in pose_results:
            best_match = None
            best_iou = 0.0
            
            # 寻找最佳匹配的分割结果
            for seg in seg_results:
                iou = self._calculate_bbox_iou(pose.bbox, seg.bbox)
                if iou > best_iou and iou > 0.3:  # IoU阈值
                    best_iou = iou
                    best_match = seg
            
            matched_detections.append({
                "pose_data": pose,
                "segmentation_data": best_match,
                "detection_confidence": pose.confidence
            })
        
        return matched_detections
    
    def _calculate_bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        计算边界框IoU
        
        Args:
            bbox1: 边界框1 [x1, y1, x2, y2]
            bbox2: 边界框2 [x1, y1, x2, y2]
            
        Returns:
            IoU值
        """
        try:
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2
            
            # 计算交集区域
            x1_inter = max(x1_1, x1_2)
            y1_inter = max(y1_1, y1_2)
            x2_inter = min(x2_1, x2_2)
            y2_inter = min(y2_1, y2_2)
            
            if x2_inter <= x1_inter or y2_inter <= y1_inter:
                return 0.0
            
            # 交集面积
            inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            
            # 各自面积
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
            
            # 并集面积
            union_area = area1 + area2 - inter_area
            
            if union_area <= 0:
                return 0.0
            
            return inter_area / union_area
            
        except Exception:
            return 0.0
    
    def _update_tracking(
        self, 
        detections: List[Dict[str, Any]], 
        frame_idx: int
    ) -> List[TrackingResult]:
        """
        更新追踪状态
        
        Args:
            detections: 当前帧检测结果
            frame_idx: 帧索引
            
        Returns:
            追踪结果列表
        """
        # 简化的追踪算法实现
        tracking_results = []
        
        # 为新检测分配ID或匹配现有轨迹
        for detection in detections:
            pose_data = detection["pose_data"]
            seg_data = detection["segmentation_data"]
            
            # 寻找最佳匹配的现有轨迹
            best_track_id = self._find_best_match(pose_data, frame_idx)
            
            if best_track_id is None:
                # 创建新轨迹
                person_id = self.next_person_id
                self.next_person_id += 1
                
                self.active_tracks[person_id] = {
                    "last_seen": frame_idx,
                    "hit_count": 1,
                    "last_bbox": pose_data.bbox,
                    "confirmed": False
                }
            else:
                # 更新现有轨迹
                person_id = best_track_id
                self.active_tracks[person_id]["last_seen"] = frame_idx
                self.active_tracks[person_id]["hit_count"] += 1
                self.active_tracks[person_id]["last_bbox"] = pose_data.bbox
                
                # 确认轨迹
                if (self.active_tracks[person_id]["hit_count"] >= self.min_hits and 
                    not self.active_tracks[person_id]["confirmed"]):
                    self.active_tracks[person_id]["confirmed"] = True
            
            # 只返回已确认的轨迹
            if self.active_tracks[person_id]["confirmed"]:
                tracking_result = TrackingResult(
                    person_id=person_id,
                    bbox=pose_data.bbox,
                    confidence=detection["detection_confidence"],
                    pose_data=pose_data,
                    segmentation_data=seg_data
                )
                tracking_results.append(tracking_result)
        
        # 清理过期轨迹
        self._cleanup_tracks(frame_idx)
        
        return tracking_results
    
    def _find_best_match(
        self, 
        pose_data: PoseDetectionResult, 
        frame_idx: int
    ) -> Optional[int]:
        """
        为新检测寻找最佳匹配的现有轨迹
        
        Args:
            pose_data: 姿态检测数据
            frame_idx: 当前帧索引
            
        Returns:
            匹配的轨迹ID或None
        """
        best_track_id = None
        best_distance = float('inf')
        
        for track_id, track_info in self.active_tracks.items():
            # 检查轨迹是否太旧
            if frame_idx - track_info["last_seen"] > 5:
                continue
            
            # 计算距离（这里使用简单的中心点距离）
            distance = self._calculate_center_distance(
                pose_data.bbox, track_info["last_bbox"]
            )
            
            if distance < best_distance and distance < 100:  # 距离阈值
                best_distance = distance
                best_track_id = track_id
        
        return best_track_id
    
    def _calculate_center_distance(
        self, 
        bbox1: List[float], 
        bbox2: List[float]
    ) -> float:
        """
        计算两个边界框中心点的距离
        
        Args:
            bbox1: 边界框1
            bbox2: 边界框2
            
        Returns:
            中心点距离
        """
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2
        
        distance = np.sqrt(
            (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2
        )
        
        return distance
    
    def _cleanup_tracks(self, frame_idx: int):
        """
        清理过期的轨迹
        
        Args:
            frame_idx: 当前帧索引
        """
        expired_tracks = []
        
        for track_id, track_info in self.active_tracks.items():
            if frame_idx - track_info["last_seen"] > self.max_age:
                expired_tracks.append(track_id)
        
        for track_id in expired_tracks:
            del self.active_tracks[track_id]
            
        if expired_tracks:
            logger.info(f"清理了 {len(expired_tracks)} 个过期轨迹")
    
    async def cleanup(self):
        """
        清理资源
        """
        try:
            self.active_tracks.clear()
            logger.info("人物追踪器资源清理完成")
        except Exception as e:
            logger.error(f"追踪器资源清理失败: {str(e)}") 