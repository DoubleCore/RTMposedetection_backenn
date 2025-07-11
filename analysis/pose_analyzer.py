"""
姿态分析主逻辑
"""
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from analysis.speed_calculator import SpeedCalculator
from analysis.angle_calculator import AngleCalculator
from analysis.metrics_exporter import MetricsExporter
from utils.data_structures import TrackingResult, AnalysisMetrics
from utils.exceptions import ProcessingError

logger = logging.getLogger(__name__)

class PoseAnalyzer:
    """
    姿态分析器
    """
    
    def __init__(self):
        """
        初始化姿态分析器
        """
        self.speed_calculator = SpeedCalculator()
        self.angle_calculator = AngleCalculator()
        self.metrics_exporter = MetricsExporter()
        
        logger.info("姿态分析器初始化完成")
    
    async def analyze_sequence(
        self,
        tracking_results: List[Dict[str, Any]],
        analysis_type: str = "full"
    ) -> Dict[str, Any]:
        """
        分析追踪结果序列
        
        Args:
            tracking_results: 追踪结果序列
            analysis_type: 分析类型
            
        Returns:
            分析结果
        """
        try:
            logger.info(f"开始分析序列，类型: {analysis_type}")
            
            # 按人物ID组织数据
            person_sequences = self._organize_by_person(tracking_results)
            
            # 分析每个人物
            person_analyses = {}
            for person_id, person_sequence in person_sequences.items():
                analysis = await self._analyze_person_sequence(
                    person_id, person_sequence, analysis_type
                )
                person_analyses[person_id] = analysis
            
            # 生成综合分析结果
            comprehensive_analysis = self._generate_comprehensive_analysis(
                person_analyses, tracking_results
            )
            
            logger.info(f"分析完成，共分析 {len(person_analyses)} 个人物")
            
            return {
                "analysis_type": analysis_type,
                "person_analyses": person_analyses,
                "comprehensive_analysis": comprehensive_analysis,
                "metadata": {
                    "total_persons": len(person_analyses),
                    "total_frames": len(tracking_results),
                    "analysis_metrics": self._calculate_overall_metrics(person_analyses)
                }
            }
            
        except Exception as e:
            logger.error(f"序列分析失败: {str(e)}")
            raise ProcessingError(f"序列分析失败: {str(e)}")
    
    def _organize_by_person(
        self, 
        tracking_results: List[Dict[str, Any]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        按人物ID组织追踪数据
        
        Args:
            tracking_results: 追踪结果序列
            
        Returns:
            按人物ID组织的数据
        """
        person_sequences = {}
        
        for frame_result in tracking_results:
            frame_id = frame_result["frame_id"]
            timestamp = frame_result["timestamp"]
            
            for track_result in frame_result.get("tracking_results", []):
                person_id = track_result.person_id
                
                if person_id not in person_sequences:
                    person_sequences[person_id] = []
                
                person_sequences[person_id].append({
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "tracking_data": track_result
                })
        
        return person_sequences
    
    async def _analyze_person_sequence(
        self,
        person_id: int,
        person_sequence: List[Dict[str, Any]],
        analysis_type: str
    ) -> Dict[str, Any]:
        """
        分析单个人物的序列
        
        Args:
            person_id: 人物ID
            person_sequence: 人物序列数据
            analysis_type: 分析类型
            
        Returns:
            人物分析结果
        """
        try:
            analysis_results = []
            
            # 速度分析
            if analysis_type in ["full", "speed_only"]:
                speed_results = await self.speed_calculator.calculate_speed_sequence(
                    person_sequence
                )
            else:
                speed_results = []
            
            # 角度分析
            if analysis_type in ["full", "angle_only"]:
                angle_results = await self.angle_calculator.calculate_angle_sequence(
                    person_sequence
                )
            else:
                angle_results = []
            
            # 组合分析结果
            for i, frame_data in enumerate(person_sequence):
                frame_id = frame_data["frame_id"]
                
                # 获取对应的速度和角度数据
                speed_data = speed_results[i] if i < len(speed_results) else None
                angle_data = [a for a in angle_results if a.frame_id == frame_id]
                
                # 创建分析指标
                analysis_metric = AnalysisMetrics(
                    person_id=person_id,
                    frame_id=frame_id,
                    speed_metrics=speed_data,
                    angle_metrics=angle_data
                )
                
                analysis_results.append(analysis_metric)
            
            # 计算统计信息
            stats = self._calculate_person_statistics(analysis_results)
            
            return {
                "person_id": person_id,
                "sequence_length": len(person_sequence),
                "analysis_results": analysis_results,
                "statistics": stats,
                "key_moments": self._identify_key_moments(analysis_results)
            }
            
        except Exception as e:
            logger.error(f"人物 {person_id} 分析失败: {str(e)}")
            raise
    
    def _calculate_person_statistics(
        self, 
        analysis_results: List[AnalysisMetrics]
    ) -> Dict[str, Any]:
        """
        计算人物统计信息
        
        Args:
            analysis_results: 分析结果列表
            
        Returns:
            统计信息
        """
        stats = {}
        
        # 速度统计
        speeds = []
        accelerations = []
        
        for result in analysis_results:
            if result.speed_metrics:
                speeds.append(result.speed_metrics.speed_magnitude)
                accelerations.append(result.speed_metrics.acceleration)
        
        if speeds:
            stats["speed"] = {
                "max_speed": max(speeds),
                "min_speed": min(speeds),
                "avg_speed": np.mean(speeds),
                "speed_std": np.std(speeds)
            }
        
        if accelerations:
            stats["acceleration"] = {
                "max_acceleration": max(accelerations),
                "min_acceleration": min(accelerations),
                "avg_acceleration": np.mean(accelerations)
            }
        
        # 角度统计
        angle_stats = {}
        for result in analysis_results:
            for angle_metric in result.angle_metrics:
                joint_name = angle_metric.joint_name
                angle = angle_metric.angle
                
                if joint_name not in angle_stats:
                    angle_stats[joint_name] = []
                angle_stats[joint_name].append(angle)
        
        for joint_name, angles in angle_stats.items():
            stats[f"angle_{joint_name}"] = {
                "max_angle": max(angles),
                "min_angle": min(angles),
                "avg_angle": np.mean(angles),
                "angle_range": max(angles) - min(angles)
            }
        
        return stats
    
    def _identify_key_moments(
        self, 
        analysis_results: List[AnalysisMetrics]
    ) -> List[Dict[str, Any]]:
        """
        识别关键时刻
        
        Args:
            analysis_results: 分析结果
            
        Returns:
            关键时刻列表
        """
        key_moments = []
        
        # 最大速度时刻
        max_speed = 0
        max_speed_frame = None
        
        # 最大加速度时刻
        max_acceleration = 0
        max_acceleration_frame = None
        
        for result in analysis_results:
            if result.speed_metrics:
                speed = result.speed_metrics.speed_magnitude
                acceleration = abs(result.speed_metrics.acceleration)
                
                if speed > max_speed:
                    max_speed = speed
                    max_speed_frame = result.frame_id
                
                if acceleration > max_acceleration:
                    max_acceleration = acceleration
                    max_acceleration_frame = result.frame_id
        
        if max_speed_frame is not None:
            key_moments.append({
                "type": "max_speed",
                "frame_id": max_speed_frame,
                "value": max_speed,
                "description": f"最大速度: {max_speed:.2f}"
            })
        
        if max_acceleration_frame is not None:
            key_moments.append({
                "type": "max_acceleration",
                "frame_id": max_acceleration_frame,
                "value": max_acceleration,
                "description": f"最大加速度: {max_acceleration:.2f}"
            })
        
        # 可以添加更多关键时刻识别逻辑
        # 如：入水时刻、起跳时刻等
        
        return key_moments
    
    def _generate_comprehensive_analysis(
        self,
        person_analyses: Dict[int, Dict[str, Any]],
        tracking_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        生成综合分析
        
        Args:
            person_analyses: 各人物分析结果
            tracking_results: 原始追踪结果
            
        Returns:
            综合分析结果
        """
        comprehensive = {
            "scene_analysis": self._analyze_scene(tracking_results),
            "comparative_analysis": self._compare_persons(person_analyses),
            "temporal_analysis": self._analyze_temporal_patterns(person_analyses)
        }
        
        return comprehensive
    
    def _analyze_scene(self, tracking_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        场景分析
        
        Args:
            tracking_results: 追踪结果
            
        Returns:
            场景分析结果
        """
        # 分析人数变化、活动区域等
        person_counts = []
        active_areas = []
        
        for frame_result in tracking_results:
            person_count = len(frame_result.get("tracking_results", []))
            person_counts.append(person_count)
            
            # 计算活动区域（所有人物边界框的并集）
            # 这里简化实现
            
        return {
            "max_persons": max(person_counts) if person_counts else 0,
            "avg_persons": np.mean(person_counts) if person_counts else 0,
            "person_count_variation": np.std(person_counts) if person_counts else 0
        }
    
    def _compare_persons(
        self, 
        person_analyses: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        人物对比分析
        
        Args:
            person_analyses: 各人物分析结果
            
        Returns:
            对比分析结果
        """
        if len(person_analyses) < 2:
            return {"message": "需要至少2个人物进行对比分析"}
        
        comparison = {}
        
        # 速度对比
        max_speeds = {}
        for person_id, analysis in person_analyses.items():
            stats = analysis.get("statistics", {})
            if "speed" in stats:
                max_speeds[person_id] = stats["speed"]["max_speed"]
        
        if max_speeds:
            fastest_person = max(max_speeds, key=max_speeds.get)
            comparison["fastest_person"] = {
                "person_id": fastest_person,
                "max_speed": max_speeds[fastest_person]
            }
        
        return comparison
    
    def _analyze_temporal_patterns(
        self, 
        person_analyses: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        时序模式分析
        
        Args:
            person_analyses: 各人物分析结果
            
        Returns:
            时序分析结果
        """
        # 分析时序模式，如周期性运动、同步性等
        temporal_patterns = {
            "synchronization": self._analyze_synchronization(person_analyses),
            "periodic_patterns": self._detect_periodic_patterns(person_analyses)
        }
        
        return temporal_patterns
    
    def _analyze_synchronization(
        self, 
        person_analyses: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        分析人物动作同步性
        """
        # 简化实现
        return {"synchronization_score": 0.5}
    
    def _detect_periodic_patterns(
        self, 
        person_analyses: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        检测周期性模式
        """
        # 简化实现
        return {"periodic_score": 0.3}
    
    def _calculate_overall_metrics(
        self, 
        person_analyses: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        计算整体指标
        
        Args:
            person_analyses: 各人物分析结果
            
        Returns:
            整体指标
        """
        return {
            "analysis_completeness": len(person_analyses) / max(1, len(person_analyses)),
            "data_quality_score": 0.8,  # 简化实现
            "processing_confidence": 0.9  # 简化实现
        } 