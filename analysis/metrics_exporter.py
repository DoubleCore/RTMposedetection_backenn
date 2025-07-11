"""
数据导出模块
"""
import json
import csv
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

from utils.data_structures import AnalysisMetrics, SpeedMetrics, AngleMetrics
from config.settings import settings

logger = logging.getLogger(__name__)

class MetricsExporter:
    """
    指标导出器
    """
    
    def __init__(self):
        """
        初始化指标导出器
        """
        self.output_dir = Path(settings.JSON_OUTPUT_DIR)
        
    def export_to_json(
        self, 
        analysis_results: Dict[str, Any], 
        output_path: str
    ) -> bool:
        """
        导出分析结果为JSON格式
        
        Args:
            analysis_results: 分析结果
            output_path: 输出路径
            
        Returns:
            是否成功
        """
        try:
            # 转换数据为可序列化格式
            serializable_data = self._make_serializable(analysis_results)
            
            # 添加导出元数据
            export_data = {
                "export_metadata": {
                    "export_time": datetime.now().isoformat(),
                    "version": "1.0.0",
                    "format": "diving_pose_analysis_json"
                },
                "analysis_results": serializable_data
            }
            
            # 写入JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"分析结果已导出到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"JSON导出失败: {str(e)}")
            return False
    
    def export_to_csv(
        self, 
        person_analyses: Dict[int, Dict[str, Any]], 
        output_dir: str
    ) -> bool:
        """
        导出分析结果为CSV格式
        
        Args:
            person_analyses: 人物分析结果
            output_dir: 输出目录
            
        Returns:
            是否成功
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for person_id, analysis in person_analyses.items():
                # 导出速度数据
                self._export_speed_csv(
                    person_id, 
                    analysis.get("analysis_results", []),
                    output_path / f"person_{person_id}_speed.csv"
                )
                
                # 导出角度数据
                self._export_angle_csv(
                    person_id,
                    analysis.get("analysis_results", []),
                    output_path / f"person_{person_id}_angles.csv"
                )
                
                # 导出统计摘要
                self._export_summary_csv(
                    person_id,
                    analysis.get("statistics", {}),
                    output_path / f"person_{person_id}_summary.csv"
                )
            
            logger.info(f"CSV文件已导出到: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"CSV导出失败: {str(e)}")
            return False
    
    def _export_speed_csv(
        self, 
        person_id: int, 
        analysis_results: List[AnalysisMetrics],
        output_path: Path
    ):
        """
        导出速度数据为CSV
        """
        speed_data = []
        
        for result in analysis_results:
            if result.speed_metrics:
                speed_data.append({
                    "person_id": person_id,
                    "frame_id": result.frame_id,
                    "velocity_x": result.speed_metrics.velocity_x,
                    "velocity_y": result.speed_metrics.velocity_y,
                    "speed_magnitude": result.speed_metrics.speed_magnitude,
                    "acceleration": result.speed_metrics.acceleration
                })
        
        if speed_data:
            df = pd.DataFrame(speed_data)
            df.to_csv(output_path, index=False)
    
    def _export_angle_csv(
        self, 
        person_id: int, 
        analysis_results: List[AnalysisMetrics],
        output_path: Path
    ):
        """
        导出角度数据为CSV
        """
        angle_data = []
        
        for result in analysis_results:
            for angle_metric in result.angle_metrics:
                angle_data.append({
                    "person_id": person_id,
                    "frame_id": result.frame_id,
                    "joint_name": angle_metric.joint_name,
                    "angle": angle_metric.angle
                })
        
        if angle_data:
            df = pd.DataFrame(angle_data)
            df.to_csv(output_path, index=False)
    
    def _export_summary_csv(
        self, 
        person_id: int, 
        statistics: Dict[str, Any],
        output_path: Path
    ):
        """
        导出统计摘要为CSV
        """
        summary_data = []
        
        for stat_name, stat_value in statistics.items():
            if isinstance(stat_value, dict):
                for sub_name, sub_value in stat_value.items():
                    summary_data.append({
                        "person_id": person_id,
                        "metric_category": stat_name,
                        "metric_name": sub_name,
                        "value": sub_value
                    })
            else:
                summary_data.append({
                    "person_id": person_id,
                    "metric_category": "general",
                    "metric_name": stat_name,
                    "value": stat_value
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(output_path, index=False)
    
    def export_key_moments(
        self, 
        person_analyses: Dict[int, Dict[str, Any]], 
        output_path: str
    ) -> bool:
        """
        导出关键时刻数据
        
        Args:
            person_analyses: 人物分析结果
            output_path: 输出路径
            
        Returns:
            是否成功
        """
        try:
            key_moments_data = []
            
            for person_id, analysis in person_analyses.items():
                key_moments = analysis.get("key_moments", [])
                for moment in key_moments:
                    key_moments_data.append({
                        "person_id": person_id,
                        "type": moment.get("type"),
                        "frame_id": moment.get("frame_id"),
                        "value": moment.get("value"),
                        "description": moment.get("description")
                    })
            
            # 导出为CSV
            if key_moments_data:
                df = pd.DataFrame(key_moments_data)
                df.to_csv(output_path, index=False)
                logger.info(f"关键时刻数据已导出到: {output_path}")
                return True
            else:
                logger.warning("没有关键时刻数据可导出")
                return False
                
        except Exception as e:
            logger.error(f"关键时刻导出失败: {str(e)}")
            return False
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        将对象转换为可序列化格式
        
        Args:
            obj: 输入对象
            
        Returns:
            可序列化的对象
        """
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # 处理数据类对象
            return self._make_serializable(obj.__dict__)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, datetime):
            return obj.isoformat()
        else:
            # 其他类型转为字符串
            return str(obj)
    
    def generate_report(
        self, 
        analysis_results: Dict[str, Any], 
        output_path: str
    ) -> bool:
        """
        生成详细分析报告
        
        Args:
            analysis_results: 分析结果
            output_path: 输出路径
            
        Returns:
            是否成功
        """
        try:
            report = self._create_text_report(analysis_results)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"分析报告已生成: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"报告生成失败: {str(e)}")
            return False
    
    def _create_text_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        创建文本格式的分析报告
        
        Args:
            analysis_results: 分析结果
            
        Returns:
            报告文本
        """
        report_lines = []
        
        # 报告头部
        report_lines.append("=" * 60)
        report_lines.append("跳水运动员姿态分析报告")
        report_lines.append("=" * 60)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 基本信息
        metadata = analysis_results.get("metadata", {})
        report_lines.append("基本信息:")
        report_lines.append(f"  总人物数: {metadata.get('total_persons', 0)}")
        report_lines.append(f"  总帧数: {metadata.get('total_frames', 0)}")
        report_lines.append(f"  分析类型: {analysis_results.get('analysis_type', 'unknown')}")
        report_lines.append("")
        
        # 各人物详细分析
        person_analyses = analysis_results.get("person_analyses", {})
        for person_id, analysis in person_analyses.items():
            report_lines.append(f"人物 {person_id} 分析结果:")
            report_lines.append("-" * 40)
            
            # 序列信息
            report_lines.append(f"  序列长度: {analysis.get('sequence_length', 0)} 帧")
            
            # 统计信息
            stats = analysis.get("statistics", {})
            if "speed" in stats:
                speed_stats = stats["speed"]
                report_lines.append("  速度统计:")
                report_lines.append(f"    最大速度: {speed_stats.get('max_speed', 0):.2f}")
                report_lines.append(f"    平均速度: {speed_stats.get('avg_speed', 0):.2f}")
                report_lines.append(f"    最小速度: {speed_stats.get('min_speed', 0):.2f}")
            
            # 关键时刻
            key_moments = analysis.get("key_moments", [])
            if key_moments:
                report_lines.append("  关键时刻:")
                for moment in key_moments:
                    report_lines.append(f"    {moment.get('description', '')}")
            
            report_lines.append("")
        
        # 综合分析
        comprehensive = analysis_results.get("comprehensive_analysis", {})
        if comprehensive:
            report_lines.append("综合分析:")
            report_lines.append("-" * 40)
            
            scene_analysis = comprehensive.get("scene_analysis", {})
            if scene_analysis:
                report_lines.append(f"  最大人物数: {scene_analysis.get('max_persons', 0)}")
                report_lines.append(f"  平均人物数: {scene_analysis.get('avg_persons', 0):.1f}")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        report_lines.append("报告结束")
        
        return "\n".join(report_lines) 