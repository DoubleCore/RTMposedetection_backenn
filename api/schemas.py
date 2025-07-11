"""
API数据模型定义
"""
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime

# 基础响应模型
class BaseResponse(BaseModel):
    success: bool
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)

# 关键点数据模型
class Keypoint(BaseModel):
    x: float
    y: float
    confidence: float

# 姿态数据模型
class PoseData(BaseModel):
    person_id: int
    keypoints: List[Keypoint]
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float

# 速度数据模型
class SpeedData(BaseModel):
    frame_id: int
    velocity_x: float
    velocity_y: float
    speed_magnitude: float
    acceleration: float

# 角度数据模型
class AngleData(BaseModel):
    joint_name: str
    angle: float
    frame_id: int

# 分析结果模型
class AnalysisResult(BaseModel):
    person_id: int
    frame_id: int
    pose_data: PoseData
    speed_data: Optional[SpeedData] = None
    angle_data: List[AngleData] = []

# 处理任务模型
class ProcessingTask(BaseModel):
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float = 0.0
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

# 文件上传响应
class UploadResponse(BaseResponse):
    task_id: str
    estimated_processing_time: Optional[float] = None

# 分析结果响应
class AnalysisResponse(BaseResponse):
    task_id: str
    results: List[AnalysisResult]
    total_frames: int
    processing_time: float
    output_files: Dict[str, str]  # {"json": "path", "images": "path"}

# 任务状态响应
class TaskStatusResponse(BaseResponse):
    task: ProcessingTask

# 批量处理请求
class BatchProcessRequest(BaseModel):
    file_paths: List[str]
    analysis_type: str = "full"  # "pose_only", "speed_only", "angle_only", "full"

# 批量处理响应
class BatchProcessResponse(BaseResponse):
    task_ids: List[str]
    total_files: int 