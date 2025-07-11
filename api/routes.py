"""
API路由定义
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from typing import List
import uuid
import os
from datetime import datetime

from api.schemas import (
    UploadResponse, AnalysisResponse, TaskStatusResponse, 
    BatchProcessRequest, BatchProcessResponse
)
from core.pipeline import ProcessingPipeline
from utils.file_manager import FileManager
from utils.exceptions import ProcessingError, ValidationError
from config.settings import settings

router = APIRouter()

# 全局处理管道实例
pipeline = ProcessingPipeline()
file_manager = FileManager()

# 任务状态存储（生产环境建议使用Redis）
task_status = {}

@router.post("/analyze", response_model=UploadResponse)
async def analyze_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    analysis_type: str = "full"
):
    """
    单文件分析接口
    """
    try:
        # 打印文件接收日志
        if file.content_type and file.content_type.startswith('video/'):
            print("video attach")
            print(f"📹 视频文件接收成功: {file.filename} ({file.content_type})")
        elif file.content_type and file.content_type.startswith('image/'):
            print(f"📷 图片文件接收成功: {file.filename} ({file.content_type})")
        else:
            print(f"📄 文件接收成功: {file.filename} ({file.content_type})")
        
        # 验证文件
        if not file_manager.validate_file(file):
            raise HTTPException(status_code=400, detail="不支持的文件格式")
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 保存上传文件
        file_path = await file_manager.save_upload_file(file, task_id)
        
        # 创建任务记录
        task_status[task_id] = {
            "status": "pending",
            "progress": 0.0,
            "created_at": datetime.now(),
            "file_path": file_path,
            "analysis_type": analysis_type
        }
        
        # 添加后台处理任务
        background_tasks.add_task(process_file_task, task_id, file_path, analysis_type)
        
        return UploadResponse(
            success=True,
            message="文件上传成功，开始处理",
            task_id=task_id,
            estimated_processing_time=file_manager.estimate_processing_time(file_path)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@router.post("/batch_analyze", response_model=BatchProcessResponse)
async def batch_analyze(
    background_tasks: BackgroundTasks,
    request: BatchProcessRequest
):
    """
    批量分析接口
    """
    try:
        task_ids = []
        
        for file_path in request.file_paths:
            if not os.path.exists(file_path):
                continue
                
            task_id = str(uuid.uuid4())
            task_ids.append(task_id)
            
            # 创建任务记录
            task_status[task_id] = {
                "status": "pending",
                "progress": 0.0,
                "created_at": datetime.now(),
                "file_path": file_path,
                "analysis_type": request.analysis_type
            }
            
            # 添加后台处理任务
            background_tasks.add_task(process_file_task, task_id, file_path, request.analysis_type)
        
        return BatchProcessResponse(
            success=True,
            message=f"批量处理任务已创建",
            task_ids=task_ids,
            total_files=len(task_ids)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量处理失败: {str(e)}")

@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    查询任务状态
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = task_status[task_id]
    return TaskStatusResponse(
        success=True,
        message="状态查询成功",
        task={
            "task_id": task_id,
            "status": task["status"],
            "progress": task["progress"],
            "created_at": task["created_at"],
            "completed_at": task.get("completed_at")
        }
    )

@router.get("/download/{task_id}")
async def download_results(task_id: str, file_type: str = "json"):
    """
    下载处理结果
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = task_status[task_id]
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="任务未完成")
    
    try:
        if file_type == "json":
            file_path = os.path.join(settings.JSON_OUTPUT_DIR, f"{task_id}.json")
        elif file_type == "images":
            file_path = os.path.join(settings.IMAGE_OUTPUT_DIR, f"{task_id}.zip")
        else:
            raise HTTPException(status_code=400, detail="不支持的文件类型")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="结果文件不存在")
        
        return FileResponse(
            path=file_path,
            filename=f"{task_id}_{file_type}",
            media_type='application/octet-stream'
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下载失败: {str(e)}")

async def process_file_task(task_id: str, file_path: str, analysis_type: str):
    """
    后台文件处理任务
    """
    try:
        # 更新任务状态
        task_status[task_id]["status"] = "processing"
        
        # 执行处理管道
        results = await pipeline.process_file(
            file_path, 
            analysis_type,
            progress_callback=lambda p: update_task_progress(task_id, p)
        )
        
        # 保存结果
        await file_manager.save_results(task_id, results)
        
        # 更新任务状态
        task_status[task_id].update({
            "status": "completed",
            "progress": 100.0,
            "completed_at": datetime.now(),
            "results": results
        })
        
    except Exception as e:
        task_status[task_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now()
        })

def update_task_progress(task_id: str, progress: float):
    """
    更新任务进度
    """
    if task_id in task_status:
        task_status[task_id]["progress"] = progress 