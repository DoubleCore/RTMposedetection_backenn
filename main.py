"""
主应用入口 - 跳水姿态分析系统
使用FastAPI和RTMPose进行实时姿态检测
"""

import os
import sys
import uuid
import aiofiles
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.simple_handler import get_file_handler
from utils.exceptions import ProcessingError

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('diving_pose_analysis.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="跳水姿态分析系统",
    description="基于RTMPose的实时姿态检测和分析API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建输出目录
OUTPUT_DIR = "output"
UPLOAD_DIR = "uploads"
TEMP_DIR = "temp"

for dir_path in [OUTPUT_DIR, UPLOAD_DIR, TEMP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 静态文件服务
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
app.mount("/origin", StaticFiles(directory=os.path.join(OUTPUT_DIR, "origin")), name="origin")

# 获取处理器实例
file_handler = get_file_handler(OUTPUT_DIR)

# 任务状态存储
task_status: Dict[str, Dict[str, Any]] = {}

# 数据模型
class UploadResponse(BaseModel):
    task_id: str
    filename: str
    file_size: int
    file_type: str
    message: str

class ProcessingStatus(BaseModel):
    task_id: str
    status: str  # "processing", "completed", "failed"
    progress: float
    message: str
    result: Optional[Dict[str, Any]] = None

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("🚀 启动跳水姿态分析系统...")
    logger.info(f"📍 API文档: http://localhost:8000/docs")
    logger.info(f"📁 输出目录: {OUTPUT_DIR}")
    logger.info("✅ 系统启动完成")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("🔄 正在关闭系统...")
    # 清理临时文件
    file_handler.cleanup_temp_files(max_age_hours=1)
    logger.info("✅ 系统关闭完成")

@app.get("/")
async def root():
    """根路径 - 系统信息"""
    return {
        "system": "跳水姿态分析系统",
        "version": "1.0.0",
        "status": "运行中",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs",
        "endpoints": {
            "upload": "/upload",
            "status": "/status/{task_id}",
            "results": "/results/{task_id}",
            "download": "/download/{task_id}/{file_type}"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "detector_ready": True
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    上传文件并开始处理
    支持图片和视频文件
    """
    try:
        # 检查文件类型
        allowed_types = [
            "image/jpeg", "image/jpg", "image/png", "image/bmp",
            "video/mp4", "video/avi", "video/mov", "video/mkv"
        ]
        
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {file.content_type}。支持的类型: {', '.join(allowed_types)}"
            )
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        
        # 保存上传的文件
        file_extension = Path(file.filename).suffix
        temp_filename = f"{task_id}{file_extension}"
        temp_filepath = os.path.join(TEMP_DIR, temp_filename)
        
        async with aiofiles.open(temp_filepath, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        file_size = len(content)
        
        # 确定文件类型
        file_type = "image" if file.content_type.startswith("image/") else "video"
        
        # 初始化任务状态
        task_status[task_id] = {
            "status": "processing",
            "progress": 0.0,
            "message": "开始处理文件...",
            "filename": file.filename,
            "file_type": file_type,
            "file_size": file_size,
            "created_at": datetime.now().isoformat()
        }
        
        # 添加后台处理任务
        background_tasks.add_task(
            process_file_background,
            task_id,
            temp_filepath,
            file.filename,
            file_type
        )
        
        logger.info(f"📁 文件上传成功: {file.filename} (任务ID: {task_id})")
        
        return UploadResponse(
            task_id=task_id,
            filename=file.filename,
            file_size=file_size,
            file_type=file_type,
            message="文件上传成功，开始处理"
        )
        
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

async def process_file_background(
    task_id: str,
    file_path: str,
    filename: str,
    file_type: str
):
    """
    后台处理文件的任务
    """
    try:
        logger.info(f"🔄 开始处理任务: {task_id}")
        
        # 更新状态
        task_status[task_id].update({
            "status": "processing",
            "progress": 10.0,
            "message": "正在初始化处理器..."
        })
        
        # 根据文件类型调用相应的处理方法
        if file_type == "image":
            result = await file_handler.process_image_file(file_path, task_id, filename)
        else:  # video
            result = await file_handler.process_video_file(file_path, task_id, filename)
        
        # 更新完成状态
        task_status[task_id].update({
            "status": "completed",
            "progress": 100.0,
            "message": "处理完成",
            "result": result,
            "completed_at": datetime.now().isoformat()
        })
        
        logger.info(f"✅ 任务处理完成: {task_id}")
        
    except Exception as e:
        # 更新失败状态
        task_status[task_id].update({
            "status": "failed",
            "progress": 0.0,
            "message": f"处理失败: {str(e)}",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })
        
        logger.error(f"❌ 任务处理失败: {task_id}, 错误: {str(e)}")
    
    finally:
        # 清理临时文件
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"🗑️ 清理临时文件: {file_path}")
        except Exception as e:
            logger.error(f"清理临时文件失败: {str(e)}")

@app.get("/status/{task_id}", response_model=ProcessingStatus)
async def get_task_status(task_id: str):
    """
    获取任务状态
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    status_data = task_status[task_id]
    
    return ProcessingStatus(
        task_id=task_id,
        status=status_data["status"],
        progress=status_data["progress"],
        message=status_data["message"],
        result=status_data.get("result")
    )

@app.get("/results/{task_id}")
async def get_results(task_id: str):
    """
    获取处理结果
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    status_data = task_status[task_id]
    
    if status_data["status"] != "completed":
        raise HTTPException(status_code=400, detail="任务尚未完成")
    
    # 获取结果文件信息
    result_files = file_handler.get_result_files(task_id)
    
    return {
        "task_id": task_id,
        "status": status_data["status"],
        "result": status_data.get("result"),
        "files": result_files,
        "download_links": {
            "json": f"/download/{task_id}/json",
            "images": f"/download/{task_id}/images"
        }
    }

@app.get("/download/{task_id}/json")
async def download_json(task_id: str):
    """
    下载JSON结果文件
    """
    json_path = os.path.join(OUTPUT_DIR, "json", f"{task_id}.json")
    
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="JSON文件不存在")
    
    return FileResponse(
        json_path,
        media_type="application/json",
        filename=f"{task_id}_results.json"
    )

@app.get("/download/{task_id}/images")
async def list_images(task_id: str):
    """
    列出可下载的图片文件
    """
    images_dir = os.path.join(OUTPUT_DIR, "images")
    image_files = []
    
    for filename in os.listdir(images_dir):
        if filename.startswith(task_id):
            image_files.append({
                "filename": filename,
                "download_url": f"/output/images/{filename}"
            })
    
    return {
        "task_id": task_id,
        "image_files": image_files
    }

@app.get("/download/{task_id}/image/{filename}")
async def download_image(task_id: str, filename: str):
    """
    下载特定的图片文件
    """
    if not filename.startswith(task_id):
        raise HTTPException(status_code=400, detail="文件名不匹配任务ID")
    
    image_path = os.path.join(OUTPUT_DIR, "images", filename)
    
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="图片文件不存在")
    
    return FileResponse(
        image_path,
        media_type="image/jpeg",
        filename=filename
    )

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """
    删除任务和相关文件
    """
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    try:
        # 删除结果文件
        json_path = os.path.join(OUTPUT_DIR, "json", f"{task_id}.json")
        if os.path.exists(json_path):
            os.remove(json_path)
        
        # 删除图片文件
        images_dir = os.path.join(OUTPUT_DIR, "images")
        for filename in os.listdir(images_dir):
            if filename.startswith(task_id):
                os.remove(os.path.join(images_dir, filename))
        
        # 删除任务状态
        del task_status[task_id]
        
        return {"message": f"任务 {task_id} 已删除"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除任务失败: {str(e)}")

@app.get("/tasks")
async def list_tasks():
    """
    列出所有任务
    """
    return {
        "tasks": [
            {
                "task_id": task_id,
                "status": data["status"],
                "filename": data.get("filename"),
                "file_type": data.get("file_type"),
                "created_at": data.get("created_at"),
                "progress": data.get("progress", 0.0)
            }
            for task_id, data in task_status.items()
        ]
    }

if __name__ == "__main__":
    print("🚀 启动跳水姿态分析系统...")
    print("📍 API地址: http://localhost:8000")
    print("📖 API文档: http://localhost:8000/docs")
    print("📁 输出目录: ./output/")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 