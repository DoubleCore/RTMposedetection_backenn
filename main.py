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

import cv2
import numpy as np
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
from models.sam import get_sam_processor
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
sam_processor = get_sam_processor(OUTPUT_DIR)  # 添加SAM处理器

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

@app.get("/origin/{task_id}")
async def list_origin_frames(task_id: str):
    """
    列出指定任务的原始帧文件
    """
    origin_task_dir = os.path.join(OUTPUT_DIR, "origin", task_id)
    
    if not os.path.exists(origin_task_dir):
        raise HTTPException(status_code=404, detail="任务原始帧目录不存在")
    
    frame_files = []
    for filename in os.listdir(origin_task_dir):
        if filename.startswith("frame_") and filename.endswith('.jpg'):
            frame_files.append({
                "filename": filename,
                "download_url": f"/origin/{task_id}/{filename}"
            })
    
    # 按帧号排序
    frame_files.sort(key=lambda x: int(x["filename"].split('_')[1].split('.')[0]))
    
    return {
        "task_id": task_id,
        "origin_frames": frame_files,
        "total_frames": len(frame_files)
    }

@app.get("/origin/{task_id}/{filename}")
async def download_origin_frame(task_id: str, filename: str):
    """
    下载指定任务的原始帧文件
    """
    if not filename.startswith("frame_") or not filename.endswith(".jpg"):
        raise HTTPException(status_code=400, detail="无效的文件名格式")
    
    frame_path = os.path.join(OUTPUT_DIR, "origin", task_id, filename)
    
    if not os.path.exists(frame_path):
        raise HTTPException(status_code=404, detail="原始帧文件不存在")
    
    return FileResponse(
        frame_path,
        media_type="image/jpeg",
        filename=filename
    )

# ===== SAM分割相关API端点 =====

@app.post("/sam/process")
async def process_sam_segmentation(
    background_tasks: BackgroundTasks,
    processing_mode: str = "separate_person_pool"
):
    """
    对所有pose检测结果进行人物和游泳池分离处理
    
    Args:
        processing_mode: 处理模式
            - "separate_person_pool": 分离人物和游泳池
            - "highlight_person": 突出人物，模糊背景  
            - "extract_person": 提取人物，透明背景
            - "pool_only": 只保留游泳池，移除人物
    """
    try:
        logger.info(f"🎯 开始人物和游泳池分离处理，模式: {processing_mode}")
        
        # 检查是否有pose数据可以处理
        pose_dir = os.path.join(OUTPUT_DIR, "pose")
        if not os.path.exists(pose_dir) or not os.listdir(pose_dir):
            raise HTTPException(status_code=400, detail="没有找到pose检测结果，请先进行姿态检测")
        
        # 验证处理模式
        valid_modes = ["separate_person_pool", "highlight_person", "extract_person", "pool_only"]
        if processing_mode not in valid_modes:
            raise HTTPException(status_code=400, detail=f"无效的处理模式。支持的模式: {', '.join(valid_modes)}")
        
        # 添加后台处理任务
        background_tasks.add_task(process_sam_background, processing_mode)
        
        return {
            "message": f"人物和游泳池分离处理已开始",
            "processing_mode": processing_mode,
            "status": "processing",
            "mode_description": {
                "separate_person_pool": "分离人物和游泳池为独立图像",
                "highlight_person": "突出人物，模糊游泳池背景", 
                "extract_person": "提取人物，透明背景",
                "pool_only": "只保留游泳池，移除人物"
            }.get(processing_mode, "")
        }
        
    except Exception as e:
        logger.error(f"启动人物和游泳池分离处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动处理失败: {str(e)}")

async def process_sam_background(processing_mode: str):
    """
    后台人物和游泳池分离处理任务
    """
    try:
        logger.info(f"🔄 开始后台人物和游泳池分离处理，模式: {processing_mode}")
        result = sam_processor.process_all_frames(processing_mode)
        logger.info(f"✅ 人物和游泳池分离处理完成: {result}")
    except Exception as e:
        logger.error(f"❌ 后台处理失败: {str(e)}")

@app.get("/sam/status")
async def get_sam_status():
    """
    获取人物和游泳池分离处理状态和统计信息
    """
    try:
        stats = sam_processor.get_processing_stats()
        
        # 检查pose和origin目录的文件数量
        pose_count = len([f for f in os.listdir(os.path.join(OUTPUT_DIR, "pose")) 
                         if f.startswith("frame_") and f.endswith(".jpg")])
        origin_count = len([f for f in os.listdir(os.path.join(OUTPUT_DIR, "origin")) 
                           if f.startswith("frame_") and f.endswith(".jpg")])
        
        return {
            "sam_processing": stats,
            "input_data": {
                "pose_images": pose_count,
                "origin_frames": origin_count
            },
            "ready_for_processing": pose_count > 0,
            "supported_modes": [
                {
                    "mode": "separate_person_pool",
                    "description": "分离人物和游泳池为独立图像"
                },
                {
                    "mode": "highlight_person", 
                    "description": "突出人物，模糊游泳池背景"
                },
                {
                    "mode": "extract_person",
                    "description": "提取人物，透明背景"
                },
                {
                    "mode": "pool_only",
                    "description": "只保留游泳池，移除人物"
                }
            ]
        }
        
    except Exception as e:
        logger.error(f"获取处理状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@app.get("/sam/images")
async def list_sam_images():
    """
    列出所有人物和游泳池分离后的图片
    """
    try:
        sam_dir = os.path.join(OUTPUT_DIR, "sam")
        
        if not os.path.exists(sam_dir):
            return {"sam_images": [], "total": 0}
        
        # 获取不同类型的处理结果
        image_categories = {
            "person_only": [],
            "pool_only": [],
            "combined": [],
            "highlighted_person": [],
            "mask_visualization": [],
            "person_extracted": []
        }
        
        for filename in os.listdir(sam_dir):
            if filename.endswith(('.jpg', '.png')):
                frame_number_match = filename.split('_')[1] if '_' in filename else None
                try:
                    frame_number = int(frame_number_match) if frame_number_match else 0
                except ValueError:
                    frame_number = 0
                
                image_info = {
                    "filename": filename,
                    "download_url": f"/output/sam/{filename}",
                    "frame_number": frame_number
                }
                
                # 根据文件名分类
                for category in image_categories.keys():
                    if category in filename:
                        image_categories[category].append(image_info)
                        break
        
        # 对每个类别按帧号排序
        for category in image_categories:
            image_categories[category].sort(key=lambda x: x["frame_number"])
        
        return {
            "image_categories": image_categories,
            "total_by_category": {k: len(v) for k, v in image_categories.items()},
            "total_images": sum(len(v) for v in image_categories.values())
        }
        
    except Exception as e:
        logger.error(f"列出处理结果失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"列出图片失败: {str(e)}")

@app.get("/sam/comparison/{frame_number}")
async def create_comparison_image(frame_number: int):
    """
    创建指定帧的处理结果对比图像
    """
    try:
        sam_dir = os.path.join(OUTPUT_DIR, "sam")
        
        # 查找该帧的所有处理结果
        frame_files = []
        for filename in os.listdir(sam_dir):
            if filename.startswith(f"frame_{frame_number}_") and filename.endswith(('.jpg', '.png')):
                frame_files.append(filename)
        
        if not frame_files:
            raise HTTPException(status_code=404, detail=f"未找到帧 {frame_number} 的处理结果")
        
        # 创建对比图像
        images_to_combine = []
        labels = []
        
        # 按优先级顺序加载图像
        priority_types = ["original", "person_only", "pool_only", "combined", "mask_visualization"]
        
        for img_type in priority_types:
            matching_file = next((f for f in frame_files if img_type in f), None)
            if matching_file:
                img_path = os.path.join(sam_dir, matching_file)
                img = cv2.imread(img_path)
                if img is not None:
                    images_to_combine.append(img)
                    labels.append(img_type.replace('_', ' ').title())
        
        if len(images_to_combine) < 2:
            raise HTTPException(status_code=404, detail=f"帧 {frame_number} 的处理结果不足，无法创建对比图")
        
        # 调整所有图像到相同尺寸
        target_height = 300
        resized_images = []
        for img in images_to_combine:
            h, w = img.shape[:2]
            target_width = int(w * target_height / h)
            resized = cv2.resize(img, (target_width, target_height))
            resized_images.append(resized)
        
        # 水平拼接图像
        comparison = np.hstack(resized_images)
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        x_offset = 0
        for i, (img, label) in enumerate(zip(resized_images, labels)):
            cv2.putText(comparison, label, (x_offset + 10, 25), font, 0.7, (255, 255, 255), 2)
            x_offset += img.shape[1]
        
        # 保存对比图像
        comparison_filename = f"frame_{frame_number}_comparison.jpg"
        comparison_path = os.path.join(sam_dir, comparison_filename)
        cv2.imwrite(comparison_path, comparison)
        
        return {
            "success": True,
            "frame_number": frame_number,
            "comparison_image": comparison_filename,
            "download_url": f"/output/sam/{comparison_filename}",
            "included_types": labels
        }
        
    except Exception as e:
        logger.error(f"创建对比图像失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建对比图像失败: {str(e)}")

@app.delete("/sam/clear")
async def clear_sam_results():
    """
    清理所有SAM处理结果
    """
    try:
        sam_dir = os.path.join(OUTPUT_DIR, "sam")
        
        if os.path.exists(sam_dir):
            import shutil
            shutil.rmtree(sam_dir)
            os.makedirs(sam_dir, exist_ok=True)
        
        logger.info("🗑️ SAM结果已清理")
        
        return {
            "message": "SAM处理结果已清理",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"清理SAM结果失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清理SAM结果失败: {str(e)}")

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