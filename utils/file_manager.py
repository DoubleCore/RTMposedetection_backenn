"""
文件管理工具
"""
import os
import json
import zipfile
import shutil
from typing import Dict, Any, Optional
from pathlib import Path
import aiofiles
import logging
from datetime import datetime

from config.settings import settings
from utils.exceptions import FileManagerError

logger = logging.getLogger(__name__)

class FileManager:
    """文件管理器"""
    
    def __init__(self):
        """初始化文件管理器"""
        self.supported_formats = settings.ALLOWED_EXTENSIONS
        self.max_file_size = settings.MAX_FILE_SIZE
        
    def validate_file(self, file) -> bool:
        """
        验证上传文件
        
        Args:
            file: 上传的文件对象
            
        Returns:
            是否为有效文件
        """
        try:
            # 检查文件扩展名
            file_extension = Path(file.filename).suffix.lower()
            if file_extension not in self.supported_formats:
                return False
            
            # 检查文件大小（如果可以获取）
            if hasattr(file, 'size') and file.size > self.max_file_size:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"文件验证失败: {str(e)}")
            return False
    
    async def save_upload_file(self, file, task_id: str) -> str:
        """
        保存上传的文件
        
        Args:
            file: 上传的文件对象
            task_id: 任务ID
            
        Returns:
            保存的文件路径
        """
        try:
            # 创建文件名
            file_extension = Path(file.filename).suffix.lower()
            filename = f"{task_id}_{file.filename}"
            file_path = os.path.join(settings.TEMP_DIR, filename)
            
            # 异步保存文件
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            logger.info(f"文件已保存: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"保存文件失败: {str(e)}")
            raise FileManagerError(f"保存文件失败: {str(e)}")
    
    async def save_results(self, task_id: str, results: Dict[str, Any]):
        """
        保存分析结果
        
        Args:
            task_id: 任务ID
            results: 分析结果
        """
        try:
            # 保存JSON结果
            json_path = os.path.join(settings.JSON_OUTPUT_DIR, f"{task_id}.json")
            await self._save_json_results(json_path, results)
            
            # 如果有图像结果，打包保存
            if 'analysis_results' in results and results['analysis_results']:
                await self._save_image_results(task_id, results)
            
            logger.info(f"结果已保存: 任务 {task_id}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")
            raise FileManagerError(f"保存结果失败: {str(e)}")
    
    async def _save_json_results(self, file_path: str, results: Dict[str, Any]):
        """
        保存JSON格式的结果
        
        Args:
            file_path: 文件路径
            results: 结果数据
        """
        try:
            # 准备JSON数据（移除不可序列化的对象）
            json_data = self._prepare_json_data(results)
            
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(json_data, ensure_ascii=False, indent=2))
                
        except Exception as e:
            logger.error(f"保存JSON结果失败: {str(e)}")
            raise
    
    async def _save_image_results(self, task_id: str, results: Dict[str, Any]):
        """
        保存处理后的图像结果
        
        Args:
            task_id: 任务ID
            results: 结果数据
        """
        try:
            # 创建临时目录
            temp_image_dir = os.path.join(settings.TEMP_DIR, f"{task_id}_images")
            os.makedirs(temp_image_dir, exist_ok=True)
            
            # 这里应该生成带有标注的图像
            # 目前为占位实现
            image_files = []
            
            # 如果有图像文件，创建ZIP压缩包
            if image_files:
                zip_path = os.path.join(settings.IMAGE_OUTPUT_DIR, f"{task_id}.zip")
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for image_file in image_files:
                        zipf.write(image_file, os.path.basename(image_file))
                
                # 清理临时目录
                shutil.rmtree(temp_image_dir)
                
        except Exception as e:
            logger.error(f"保存图像结果失败: {str(e)}")
            raise
    
    def _prepare_json_data(self, data: Any) -> Any:
        """
        准备JSON可序列化的数据
        
        Args:
            data: 原始数据
            
        Returns:
            可序列化的数据
        """
        if isinstance(data, dict):
            return {key: self._prepare_json_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._prepare_json_data(item) for item in data]
        elif hasattr(data, '__dict__'):
            # 对于自定义对象，转换为字典
            return self._prepare_json_data(data.__dict__)
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif isinstance(data, datetime):
            return data.isoformat()
        else:
            # 对于其他类型，转换为字符串
            return str(data)
    
    def estimate_processing_time(self, file_path: str) -> Optional[float]:
        """
        估算处理时间
        
        Args:
            file_path: 文件路径
            
        Returns:
            估算的处理时间（秒）
        """
        try:
            file_size = os.path.getsize(file_path)
            file_extension = Path(file_path).suffix.lower()
            
            # 简单的时间估算（实际应用中可以更精确）
            if file_extension in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                # 图像文件：基于文件大小
                return max(2.0, file_size / (1024 * 1024) * 0.5)  # 0.5秒/MB
            else:
                # 视频文件：基于文件大小（更长时间）
                return max(10.0, file_size / (1024 * 1024) * 2.0)  # 2秒/MB
                
        except Exception as e:
            logger.error(f"时间估算失败: {str(e)}")
            return None
    
    def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        清理临时文件
        
        Args:
            max_age_hours: 最大保存时间（小时）
        """
        try:
            current_time = datetime.now()
            temp_dir = Path(settings.TEMP_DIR)
            
            for file_path in temp_dir.iterdir():
                if file_path.is_file():
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age.total_seconds() > max_age_hours * 3600:
                        file_path.unlink()
                        logger.info(f"已清理临时文件: {file_path}")
                        
        except Exception as e:
            logger.error(f"清理临时文件失败: {str(e)}")
    
    def get_result_files(self, task_id: str) -> Dict[str, str]:
        """
        获取结果文件路径
        
        Args:
            task_id: 任务ID
            
        Returns:
            结果文件路径字典
        """
        files = {}
        
        # JSON结果文件
        json_path = os.path.join(settings.JSON_OUTPUT_DIR, f"{task_id}.json")
        if os.path.exists(json_path):
            files['json'] = json_path
        
        # 图像结果文件
        image_path = os.path.join(settings.IMAGE_OUTPUT_DIR, f"{task_id}.zip")
        if os.path.exists(image_path):
            files['images'] = image_path
        
        return files 