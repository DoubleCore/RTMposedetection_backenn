"""
全局配置管理
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings

# 项目根目录
BASE_DIR = Path(__file__).parent.parent

class Settings(BaseSettings):
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # 文件上传配置
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
    ALLOWED_EXTENSIONS: set = {".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    # 输出路径配置
    OUTPUT_DIR: str = str(BASE_DIR / "output")
    JSON_OUTPUT_DIR: str = str(BASE_DIR / "output" / "json")
    IMAGE_OUTPUT_DIR: str = str(BASE_DIR / "output" / "images")
    TEMP_DIR: str = str(BASE_DIR / "output" / "temp")
    
    # 模型权重路径
    MODEL_WEIGHTS_DIR: str = str(BASE_DIR / "model_weights")
    
    # 处理参数
    VIDEO_FPS: int = 30
    IMAGE_QUALITY: int = 95
    
    # 分析配置
    CONFIDENCE_THRESHOLD: float = 0.3
    NMS_THRESHOLD: float = 0.5
    
    class Config:
        env_file = ".env"

settings = Settings()

# 创建必要的目录
for directory in [
    settings.OUTPUT_DIR,
    settings.JSON_OUTPUT_DIR,
    settings.IMAGE_OUTPUT_DIR,
    settings.TEMP_DIR,
    settings.MODEL_WEIGHTS_DIR
]:
    os.makedirs(directory, exist_ok=True) 