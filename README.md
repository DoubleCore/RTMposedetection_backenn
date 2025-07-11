# 跳水运动员姿态分析系统

## 项目简介

这是一个专门用于跳水运动员姿态分析的计算机视觉系统。该系统能够处理图像和视频，自动检测运动员姿态，分析运动速度和角度，并生成详细的分析报告。

## 功能特性

### 核心功能
1. **多格式输入支持**：支持图像和视频文件输入
2. **姿态检测**：基于RTMPose进行精确的人体姿态检测
3. **智能分割**：使用SAM模型进行人物分割，去除背景保留游泳池环境
4. **人物追踪**：多人场景下的个体追踪和标签化
5. **专业分析**：跳水运动员的速度和角度分析
6. **数据导出**：生成JSON格式的分析数据和处理后的图片

### 分析指标
- **速度分析**：运动员在不同阶段的速度变化
- **角度分析**：身体各部位的角度变化
- **帧级分析**：逐帧的详细姿态数据

## 技术架构

### 处理流程
```
输入(图像/视频) → RTMPose姿态检测 → SAM背景分割 → 人物追踪 → 姿态分析 → 数据导出
```

### 核心组件
- **RTMPose模块**：实时姿态检测
- **SAM模块**：语义分割和背景处理
- **追踪模块**：多目标追踪
- **分析模块**：运动学分析

## 项目结构

```
diving_pose_analysis/
├── main.py                 # 主入口文件
├── config/                 # 配置管理
├── api/                    # Web API接口
├── core/                   # 核心处理逻辑
├── models/                 # AI模型集成
│   ├── rtmpose/           # RTMPose相关
│   ├── sam/               # SAM分割相关
│   └── tracking/          # 追踪相关
├── analysis/              # 专业分析模块
├── utils/                 # 工具函数
├── output/                # 输出文件
└── model_weights/         # 模型权重文件
```

## 安装指南

### 环境要求
- Python 3.8+
- CUDA 11.0+ (GPU支持)
- 8GB+ RAM

### 依赖安装
```bash
pip install -r requirements.txt
```

### 模型权重
1. 下载RTMPose预训练模型
2. 下载SAM模型权重
3. 将模型文件放置在 `model_weights/` 对应目录

## 使用说明

### 快速开始
```bash
# 启动Web服务
python main.py

# API调用示例
curl -X POST "http://localhost:8000/api/v1/analyze" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_video.mp4"
```

### 批量处理
```bash
# 批量处理视频文件
python -m core.pipeline --input_dir ./videos --output_dir ./results
```

## API接口

### 主要端点
- `POST /api/v1/analyze` - 单个文件分析
- `POST /api/v1/batch_analyze` - 批量文件分析
- `GET /api/v1/status/{task_id}` - 查询处理状态
- `GET /api/v1/download/{task_id}` - 下载结果

### 输入格式
支持的文件格式：
- 图像：JPG, PNG, BMP, WEBP
- 视频：MP4, AVI, MOV, MKV

### 输出格式
- **JSON文件**：包含详细的姿态数据、速度、角度信息
- **处理图片**：带有标注的分析结果图

## 配置说明

### 模型配置 (config/model_configs.py)
```python
# RTMPose配置
RTMPOSE_CONFIG = {
    "model_path": "model_weights/rtmpose/rtmpose-l_8xb64-210e_coco-384x288.onnx",
    "confidence_threshold": 0.3
}

# SAM配置
SAM_CONFIG = {
    "model_path": "model_weights/sam/sam_vit_h_4b8939.pth",
    "device": "cuda"
}
```

### 系统配置 (config/settings.py)
```python
# 服务器设置
HOST = "0.0.0.0"
PORT = 8000

# 文件处理设置
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
SUPPORTED_FORMATS = [".mp4", ".avi", ".mov", ".jpg", ".png"]
```

## 开发指南

### 添加新的分析功能
1. 在 `analysis/` 目录下创建新的分析模块
2. 在 `core/pipeline.py` 中集成新功能
3. 更新API接口和文档

### 模型替换
1. 在对应的 `models/` 子目录中实现新的模型接口
2. 遵循现有的输入输出格式
3. 更新配置文件

## 性能优化

### 硬件建议
- GPU：RTX 3080 或更高
- CPU：8核以上
- 内存：16GB+
- 存储：SSD推荐

### 处理速度参考
- 单张图片：~0.5秒
- 1分钟视频：~30秒（30fps）

## 故障排除

### 常见问题
1. **模型加载失败**：检查模型文件路径和权限
2. **内存不足**：减少批处理数量或使用更小的模型
3. **GPU错误**：确认CUDA版本兼容性

### 日志查看
```bash
# 查看详细日志
tail -f logs/diving_analysis.log
```

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 许可证

MIT License

## 联系方式

- 项目维护者：[您的姓名]
- 邮箱：[您的邮箱]
- 项目地址：[GitHub链接]

## 更新日志

### v1.0.0 (2024-01-XX)
- 初始版本发布
- 支持基础姿态检测和分析
- 集成RTMPose和SAM模型
- 实现速度和角度分析功能

---

## 快速命令参考

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python main.py

# 运行测试
pytest tests/

# 代码格式化
black .
isort .

# 类型检查
mypy .
``` 