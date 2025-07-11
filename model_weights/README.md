# 模型权重文件目录

此目录用于存储各种AI模型的权重文件。

## 目录结构

```
model_weights/
├── rtmpose/          # RTMPose模型权重
├── sam/              # SAM模型权重
└── tracking/         # 追踪模型权重
```

## 所需模型文件

### RTMPose 模型
- **文件名**: `rtmpose-l_8xb64-210e_coco-384x288.onnx`
- **下载地址**: [RTMPose官方仓库](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose)
- **存放位置**: `rtmpose/`
- **说明**: COCO 17关键点人体姿态检测模型

### SAM 模型
- **文件名**: `sam_vit_h_4b8939.pth`
- **下载地址**: [Segment Anything官方仓库](https://github.com/facebookresearch/segment-anything)
- **存放位置**: `sam/`
- **说明**: Segment Anything Model，用于人物分割

### 追踪模型（可选）
- **文件名**: `osnet_x1_0_market1501.pth`
- **下载地址**: [Deep Person ReID](https://github.com/KaiyangZhou/deep-person-reid)
- **存放位置**: `tracking/`
- **说明**: 人物重识别模型，用于追踪

## 下载指南

1. **创建子目录**:
   ```bash
   mkdir -p model_weights/rtmpose
   mkdir -p model_weights/sam
   mkdir -p model_weights/tracking
   ```

2. **下载RTMPose模型**:
   ```bash
   # 访问 https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
   # 下载对应的ONNX模型文件
   ```

3. **下载SAM模型**:
   ```bash
   # 访问 https://github.com/facebookresearch/segment-anything#model-checkpoints
   # 下载 sam_vit_h_4b8939.pth
   ```

4. **验证文件**:
   ```bash
   ls -la model_weights/*/
   ```

## 注意事项

- 模型文件较大，请确保有足够的存储空间
- 建议使用GPU版本的模型以获得更好的性能
- 如果不使用某个功能，可以不下载对应的模型文件
- 模型路径配置在 `config/model_configs.py` 中 