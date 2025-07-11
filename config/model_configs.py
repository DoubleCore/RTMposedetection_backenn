"""
模型相关配置
"""
from pathlib import Path
from config.settings import settings

MODEL_WEIGHTS_DIR = Path(settings.MODEL_WEIGHTS_DIR)

# RTMPose配置
RTMPOSE_CONFIG = {
    "model_path": str(MODEL_WEIGHTS_DIR / "rtmpose" / "rtmpose-l_8xb64-210e_coco-384x288.onnx"),
    "confidence_threshold": 0.3,
    "nms_threshold": 0.5,
    "input_size": (288, 384),  # (height, width)
    "keypoint_num": 17,  # COCO 17个关键点
}

# SAM配置
SAM_CONFIG = {
    "model_path": str(MODEL_WEIGHTS_DIR / "sam" / "sam_vit_h_4b8939.pth"),
    "model_type": "vit_h",
    "device": "cuda",
    "points_per_side": 32,
    "pred_iou_thresh": 0.88,
    "stability_score_thresh": 0.95,
}

# 追踪配置
TRACKING_CONFIG = {
    "model_path": str(MODEL_WEIGHTS_DIR / "tracking" / "osnet_x1_0_market1501.pth"),
    "max_dist": 0.2,
    "min_confidence": 0.3,
    "nms_max_overlap": 0.5,
    "max_iou_distance": 0.7,
    "max_age": 70,
    "n_init": 3,
}

# 分析配置
ANALYSIS_CONFIG = {
    "speed_calculation": {
        "fps": 30,
        "smoothing_window": 5,
        "units": "m/s"
    },
    "angle_calculation": {
        "joint_pairs": [
            ("shoulder", "elbow", "wrist"),
            ("hip", "knee", "ankle"),
            ("neck", "shoulder", "hip")
        ],
        "smoothing_window": 3
    }
} 