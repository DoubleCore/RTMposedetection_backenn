"""
SAM处理器
专门用于分离人物和游泳池背景的FastSAM分割处理
"""
import cv2
import numpy as np
import os
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

from .fast_sam import get_fast_sam_segmenter

logger = logging.getLogger(__name__)

class SAMProcessor:
    """
    SAM处理器 - 专门分离人物和游泳池背景
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化SAM处理器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.pose_dir = os.path.join(output_dir, "pose")
        self.sam_dir = os.path.join(output_dir, "sam")
        self.json_dir = os.path.join(output_dir, "json")
        
        # 确保SAM输出目录存在
        os.makedirs(self.sam_dir, exist_ok=True)
        
        # 获取FastSAM分割器
        self.segmenter = get_fast_sam_segmenter()
        
        logger.info(f"SAM处理器初始化完成，专门用于人物和游泳池背景分离")
    
    def process_all_frames(self, processing_mode: str = "separate_person_pool") -> Dict[str, Any]:
        """
        处理所有帧的人物和游泳池背景分离
        
        Args:
            processing_mode: 处理模式
                - "separate_person_pool": 分离人物和游泳池
                - "highlight_person": 突出人物，模糊背景
                - "extract_person": 提取人物，透明背景
                - "pool_only": 只保留游泳池，移除人物
                
        Returns:
            处理结果统计
        """
        try:
            logger.info(f"🔄 开始人物和游泳池背景分离处理，模式: {processing_mode}")
            
            # 查找所有pose图片和对应的JSON文件
            pose_files = []
            for filename in os.listdir(self.pose_dir):
                if filename.startswith("frame_") and filename.endswith(".jpg"):
                    pose_files.append(filename)
            
            pose_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            
            processed_count = 0
            error_count = 0
            person_detected_count = 0
            
            for pose_filename in pose_files:
                try:
                    # 提取帧号
                    frame_number = int(pose_filename.split('_')[1].split('.')[0])
                    
                    # 处理单个帧
                    result = self.process_single_frame(frame_number, processing_mode)
                    
                    if result["success"]:
                        processed_count += 1
                        if result.get("person_detected", False):
                            person_detected_count += 1
                        
                        # 进度日志
                        if frame_number % 50 == 0:
                            logger.info(f"🎯 已处理 {processed_count} 帧，检测到人物 {person_detected_count} 帧...")
                        elif frame_number <= 10:
                            logger.info(f"🎯 处理帧 {frame_number}: 人物={'是' if result.get('person_detected') else '否'}")
                    else:
                        error_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ 处理帧 {pose_filename} 失败: {str(e)}")
                    error_count += 1
                    continue
            
            logger.info(f"✅ 人物和游泳池分离完成: 成功 {processed_count} 帧, 人物检测 {person_detected_count} 帧, 失败 {error_count} 帧")
            
            return {
                "total_processed": processed_count,
                "person_detected_frames": person_detected_count,
                "total_errors": error_count,
                "processing_mode": processing_mode,
                "output_directory": self.sam_dir
            }
            
        except Exception as e:
            logger.error(f"批量处理失败: {str(e)}")
            return {"error": str(e)}
    
    def process_single_frame(self, frame_number: int, processing_mode: str = "separate_person_pool") -> Dict[str, Any]:
        """
        处理单个帧的人物和游泳池背景分离
        
        Args:
            frame_number: 帧号
            processing_mode: 处理模式
            
        Returns:
            处理结果
        """
        try:
            # 构建文件路径
            pose_filename = f"frame_{frame_number}.jpg"
            json_filename = f"frame_{frame_number}.json"
            
            pose_path = os.path.join(self.pose_dir, pose_filename)
            json_path = os.path.join(self.json_dir, json_filename)
            
            # 读取原始图片（优先从origin目录）
            origin_path = os.path.join(self.output_dir, "origin", pose_filename)
            if os.path.exists(origin_path):
                image = cv2.imread(origin_path)
                logger.info(f"🎯 处理帧 {frame_number}: 使用原始图片 {origin_path}")
            else:
                image = cv2.imread(pose_path)
                logger.info(f"🎯 处理帧 {frame_number}: 使用pose图片 {pose_path}")
            
            if image is None:
                logger.error(f"❌ 无法读取帧 {frame_number} 的图像")
                return {"success": False, "error": f"无法读取图像"}
            
            # 读取pose检测结果
            pose_data = None
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    pose_data = json.load(f)
                logger.info(f"📋 帧 {frame_number}: 读取到pose数据，包含 {len(pose_data.get('persons', []))} 个人物")
            else:
                logger.warning(f"⚠️ 帧 {frame_number}: 未找到pose数据文件 {json_path}")
            
            # 使用FastSAM进行全图分割
            logger.info(f"🔄 帧 {frame_number}: 开始FastSAM分割...")
            segment_result = self.segmenter.segment_everything(image)
            logger.info(f"✅ 帧 {frame_number}: FastSAM分割完成，检测到 {segment_result['num_segments']} 个区域")
            
            # 分析分割结果，识别人物和游泳池区域
            logger.info(f"🔍 帧 {frame_number}: 开始识别人物和游泳池区域...")
            person_mask, pool_mask = self._identify_person_and_pool_regions(
                image, segment_result, pose_data
            )
            
            person_detected = np.any(person_mask > 0) if person_mask is not None else False
            pool_detected = np.any(pool_mask > 0) if pool_mask is not None else False
            
            logger.info(f"🎯 帧 {frame_number}: 识别结果 - 人物: {'✅' if person_detected else '❌'}, 游泳池: {'✅' if pool_detected else '❌'}")
            
            # 根据处理模式生成结果图像
            logger.info(f"🖼️ 帧 {frame_number}: 生成 {processing_mode} 模式的处理图像...")
            result_images = self._generate_processed_images(
                image, person_mask, pool_mask, processing_mode
            )
            
            # 保存处理结果
            saved_files = self._save_processing_results(
                frame_number, result_images, person_mask, pool_mask, 
                segment_result, processing_mode
            )
            
            logger.info(f"💾 帧 {frame_number}: 保存了 {len(saved_files)} 个文件")
            
            return {
                "success": True,
                "frame_number": frame_number,
                "person_detected": person_detected,
                "pool_detected": pool_detected,
                "processing_mode": processing_mode,
                "saved_files": saved_files,
                "num_segments": segment_result["num_segments"],
                "debug_info": {
                    "image_source": "origin" if os.path.exists(origin_path) else "pose",
                    "pose_data_available": pose_data is not None,
                    "segments_detected": segment_result["num_segments"],
                    "result_images_generated": len(result_images)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 处理帧 {frame_number} 失败: {str(e)}")
            return {"success": False, "error": str(e), "frame_number": frame_number}
    
    def _identify_person_and_pool_regions(self, image: np.ndarray, segment_result: Dict, 
                                        pose_data: Optional[Dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        识别人物和游泳池区域
        
        Args:
            image: 原始图像
            segment_result: FastSAM分割结果
            pose_data: pose检测数据
            
        Returns:
            (person_mask, pool_mask) 元组
        """
        try:
            height, width = image.shape[:2]
            person_mask = np.zeros((height, width), dtype=np.uint8)
            pool_mask = np.zeros((height, width), dtype=np.uint8)
            
            if not segment_result["masks"]:
                return None, None
            
            # 如果有pose数据，优先使用pose信息识别人物
            person_regions = []
            if pose_data and "persons" in pose_data:
                person_regions = self._extract_person_regions_from_pose(pose_data, image.shape[:2])
            
            # 分析每个分割区域
            for i, mask in enumerate(segment_result["masks"]):
                if i >= len(segment_result["segments"]):
                    continue
                    
                segment_info = segment_result["segments"][i]
                mask_resized = cv2.resize(mask, (width, height))
                
                # 判断是否为人物区域
                if self._is_person_region(mask_resized, segment_info, person_regions, image):
                    person_mask = np.logical_or(person_mask, mask_resized > 0)
                
                # 判断是否为游泳池区域（大面积，蓝色调，在底部区域）
                elif self._is_pool_region(mask_resized, segment_info, image):
                    pool_mask = np.logical_or(pool_mask, mask_resized > 0)
            
            # 如果没有检测到明显的游泳池，将非人物区域作为背景
            if not np.any(pool_mask):
                pool_mask = ~person_mask
            
            return person_mask.astype(np.uint8), pool_mask.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"识别人物和游泳池区域失败: {str(e)}")
            return None, None
    
    def _extract_person_regions_from_pose(self, pose_data: Dict, image_size: Tuple[int, int]) -> List[np.ndarray]:
        """
        从pose数据中提取人物区域
        """
        person_regions = []
        height, width = image_size
        
        try:
            for person in pose_data.get("persons", []):
                if "keypoints" in person and "scores" in person:
                    keypoints = person["keypoints"]
                    scores = person["scores"]
                    
                    # 找到有效关键点
                    valid_points = []
                    for i, (x, y) in enumerate(keypoints):
                        if i < len(scores) and scores[i] > 0.1:
                            valid_points.append((int(x), int(y)))
                    
                    if len(valid_points) >= 3:
                        # 创建人物区域mask
                        person_mask = np.zeros((height, width), dtype=np.uint8)
                        
                        # 计算边界框并扩展
                        x_coords = [pt[0] for pt in valid_points]
                        y_coords = [pt[1] for pt in valid_points]
                        
                        margin = 50  # 增大边距以包含更多人物区域
                        x1 = max(0, min(x_coords) - margin)
                        y1 = max(0, min(y_coords) - margin)
                        x2 = min(width, max(x_coords) + margin)
                        y2 = min(height, max(y_coords) + margin)
                        
                        person_mask[y1:y2, x1:x2] = 255
                        person_regions.append(person_mask)
            
        except Exception as e:
            logger.error(f"提取人物区域失败: {str(e)}")
        
        return person_regions
    
    def _is_person_region(self, mask: np.ndarray, segment_info: Dict, 
                         person_regions: List[np.ndarray], image: np.ndarray) -> bool:
        """
        判断是否为人物区域
        """
        try:
            # 如果有pose信息，检查与pose区域的重叠
            if person_regions:
                for person_region in person_regions:
                    overlap = np.logical_and(mask > 0, person_region > 0)
                    overlap_ratio = np.sum(overlap) / max(int(np.sum(mask > 0)), 1)
                    if overlap_ratio > 0.2:  # 降低到20%以上重叠认为是人物
                        return True
            
            # 基于区域特征判断
            area = segment_info.get("area", 0)
            bbox = segment_info.get("bbox", [0, 0, 0, 0])
            
            # 人物通常不会占据整个图像，且有合理的宽高比
            image_area = image.shape[0] * image.shape[1]
            area_ratio = area / image_area
            
            # 放宽面积条件：0.005-50%之间都可能是人物
            if 0.005 < area_ratio < 0.5:  
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                if height > 0:
                    aspect_ratio = width / height
                    # 放宽宽高比：0.2-3.0之间
                    if 0.2 < aspect_ratio < 3.0:
                        # 额外检查：如果区域不在图像边缘，更可能是人物
                        center_x = (bbox[0] + bbox[2]) / 2
                        center_y = (bbox[1] + bbox[3]) / 2
                        
                        # 如果区域中心不在图像的极边缘位置，认为是人物
                        if (0.05 * image.shape[1] < center_x < 0.95 * image.shape[1] and 
                            0.05 * image.shape[0] < center_y < 0.95 * image.shape[0]):
                            return True
                        
                        # 如果面积足够大，也可能是人物
                        if area_ratio > 0.02:
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"判断人物区域失败: {str(e)}")
            return False
    
    def _is_pool_region(self, mask: np.ndarray, segment_info: Dict, image: np.ndarray) -> bool:
        """
        判断是否为游泳池区域
        """
        try:
            area = segment_info.get("area", 0)
            bbox = segment_info.get("bbox", [0, 0, 0, 0])
            
            # 游泳池通常占据较大面积
            image_area = image.shape[0] * image.shape[1]
            area_ratio = area / image_area
            
            # 降低面积要求：15%以上
            if area_ratio > 0.15:  
                # 检查区域是否主要在图像下半部分（游泳池通常在下方）
                y_center = (bbox[1] + bbox[3]) / 2
                if y_center > image.shape[0] * 0.3:  # 重心在图像下部30%以下
                    
                    # 分析颜色特征（游泳池通常是蓝色调）
                    masked_region = image[mask > 0]
                    if len(masked_region) > 0:
                        # 计算蓝色通道的均值
                        mean_bgr = np.mean(masked_region, axis=0)
                        blue_ratio = mean_bgr[0] / (np.sum(mean_bgr) + 1e-6)  # B通道占比
                        
                        # 降低蓝色占比要求：30%以上或者是大面积的背景区域
                        if blue_ratio > 0.3 or area_ratio > 0.4:
                            return True
                        
                        # 额外检查：如果是非常大的区域（可能是背景）
                        if area_ratio > 0.6:
                            return True
            
            # 如果是占据图像大部分且位置较低的区域，也认为是游泳池
            if area_ratio > 0.3:
                bbox_bottom = bbox[3]
                if bbox_bottom > image.shape[0] * 0.7:  # 底部边界在图像下方70%以下
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"判断游泳池区域失败: {str(e)}")
            return False
    
    def _generate_processed_images(self, image: np.ndarray, person_mask: Optional[np.ndarray], 
                                 pool_mask: Optional[np.ndarray], processing_mode: str) -> Dict[str, np.ndarray]:
        """
        根据处理模式生成结果图像
        """
        result_images = {}
        
        try:
            if person_mask is None or pool_mask is None:
                result_images["original"] = image.copy()
                return result_images
            
            if processing_mode == "separate_person_pool":
                # 模式1：分离人物和游泳池
                # 人物图像（白色背景）
                person_img = image.copy()
                person_img[person_mask == 0] = [255, 255, 255]
                result_images["person_only"] = person_img
                
                # 游泳池图像（移除人物）
                pool_img = image.copy()
                pool_img[person_mask > 0] = [255, 255, 255]
                result_images["pool_only"] = pool_img
                
                # 组合图像（左：人物，右：游泳池）
                h, w = image.shape[:2]
                combined = np.zeros((h, w*2, 3), dtype=np.uint8)
                combined[:, :w] = person_img
                combined[:, w:] = pool_img
                result_images["combined"] = combined
            
            elif processing_mode == "highlight_person":
                # 模式2：突出人物，模糊游泳池背景
                highlighted = image.copy()
                
                # 模糊背景
                blurred_bg = cv2.GaussianBlur(image, (21, 21), 0)
                highlighted[person_mask == 0] = blurred_bg[person_mask == 0]
                
                result_images["highlighted_person"] = highlighted
            
            elif processing_mode == "extract_person":
                # 模式3：提取人物，透明背景
                person_extracted = image.copy()
                person_extracted = cv2.cvtColor(person_extracted, cv2.COLOR_BGR2BGRA)
                person_extracted[person_mask == 0, 3] = 0  # 设置透明
                
                result_images["person_extracted"] = person_extracted
            
            elif processing_mode == "pool_only":
                # 模式4：只保留游泳池，移除人物
                pool_only = image.copy()
                
                # 使用背景修复技术填充人物区域
                if np.any(person_mask > 0):
                    inpaint_mask = person_mask.copy()
                    pool_only = cv2.inpaint(pool_only, inpaint_mask, 3, cv2.INPAINT_TELEA)
                
                result_images["pool_only"] = pool_only
            
            # 总是生成原图和mask可视化
            result_images["original"] = image.copy()
            
            # 创建mask可视化
            mask_viz = np.zeros_like(image)
            mask_viz[person_mask > 0] = [0, 255, 0]  # 绿色表示人物
            mask_viz[pool_mask > 0] = [255, 0, 0]   # 蓝色表示游泳池
            result_images["mask_visualization"] = mask_viz
            
        except Exception as e:
            logger.error(f"生成处理图像失败: {str(e)}")
            result_images["original"] = image.copy()
        
        return result_images
    
    def _save_processing_results(self, frame_number: int, result_images: Dict[str, np.ndarray],
                               person_mask: Optional[np.ndarray], pool_mask: Optional[np.ndarray],
                               segment_result: Dict, processing_mode: str) -> List[str]:
        """
        保存处理结果
        """
        saved_files = []
        
        try:
            # 保存各种结果图像
            for image_type, image_data in result_images.items():
                filename = f"frame_{frame_number}_{image_type}.jpg"
                filepath = os.path.join(self.sam_dir, filename)
                
                # 处理RGBA图像
                if image_data.shape[-1] == 4:  # RGBA
                    cv2.imwrite(filepath.replace('.jpg', '.png'), image_data)
                    saved_files.append(filename.replace('.jpg', '.png'))
                else:
                    cv2.imwrite(filepath, image_data)
                    saved_files.append(filename)
            
            # 保存JSON信息
            json_info = {
                "frame_number": frame_number,
                "processing_time": datetime.now().isoformat(),
                "processing_mode": processing_mode,
                "person_detected": np.any(person_mask > 0) if person_mask is not None else False,
                "pool_detected": np.any(pool_mask > 0) if pool_mask is not None else False,
                "num_segments": segment_result["num_segments"],
                "saved_images": saved_files,
                "segment_info": segment_result.get("segments", [])
            }
            
            json_filename = f"frame_{frame_number}_sam_analysis.json"
            json_path = os.path.join(self.sam_dir, json_filename)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_info, f, indent=2, ensure_ascii=False)
            
            saved_files.append(json_filename)
            
        except Exception as e:
            logger.error(f"保存处理结果失败: {str(e)}")
        
        return saved_files
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        获取处理统计信息
        """
        try:
            # 统计各类文件数量
            if not os.path.exists(self.sam_dir):
                return {"error": "SAM目录不存在"}
            
            files = os.listdir(self.sam_dir)
            
            person_only_images = [f for f in files if 'person_only' in f and f.endswith('.jpg')]
            pool_only_images = [f for f in files if 'pool_only' in f and f.endswith('.jpg')]
            combined_images = [f for f in files if 'combined' in f and f.endswith('.jpg')]
            highlighted_images = [f for f in files if 'highlighted_person' in f and f.endswith('.jpg')]
            mask_viz_images = [f for f in files if 'mask_visualization' in f and f.endswith('.jpg')]
            analysis_jsons = [f for f in files if 'sam_analysis' in f and f.endswith('.json')]
            
            return {
                "person_only_images": len(person_only_images),
                "pool_only_images": len(pool_only_images),
                "combined_images": len(combined_images),
                "highlighted_images": len(highlighted_images),
                "mask_visualizations": len(mask_viz_images),
                "analysis_files": len(analysis_jsons),
                "total_files": len(files),
                "output_directory": self.sam_dir,
                "latest_files": {
                    "recent_combined": combined_images[-3:] if combined_images else [],
                    "recent_analysis": analysis_jsons[-3:] if analysis_jsons else []
                }
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {str(e)}")
            return {"error": str(e)}

# 全局处理器实例（单例模式）
_sam_processor_instance = None

def get_sam_processor(output_dir: str = "output") -> SAMProcessor:
    """
    获取SAM处理器实例（单例）
    
    Args:
        output_dir: 输出目录
        
    Returns:
        SAM处理器实例
    """
    global _sam_processor_instance
    
    if _sam_processor_instance is None:
        _sam_processor_instance = SAMProcessor(output_dir)
    
    return _sam_processor_instance 