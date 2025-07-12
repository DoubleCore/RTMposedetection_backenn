#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片序列转换为MP4视频文件
支持多种图片格式，按照文件名顺序合成视频
"""

import os
import cv2
import re
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

class ImageToVideoConverter:
    """图片转视频转换器"""
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    def natural_sort_key(self, filename: str) -> List:
        """
        自然排序键，正确处理数字序列
        例如: img1.jpg, img2.jpg, img10.jpg 而不是 img1.jpg, img10.jpg, img2.jpg
        """
        def convert(text):
            if text.isdigit():
                return int(text)
            return text.lower()
        
        return [convert(c) for c in re.split('([0-9]+)', filename)]
    
    def get_image_files(self, folder_path: str) -> List[str]:
        """
        获取文件夹中所有支持的图片文件，并按自然顺序排序
        
        Args:
            folder_path: 图片文件夹路径
            
        Returns:
            排序后的图片文件路径列表
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"文件夹不存在: {folder_path}")
        
        image_files = []
        for file_path in folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_files.append(str(file_path))
        
        if not image_files:
            raise ValueError(f"在文件夹 {folder_path} 中没有找到支持的图片文件")
        
        # 按自然顺序排序
        image_files.sort(key=lambda x: self.natural_sort_key(os.path.basename(x)))
        
        print(f"找到 {len(image_files)} 个图片文件")
        print(f"第一个文件: {os.path.basename(image_files[0])}")
        print(f"最后一个文件: {os.path.basename(image_files[-1])}")
        
        return image_files
    
    def get_image_dimensions(self, image_files: List[str]) -> Tuple[int, int]:
        """
        获取图片尺寸，确保所有图片尺寸一致
        
        Args:
            image_files: 图片文件列表
            
        Returns:
            (width, height) 图片尺寸
        """
        first_image = cv2.imread(image_files[0])
        if first_image is None:
            raise ValueError(f"无法读取第一张图片: {image_files[0]}")
        
        height, width = first_image.shape[:2]
        
        # 检查所有图片尺寸是否一致
        print("检查图片尺寸一致性...")
        inconsistent_files = []
        
        for i, img_path in enumerate(tqdm(image_files, desc="验证图片尺寸")):
            img = cv2.imread(img_path)
            if img is None:
                print(f"警告: 无法读取图片 {img_path}")
                continue
                
            h, w = img.shape[:2]
            if h != height or w != width:
                inconsistent_files.append((img_path, w, h))
        
        if inconsistent_files:
            print("发现尺寸不一致的图片:")
            for file_path, w, h in inconsistent_files:
                print(f"  {os.path.basename(file_path)}: {w}x{h} (期望: {width}x{height})")
            
            resize_choice = input("是否要将所有图片调整为统一尺寸? (y/n): ").lower().strip()
            if resize_choice != 'y':
                raise ValueError("图片尺寸不一致，无法生成视频")
        
        return width, height
    
    def create_video_writer(self, output_path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
        """
        创建视频写入器
        
        Args:
            output_path: 输出视频文件路径
            width: 视频宽度
            height: 视频高度
            fps: 帧率
            
        Returns:
            OpenCV VideoWriter对象
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 使用H.264编码器
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            raise RuntimeError(f"无法创建视频文件: {output_path}")
        
        return video_writer
    
    def convert_images_to_video(self, 
                              input_folder: str, 
                              output_path: str, 
                              fps: float = 30.0,
                              resize_mode: str = 'fit') -> None:
        """
        将图片序列转换为MP4视频
        
        Args:
            input_folder: 输入图片文件夹路径
            output_path: 输出MP4文件路径
            fps: 视频帧率
            resize_mode: 调整尺寸模式 ('fit' 保持比例, 'stretch' 拉伸填充)
        """
        print(f"开始处理图片文件夹: {input_folder}")
        print(f"输出视频文件: {output_path}")
        print(f"视频帧率: {fps} FPS")
        
        # 获取所有图片文件
        image_files = self.get_image_files(input_folder)
        
        # 获取图片尺寸
        target_width, target_height = self.get_image_dimensions(image_files)
        
        # 创建视频写入器
        video_writer = self.create_video_writer(output_path, target_width, target_height, fps)
        
        try:
            print(f"开始合成视频，共 {len(image_files)} 帧...")
            
            for i, img_path in enumerate(tqdm(image_files, desc="处理图片")):
                # 读取图片
                img = cv2.imread(img_path)
                if img is None:
                    print(f"警告: 跳过无法读取的图片 {img_path}")
                    continue
                
                # 调整图片尺寸
                h, w = img.shape[:2]
                if h != target_height or w != target_width:
                    if resize_mode == 'fit':
                        # 保持比例调整
                        img = self.resize_with_aspect_ratio(img, target_width, target_height)
                    else:
                        # 直接拉伸
                        img = cv2.resize(img, (target_width, target_height))
                
                # 写入视频帧
                video_writer.write(img)
            
            print(f"视频合成完成！")
            print(f"输出文件: {output_path}")
            print(f"视频规格: {target_width}x{target_height} @ {fps} FPS")
            print(f"总帧数: {len(image_files)}")
            print(f"视频时长: {len(image_files)/fps:.2f} 秒")
            
        finally:
            video_writer.release()
    
    def resize_with_aspect_ratio(self, img: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """
        保持宽高比调整图片尺寸
        
        Args:
            img: 输入图片
            target_width: 目标宽度
            target_height: 目标高度
            
        Returns:
            调整后的图片
        """
        h, w = img.shape[:2]
        
        # 计算缩放比例
        scale_w = target_width / w
        scale_h = target_height / h
        scale = min(scale_w, scale_h)
        
        # 新的尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 调整尺寸
        resized_img = cv2.resize(img, (new_w, new_h))
        
        # 创建目标尺寸的黑色背景
        result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # 计算居中位置
        y_offset = (target_height - new_h) // 2
        x_offset = (target_width - new_w) // 2
        
        # 将调整后的图片放在中心
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
        
        return result

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将图片序列转换为MP4视频')
    parser.add_argument('input_folder', help='输入图片文件夹路径')
    parser.add_argument('-o', '--output', help='输出MP4文件路径', default='output_video.mp4')
    parser.add_argument('-f', '--fps', type=float, help='视频帧率', default=30.0)
    parser.add_argument('-r', '--resize', choices=['fit', 'stretch'], 
                       help='尺寸调整模式 (fit: 保持比例, stretch: 拉伸填充)', default='fit')
    
    args = parser.parse_args()
    
    try:
        converter = ImageToVideoConverter()
        converter.convert_images_to_video(
            input_folder=args.input_folder,
            output_path=args.output,
            fps=args.fps,
            resize_mode=args.resize
        )
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # 如果直接运行，可以在这里设置默认参数
    import sys
    
    if len(sys.argv) == 1:
        # 交互式模式
        print("=== 图片转视频工具 ===")
        input_folder = input("请输入图片文件夹路径: ").strip()
        output_file = input("请输入输出视频文件名 (默认: output_video.mp4): ").strip()
        if not output_file:
            output_file = "output_video.mp4"
        
        fps_input = input("请输入视频帧率 (默认: 30): ").strip()
        try:
            fps = float(fps_input) if fps_input else 30.0
        except ValueError:
            fps = 30.0
        
        try:
            converter = ImageToVideoConverter()
            converter.convert_images_to_video(input_folder, output_file, fps)
            print("\n转换完成！")
        except Exception as e:
            print(f"错误: {e}")
    else:
        # 命令行模式
        exit(main()) 