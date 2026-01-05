#!/usr/bin/env python3
"""
从demo2 notebook中提取第一帧并测试DETIC官方demo
这个脚本可以在notebook中运行，也可以独立运行
"""

import os
import sys
import subprocess
from PIL import Image

def save_first_frame_from_demo2(frame_data, output_path='test_frame.jpg'):
    """
    从demo2的frame_data中提取第一帧并保存
    
    Args:
        frame_data: demo2中的frame_data字典
        output_path: 输出图像路径
    
    Returns:
        bool: 是否成功保存
    """
    try:
        if not frame_data or len(frame_data) == 0:
            print("❌ frame_data为空")
            return False
        
        # 获取第一帧
        first_frame_idx = sorted(frame_data.keys())[0]
        first_frame = frame_data[first_frame_idx]
        
        # 转换为PIL Image
        if 'rgb' not in first_frame:
            print("❌ frame_data中没有'rgb'字段")
            return False
        
        rgb_array = first_frame['rgb']
        rgb_pil = Image.fromarray(rgb_array)
        
        # 保存图像
        rgb_pil.save(output_path)
        
        print(f"✅ 已保存第一帧图像: {output_path}")
        print(f"   帧索引: {first_frame_idx}")
        print(f"   图像尺寸: {rgb_pil.size}")
        return True
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_detic_demo(image_path, detic_dir='Detic', output_name='out.jpg'):
    """
    运行DETIC官方demo
    
    Args:
        image_path: 输入图像路径（相对于项目根目录或绝对路径）
        detic_dir: Detic目录路径
        output_name: 输出文件名
    """
    # 转换为绝对路径
    if not os.path.isabs(image_path):
        # 如果路径不是绝对路径，尝试相对于当前目录
        if not os.path.exists(image_path):
            # 尝试相对于Detic目录
            detic_abs = os.path.abspath(detic_dir)
            image_path = os.path.join(detic_abs, os.path.basename(image_path))
            # 如果还是不存在，尝试上一级目录
            if not os.path.exists(image_path):
                image_path = os.path.join(os.path.dirname(detic_abs), os.path.basename(image_path))
    
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return False
    
    # 构建命令
    config_file = os.path.join(detic_dir, "configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml")
    weights_file = os.path.join(detic_dir, "models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth")
    output_file = os.path.join(detic_dir, output_name)
    
    # 相对于Detic目录的输入路径
    os.chdir(detic_dir)
    if os.path.isabs(image_path):
        input_path = image_path
    else:
        # 计算相对于Detic目录的路径
        input_path = os.path.relpath(image_path, detic_dir)
    
    cmd = [
        "python", "demo.py",
        "--config-file", "configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml",
        "--input", input_path,
        "--output", output_name,
        "--vocabulary", "lvis",
        "--cpu",
        "--opts", "MODEL.WEIGHTS", weights_file if os.path.exists(weights_file) else "models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth"
    ]
    
    print(f"\n🚀 运行DETIC demo...")
    print(f"   输入图像: {input_path}")
    print(f"   输出文件: {output_file}")
    print(f"   命令: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, cwd=detic_dir, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ DETIC demo运行成功！")
            if os.path.exists(output_file):
                print(f"   输出图像: {output_file}")
            return True
        else:
            print("❌ DETIC demo运行失败")
            if result.stderr:
                print("错误输出:")
                print(result.stderr[-1000:])
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  运行超时")
        return False
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        return False

if __name__ == "__main__":
    # 如果在notebook中运行，可以直接使用frame_data
    if 'frame_data' in globals():
        # 保存第一帧
        output_path = 'test_frame.jpg'
        if save_first_frame_from_demo2(frame_data, output_path):
            # 运行DETIC demo
            run_detic_demo(output_path)
    else:
        print("❌ frame_data未找到")
        print("   请在demo2 notebook中运行此脚本")
        print("   或手动提供图像路径:")
        print("   python test_detic_with_demo2_frame.py <image_path>")

