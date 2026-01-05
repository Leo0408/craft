#!/usr/bin/env python3
"""
查看DETIC检测结果
可以在notebook中运行，也可以作为脚本运行
"""

from PIL import Image
import matplotlib.pyplot as plt
import os

def view_detic_output(output_file='Detic/out_correct.jpg', title=None):
    """
    显示DETIC检测结果图像
    
    Args:
        output_file: 输出图像路径（相对于项目根目录或绝对路径）
        title: 图像标题
    """
    # 尝试多个可能的路径
    possible_paths = [
        output_file,
        os.path.join('Detic', os.path.basename(output_file)),
        os.path.join('/home/fdse/zzy/craft/Detic', os.path.basename(output_file)),
    ]
    
    img = None
    used_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                img = Image.open(path)
                used_path = path
                break
            except Exception as e:
                print(f"⚠️  无法打开 {path}: {e}")
                continue
    
    if img is None:
        # 查找所有可能的输出文件
        detic_dir = 'Detic'
        if os.path.exists(detic_dir):
            output_files = [f for f in os.listdir(detic_dir) if f.startswith('out') and f.endswith('.jpg')]
            if output_files:
                output_files.sort(key=lambda x: os.path.getmtime(os.path.join(detic_dir, x)), reverse=True)
                latest_file = os.path.join(detic_dir, output_files[0])
                try:
                    img = Image.open(latest_file)
                    used_path = latest_file
                    print(f"📁 使用最新输出文件: {latest_file}")
                except Exception as e:
                    print(f"❌ 无法打开 {latest_file}: {e}")
    
    if img is None:
        print("❌ 未找到输出图像文件")
        print("   请确认DETIC demo已成功运行并生成了输出文件")
        return None
    
    # 显示图像
    if title is None:
        title = f"DETIC检测结果 - {os.path.basename(used_path)}"
    
    print(f"✅ 显示图像: {used_path}")
    print(f"   图像尺寸: {img.size}")
    print(f"   文件大小: {os.path.getsize(used_path) / 1024:.1f} KB")
    
    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return img

if __name__ == "__main__":
    # 如果作为脚本运行，显示图像
    view_detic_output()

