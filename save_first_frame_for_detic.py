#!/usr/bin/env python3
"""
从demo2 notebook中保存第一帧图像用于DETIC测试
可以直接在notebook中运行这个代码块
"""

# 如果已经在demo2 notebook中运行过，可以直接使用已有的变量
# 否则需要先加载数据

try:
    # 检查是否已有frame_data
    if 'frame_data' in globals() and len(frame_data) > 0:
        # 获取第一帧
        first_frame_idx = sorted(frame_data.keys())[0]
        first_frame = frame_data[first_frame_idx]
        
        # 转换为PIL Image并保存
        from PIL import Image
        rgb_pil = Image.fromarray(first_frame['rgb'])
        
        # 保存图像
        output_path = 'test_frame.jpg'
        rgb_pil.save(output_path)
        
        print(f"✅ 已保存第一帧图像: {output_path}")
        print(f"   帧索引: {first_frame_idx}")
        print(f"   图像尺寸: {rgb_pil.size}")
        print(f"\n💡 现在可以在Detic目录中运行:")
        print(f"   cd Detic")
        print(f"   python demo.py \\")
        print(f"       --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml \\")
        print(f"       --input ../{output_path} \\")
        print(f"       --output out.jpg \\")
        print(f"       --vocabulary lvis \\")
        print(f"       --cpu \\")
        print(f"       --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth")
    else:
        print("❌ frame_data未找到或为空")
        print("   请先在demo2 notebook中运行数据加载代码")
        print("   确保frame_data变量已创建并包含数据")
except NameError as e:
    print(f"❌ 错误: {e}")
    print("   请确保在demo2 notebook中运行此代码")
except Exception as e:
    print(f"❌ 保存失败: {e}")
    import traceback
    traceback.print_exc()

