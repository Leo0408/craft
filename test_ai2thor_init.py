#!/usr/bin/env python3
"""
测试 AI2THOR Controller 初始化
用于诊断初始化卡住的问题
"""

import time
import sys

print("=" * 80)
print("AI2THOR Controller 初始化测试")
print("=" * 80)

try:
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering
    print("✅ AI2THOR 模块导入成功")
except ImportError as e:
    print(f"❌ AI2THOR 模块导入失败: {e}")
    print("   请运行: pip install ai2thor")
    sys.exit(1)

# 测试场景列表
test_scenes = ["FloorPlan1", "FloorPlan16", "FloorPlan201"]

for scene in test_scenes:
    print(f"\n{'='*80}")
    print(f"测试场景: {scene}")
    print(f"{'='*80}")
    
    print(f"[{time.strftime('%H:%M:%S')}] 开始初始化 Controller...")
    init_start = time.time()
    
    try:
        # 完全按照 REFLECT 的方式初始化
        controller = Controller(
            agentMode="default",
            massThreshold=None,
            scene=scene,
            visibilityDistance=1.5,
            gridSize=0.25,
            renderDepthImage=True,
            renderInstanceSegmentation=True,
            width=960,
            height=960,
            fieldOfView=60,
            platform=CloudRendering
        )
        
        init_duration = time.time() - init_start
        print(f"[{time.strftime('%H:%M:%S')}] ✅ Controller 初始化成功 ({init_duration:.2f}s)")
        
        # 立即验证（参考 REFLECT）
        print(f"[{time.strftime('%H:%M:%S')}] 验证 Controller (GetReachablePositions)...")
        verify_start = time.time()
        
        event = controller.step(action="GetReachablePositions")
        reachable_positions = event.metadata["actionReturn"]
        
        verify_duration = time.time() - verify_start
        print(f"[{time.strftime('%H:%M:%S')}] ✅ 验证成功: {len(reachable_positions)} 个可达位置 ({verify_duration:.2f}s)")
        
        # 清理
        controller.stop()
        print(f"✅ 场景 {scene} 测试成功，Controller 工作正常")
        break  # 如果成功，就使用这个场景
        
    except KeyboardInterrupt:
        init_duration = time.time() - init_start
        print(f"\n⚠️  用户中断 (已运行 {init_duration:.2f}s)")
        print("   如果初始化时间过长，可能是网络问题")
        sys.exit(1)
    except Exception as e:
        init_duration = time.time() - init_start
        print(f"❌ 场景 {scene} 失败 (已运行 {init_duration:.2f}s)")
        print(f"   错误: {type(e).__name__}: {e}")
        continue

print(f"\n{'='*80}")
print("测试完成")
print(f"{'='*80}")

