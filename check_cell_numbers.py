#!/usr/bin/env python3
"""
查看 demo1.ipynb 中特定 cells 的编号
"""

import json
import sys

def find_cells_by_keyword(notebook_path, keywords):
    """根据关键词查找 cells"""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    print("=" * 80)
    print(f"在 {notebook_path} 中查找 cells")
    print("=" * 80)
    
    found_cells = []
    
    for i, cell in enumerate(nb['cells']):
        source = ''.join(cell.get('source', []))
        cell_type = cell.get('cell_type', 'unknown')
        
        for keyword in keywords:
            if keyword in source:
                first_line = source.split('\n')[0].strip()[:100]
                found_cells.append({
                    'index': i,
                    'type': cell_type,
                    'keyword': keyword,
                    'first_line': first_line
                })
                break
    
    if found_cells:
        print(f"\n找到 {len(found_cells)} 个匹配的 cells:\n")
        for cell_info in found_cells:
            print(f"Cell {cell_info['index']:3d} ({cell_info['type']:8s}) - 关键词: {cell_info['keyword']}")
            print(f"   {cell_info['first_line']}")
            print()
    else:
        print("\n未找到匹配的 cells")
    
    return found_cells

if __name__ == '__main__':
    notebook_path = 'demo1.ipynb'
    
    # 查找关键 cells
    keywords = [
        '步骤 1: 生成 AI2THOR',
        '步骤 2: 从已保存数据',
        'generate_all_ai2thor_failure_cases',
        'generate_ai2thor_failure_case_data',
        'run_ai2thor_comparison_test_from_data',
        'AI2THOR 环境中的失败注入测试',
        '检查必要的函数定义'
    ]
    
    find_cells_by_keyword(notebook_path, keywords)
    
    # 显示所有 cells 的概览
    print("\n" + "=" * 80)
    print("所有 cells 概览（前 50 个）:")
    print("=" * 80)
    
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    for i, cell in enumerate(nb['cells'][:50]):
        cell_type = cell.get('cell_type', 'unknown')
        source = ''.join(cell.get('source', []))
        first_line = source.split('\n')[0].strip()[:60]
        print(f"Cell {i:3d} ({cell_type:8s}): {first_line}")

