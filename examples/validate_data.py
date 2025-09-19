#!/usr/bin/env python3
"""
数据集验证脚本
检查所有文本数据文件的完整性和格式正确性
"""

import json
import os
from pathlib import Path
import sys

def validate_json_format(json_path):
    """
    验证JSON文件格式
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            return False, "数据应该是一个列表"
        
        if len(data) == 0:
            return False, "数据为空"
        
        # 检查第一个样本的结构
        sample = data[0]
        required_keys = ['img_name', 'mask_name']
        
        for key in required_keys:
            if key not in sample:
                return False, f"缺少必需字段: {key}"
        
        # 检查是否有progressive级别
        progressive_levels = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5']
        has_progressive = any(level in sample for level in progressive_levels)
        
        if not has_progressive:
            return False, "没有发现progressive learning级别 (P0-P5)"
        
        return True, f"有效 - {len(data)} 个样本"
        
    except json.JSONDecodeError as e:
        return False, f"JSON解析错误: {e}"
    except Exception as e:
        return False, f"未知错误: {e}"

def main():
    print("=== 数据集验证工具 ===\n")
    
    # 定义数据路径
    data_base = "data/text_annotations"
    
    if not os.path.exists(data_base):
        print(f"❌ 数据目录不存在: {data_base}")
        print("请确保已复制所有文本数据到项目中")
        return
    
    # 找到所有JSON文件
    json_files = []
    for root, dirs, files in os.walk(data_base):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    print(f"发现 {len(json_files)} 个JSON文件\n")
    
    # 验证每个文件
    valid_files = 0
    total_samples = 0
    
    for json_file in sorted(json_files):
        rel_path = os.path.relpath(json_file, data_base)
        print(f"📁 验证: {rel_path}")
        
        # 基本格式验证
        is_valid, message = validate_json_format(json_file)
        if is_valid:
            print(f"  ✅ 格式: {message}")
            valid_files += 1
            
            # 获取样本数量
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                samples = len(data)
                total_samples += samples
                
        else:
            print(f"  ❌ 格式错误: {message}")
        
        print()
    
    # 总结
    print("=" * 50)
    print(f"验证完成:")
    print(f"  有效文件: {valid_files}/{len(json_files)}")
    print(f"  总样本数: {total_samples}")
    
    if valid_files == len(json_files):
        print("  🎉 所有数据文件验证通过！")
    else:
        print(f"  ⚠️  有 {len(json_files) - valid_files} 个文件存在问题")

if __name__ == "__main__":
    main()