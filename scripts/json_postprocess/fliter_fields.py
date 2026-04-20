#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本1: JSON字段过滤器 (V2 - 支持文件夹批量处理)
功能: 保留指定字段，删除其他字段
用法: python filter_fields.py [--input 文件夹] [--output 输出文件夹] [--keep 字段1 字段2 ...]

可在代码中修改 DEFAULT_CONFIG 来预设参数
"""

import json
import argparse
import sys
import os
from pathlib import Path

# ============================================
# 默认配置 - 可在此修改预设值
# ============================================
DEFAULT_CONFIG = {
    "input_dir": "/home/caep-xuben/chenchengbing/yzj/datasets/llm_baseline_366/final_json/json",      # 输入文件夹路径
    "output_dir": "/home/caep-xuben/chenchengbing/yzj/experiments/exp_0404/data/json_postprocess/fliter_fields", # 输出文件夹路径
    "keep_fields": ["papers", "alloys", "processes", "samples", "processing_steps", "structures", "properties"], # 要保留的字段列表
    "file_pattern": "*.json"             # 文件匹配模式
}
# ============================================


def filter_json_fields(data, keep_fields):
    """
    从JSON数据中保留指定字段，删除其他字段
    """
    filtered_data = {}
    removed_fields = []

    for field in keep_fields:
        if field in data:
            filtered_data[field] = data[field]
        else:
            removed_fields.append(field)

    actually_removed = [k for k in data.keys() if k not in keep_fields]

    return filtered_data, removed_fields, actually_removed


def process_single_file(input_file, output_file, keep_fields):
    """处理单个JSON文件"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ 错误 [{input_file}]: JSON解析失败 - {e}")
        return False
    except Exception as e:
        print(f"❌ 错误 [{input_file}]: {e}")
        return False

    filtered_data, missing_fields, removed_fields = filter_json_fields(data, keep_fields)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=2, ensure_ascii=False)

    print(f"✅ [{os.path.basename(input_file)}] 保留 {len(filtered_data)} 个字段, 删除 {len(removed_fields)} 个字段")
    if missing_fields:
        print(f"   ⚠️  警告: 字段 {missing_fields} 不存在")

    return True


def process_directory(input_dir, output_dir, keep_fields, pattern="*.json"):
    """批量处理文件夹中的所有JSON文件"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"❌ 错误: 输入文件夹不存在 '{input_dir}'")
        return False

    # 获取所有匹配的文件
    json_files = list(input_path.glob(pattern))

    if not json_files:
        print(f"⚠️  在 '{input_dir}' 中没有找到匹配 '{pattern}' 的文件")
        return False

    print(f"\n📁 找到 {len(json_files)} 个文件待处理")
    print(f"📋 保留字段: {keep_fields}")
    print(f"📂 输出目录: {output_dir}")
    print("="*60)

    success_count = 0
    for json_file in json_files:
        # 保持相对目录结构
        relative_path = json_file.relative_to(input_path)
        output_file = output_path / relative_path

        if process_single_file(str(json_file), str(output_file), keep_fields):
            success_count += 1

    print("="*60)
    print(f"\n✅ 处理完成: {success_count}/{len(json_files)} 个文件成功")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='保留JSON中的指定字段，删除其他字段（支持批量处理）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
默认配置 (可在代码中修改 DEFAULT_CONFIG):
  输入文件夹: {DEFAULT_CONFIG['input_dir']}
  输出文件夹: {DEFAULT_CONFIG['output_dir']}
  保留字段: {DEFAULT_CONFIG['keep_fields']}

示例:
  # 使用代码预设值
  python filter_fields.py

  # 命令行覆盖输入输出路径
  python filter_fields.py --input ./data --output ./result

  # 命令行覆盖保留字段
  python filter_fields.py --keep papers alloys samples

  # 完整命令行指定
  python filter_fields.py -i ./input -o ./output -k papers alloys
        """
    )

    parser.add_argument('--input', '-i', 
                        default=DEFAULT_CONFIG['input_dir'],
                        help=f'输入文件夹路径 (默认: {DEFAULT_CONFIG["input_dir"]})')
    parser.add_argument('--output', '-o',
                        default=DEFAULT_CONFIG['output_dir'],
                        help=f'输出文件夹路径 (默认: {DEFAULT_CONFIG["output_dir"]})')
    parser.add_argument('--keep', '-k', nargs='+',
                        default=DEFAULT_CONFIG['keep_fields'],
                        help=f'要保留的字段名 (默认: {DEFAULT_CONFIG["keep_fields"]})')
    parser.add_argument('--pattern', '-p',
                        default=DEFAULT_CONFIG['file_pattern'],
                        help=f'文件匹配模式 (默认: {DEFAULT_CONFIG["file_pattern"]})')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("脚本1: JSON字段过滤器")
    print("="*60)

    process_directory(args.input, args.output, args.keep, args.pattern)


if __name__ == '__main__':
    main()