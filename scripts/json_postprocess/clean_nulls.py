#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本2: JSON空值清理器 (V2 - 支持文件夹批量处理)
功能: 递归删除所有为null/None/空字符串/空列表/空字典的字段
用法: python clean_nulls.py [--input 文件夹] [--output 输出文件夹]

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
    "output_dir": "/home/caep-xuben/chenchengbing/yzj/experiments/exp_0404/data/json_postprocess/clean_nulls",  # 输出文件夹路径
    "file_pattern": "*.json"             # 文件匹配模式
}
# ============================================


def remove_empty_values(obj):
    """递归删除JSON中为null、None、空字符串、空列表、空字典的字段"""
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            cleaned_value = remove_empty_values(value)
            if cleaned_value is not None and cleaned_value != "":
                if isinstance(cleaned_value, (list, dict)) and len(cleaned_value) == 0:
                    continue
                new_dict[key] = cleaned_value
        return new_dict

    elif isinstance(obj, list):
        new_list = []
        for item in obj:
            cleaned_item = remove_empty_values(item)
            if cleaned_item is not None and cleaned_item != "":
                if isinstance(cleaned_item, (list, dict)) and len(cleaned_item) == 0:
                    continue
                new_list.append(cleaned_item)
        return new_list

    else:
        return obj


def count_nodes(obj, stats=None):
    """统计JSON节点数量"""
    if stats is None:
        stats = {'total': 0, 'null': 0, 'empty_str': 0, 'empty_list': 0, 'empty_dict': 0}

    if isinstance(obj, dict):
        for k, v in obj.items():
            stats['total'] += 1
            if v is None:
                stats['null'] += 1
            elif v == "":
                stats['empty_str'] += 1
            elif isinstance(v, list) and len(v) == 0:
                stats['empty_list'] += 1
            elif isinstance(v, dict) and len(v) == 0:
                stats['empty_dict'] += 1
            else:
                count_nodes(v, stats)
    elif isinstance(obj, list):
        for item in obj:
            count_nodes(item, stats)

    return stats


def process_single_file(input_file, output_file, verbose=True):
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

    # 统计清理前
    before_stats = count_nodes(data)

    # 清理空值
    cleaned_data = remove_empty_values(data)

    # 统计清理后
    after_stats = count_nodes(cleaned_data)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    removed = before_stats['total'] - after_stats['total']

    if verbose:
        print(f"✅ [{os.path.basename(input_file)}] 节点: {before_stats['total']}→{after_stats['total']} (删除{removed})")

    return True


def process_directory(input_dir, output_dir, pattern="*.json", verbose=True):
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
    print(f"📂 输出目录: {output_dir}")
    print("="*60)

    success_count = 0
    for json_file in json_files:
        relative_path = json_file.relative_to(input_path)
        output_file = output_path / relative_path

        if process_single_file(str(json_file), str(output_file), verbose):
            success_count += 1

    print("="*60)
    print(f"\n✅ 处理完成: {success_count}/{len(json_files)} 个文件成功")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='递归删除JSON中的空值字段（支持批量处理）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
默认配置 (可在代码中修改 DEFAULT_CONFIG):
  输入文件夹: {DEFAULT_CONFIG['input_dir']}
  输出文件夹: {DEFAULT_CONFIG['output_dir']}

示例:
  # 使用代码预设值
  python clean_nulls.py

  # 命令行覆盖路径
  python clean_nulls.py --input ./data --output ./cleaned

  # 处理特定模式文件
  python clean_nulls.py -i ./data -o ./cleaned -p "*.json"
        """
    )

    parser.add_argument('--input', '-i',
                        default=DEFAULT_CONFIG['input_dir'],
                        help=f'输入文件夹路径 (默认: {DEFAULT_CONFIG["input_dir"]})')
    parser.add_argument('--output', '-o',
                        default=DEFAULT_CONFIG['output_dir'],
                        help=f'输出文件夹路径 (默认: {DEFAULT_CONFIG["output_dir"]})')
    parser.add_argument('--pattern', '-p',
                        default=DEFAULT_CONFIG['file_pattern'],
                        help=f'文件匹配模式 (默认: {DEFAULT_CONFIG["file_pattern"]})')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='静默模式，减少输出信息')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("脚本2: JSON空值清理器")
    print("="*60)

    process_directory(args.input, args.output, args.pattern, verbose=not args.quiet)


if __name__ == '__main__':
    main()