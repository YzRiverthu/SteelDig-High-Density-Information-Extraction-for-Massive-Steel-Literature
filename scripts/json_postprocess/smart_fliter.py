#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本3: 智能JSON处理器 (V2 - 支持文件夹批量处理)
功能: 保留指定的一级字段（原样保留），对其他字段清理空值后保留
用法: python smart_filter.py [--input 文件夹] [--output 输出文件夹] [--keep 字段1 字段2 ...]

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
    "input_dir": "/home/caep-xuben/chenchengbing/yzj/datasets/llm_baseline_366/final_json/json",        # 输入文件夹路径
    "output_dir": "/home/caep-xuben/chenchengbing/yzj/experiments/exp_0404/data/json_postprocess/smart_fliter",       # 输出文件夹路径
    "keep_fields": ["papers", "alloys", "processes", "samples", "processing_steps", "structures", "properties"],   # 要完整保留的一级字段
    "file_pattern": "*.json"               # 文件匹配模式
}
# ============================================


def remove_nulls_from_value(value):
    """从值中递归删除null/None/空字符串/空列表/空字典"""
    if isinstance(value, list):
        cleaned_list = []
        for item in value:
            if item is None:
                continue
            cleaned_item = remove_nulls_from_value(item)
            if cleaned_item is not None and cleaned_item != {} and cleaned_item != [] and cleaned_item != "":
                cleaned_list.append(cleaned_item)
        return cleaned_list if cleaned_list else None

    elif isinstance(value, dict):
        cleaned_dict = {}
        for k, v in value.items():
            if v is None or v == [] or v == {} or v == "":
                continue
            cleaned_v = remove_nulls_from_value(v)
            if cleaned_v is not None and cleaned_v != [] and cleaned_v != {} and cleaned_v != "":
                cleaned_dict[k] = cleaned_v
        return cleaned_dict if cleaned_dict else None

    else:
        return value


def process_single_file(input_file, output_file, keep_fields, verbose=True):
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

    result = {}
    stats = {'kept_complete': 0, 'cleaned_kept': 0, 'removed_empty': 0}

    for key, value in data.items():
        if key in keep_fields:
            result[key] = value
            stats['kept_complete'] += 1
        else:
            cleaned_value = remove_nulls_from_value(value)
            if cleaned_value is not None and cleaned_value != [] and cleaned_value != {}:
                result[key] = cleaned_value
                stats['cleaned_kept'] += 1
            else:
                stats['removed_empty'] += 1

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"✅ [{os.path.basename(input_file)}] 保留原样:{stats['kept_complete']} 清理保留:{stats['cleaned_kept']} 删除:{stats['removed_empty']}")

    return True


def process_directory(input_dir, output_dir, keep_fields, pattern="*.json", verbose=True):
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
    print(f"📋 完整保留字段: {keep_fields}")
    print(f"📂 输出目录: {output_dir}")
    print("="*60)

    success_count = 0
    for json_file in json_files:
        relative_path = json_file.relative_to(input_path)
        output_file = output_path / relative_path

        if process_single_file(str(json_file), str(output_file), keep_fields, verbose):
            success_count += 1

    print("="*60)
    print(f"\n✅ 处理完成: {success_count}/{len(json_files)} 个文件成功")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="""智能JSON处理器 (支持批量处理):
        1. 对指定字段: 完整保留（不删除内部空值）
        2. 对未指定字段: 递归删除null/空值，若清理后为空则删除整个字段""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
默认配置 (可在代码中修改 DEFAULT_CONFIG):
  输入文件夹: {DEFAULT_CONFIG['input_dir']}
  输出文件夹: {DEFAULT_CONFIG['output_dir']}
  保留字段: {DEFAULT_CONFIG['keep_fields']}

示例:
  # 使用代码预设值
  python smart_filter.py

  # 命令行覆盖输入输出路径
  python smart_filter.py --input ./data --output ./result

  # 命令行覆盖保留字段
  python smart_filter.py --keep papers alloys samples

  # 完整命令行指定
  python smart_filter.py -i ./input -o ./output -k papers alloys
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
                        help=f'要完整保留的一级字段 (默认: {DEFAULT_CONFIG["keep_fields"]})')
    parser.add_argument('--pattern', '-p',
                        default=DEFAULT_CONFIG['file_pattern'],
                        help=f'文件匹配模式 (默认: {DEFAULT_CONFIG["file_pattern"]})')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='静默模式，减少输出信息')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("脚本3: 智能JSON处理器")
    print("="*60)

    process_directory(args.input, args.output, args.keep, args.pattern, verbose=not args.quiet)


if __name__ == '__main__':
    main()