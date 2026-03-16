"""
JSON 清理模块。

从 datasets/paper_parsered 读取每篇论文的 *_content_list.json，
去除冗余字段、参考文献等后写入 datasets/input_cleaned。
供主流程（main）调用，也可单独运行。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 路径与常量
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_PARSERED_DIR = PROJECT_ROOT / "datasets" / "paper_parsered"
INPUT_CLEANED_DIR = PROJECT_ROOT / "datasets" / "input_cleaned"

DEFAULT_FIELDS_TO_REMOVE = ["bbox", "page_idx"]

# 块类型
TYPE_TEXT = "text"
TYPE_IMAGE = "image"
TYPE_TABLE = "table"
TYPE_EQUATION = "equation"
TYPE_DISCARDED = "discarded"

# 参考文献标题：仅当文本为该标题且为一级标题时，丢弃该块及之后全部内容
REFERENCES_TITLE = "References"
REFERENCES_TITLE_LEVEL = 1

# 带路径的块类型（用于过滤、合并逻辑）
BLOCK_TYPES_WITH_PATH = (TYPE_IMAGE, TYPE_EQUATION, TYPE_TABLE)


# -----------------------------------------------------------------------------
# 块级清洗（内部辅助）
# -----------------------------------------------------------------------------


def _get_str(value: Any, default: str = "") -> str:
    """安全取字符串并 strip，非字符串返回 default。"""
    if value is None:
        return default
    return value.strip() if isinstance(value, str) else default


def _get_img_path(block: Dict[str, Any]) -> str:
    """从块中取图片/公式路径，兼容 img_path / image_path。"""
    path = block.get("img_path") or block.get("image_path") or ""
    return _get_str(path)


def _is_references_heading(block: Dict[str, Any]) -> bool:
    """是否为「References」一级标题块（用于截断参考文献段）。"""
    if block.get("type") != TYPE_TEXT:
        return False
    text = _get_str(block.get("text"))
    return text == REFERENCES_TITLE and block.get("text_level") == REFERENCES_TITLE_LEVEL


def _clean_text_block(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    清洗文本块：仅保留 type / text / text_level。
    空文本返回 None（References 标题由上层在循环中截断）。
    """
    text = _get_str(block.get("text"))
    if not text:
        return None

    out: Dict[str, Any] = {"type": TYPE_TEXT, "text": text}
    level = block.get("text_level")
    if level is not None:
        out["text_level"] = level
    return out


def _clean_image_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """清洗图片块：保留 type、img_path、可选 caption。"""
    img_path = _get_img_path(block)
    caption_raw = block.get("image_caption", [])
    if isinstance(caption_raw, list):
        caption = " ".join(caption_raw).strip()
    else:
        caption = _get_str(caption_raw)

    out: Dict[str, Any] = {"type": TYPE_IMAGE, "img_path": img_path}
    if caption:
        out["caption"] = caption
    return out


def _clean_equation_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """清洗公式块：保留 type、img_path、可选 text（latex）、text_format。"""
    img_path = _get_img_path(block)
    eq_text = _get_str(block.get("text"))
    text_format = block.get("text_format")

    out: Dict[str, Any] = {"type": TYPE_EQUATION, "img_path": img_path}
    if eq_text:
        out["text"] = eq_text
    if text_format:
        out["text_format"] = text_format
    return out


def _clean_table_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """清洗表格块：仅保留 type、img_path、table_caption、table_footnote（不保留 table_body）。"""
    img_path = _get_img_path(block)
    table_caption = block.get("table_caption")
    table_footnote = block.get("table_footnote")

    out: Dict[str, Any] = {"type": TYPE_TABLE, "img_path": img_path}
    if table_caption is not None and isinstance(table_caption, list) and len(table_caption) > 0:
        out["table_caption"] = table_caption
    if table_footnote is not None and isinstance(table_footnote, list) and len(table_footnote) > 0:
        out["table_footnote"] = table_footnote
    return out


# -----------------------------------------------------------------------------
# 主清洗流程
# -----------------------------------------------------------------------------


def clean_paper_json(
    paper_data: List[Dict[str, Any]],
    fields_to_remove: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    清理论文 content_list：去字段、去 discarded、截断参考文献段。

    Args:
        paper_data: MinerU 解析的 content_list（列表，每项为块字典）
        fields_to_remove: 要从每个块中删除的字段，默认 bbox、page_idx

    Returns:
        清理后的块列表（不含 discarded，References 标题及其后内容被丢弃）
    """
    to_remove = fields_to_remove if fields_to_remove is not None else DEFAULT_FIELDS_TO_REMOVE
    cleaned: List[Dict[str, Any]] = []
    skip_rest = False

    for block in paper_data:
        if skip_rest:
            continue

        block_type = block.get("type")
        if block_type == TYPE_DISCARDED:
            continue

        # 遇到 References 一级标题：丢弃本块并截断后续
        if _is_references_heading(block):
            skip_rest = True
            continue

        cleaned_block: Optional[Dict[str, Any]] = None
        if block_type == TYPE_TEXT:
            cleaned_block = _clean_text_block(block)
        elif block_type == TYPE_IMAGE:
            cleaned_block = _clean_image_block(block)
        elif block_type == TYPE_EQUATION:
            cleaned_block = _clean_equation_block(block)
        elif block_type == TYPE_TABLE:
            cleaned_block = _clean_table_block(block)
        else:
            cleaned_block = {k: v for k, v in block.items() if k not in to_remove}

        if cleaned_block is not None:
            cleaned.append(cleaned_block)

    logger.info("清理完成：原始块数 %d -> 清理后 %d", len(paper_data), len(cleaned))
    return cleaned


def filter_empty_blocks(paper_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    过滤空块：无路径的 image/equation/table、空文本的 text 均丢弃。
    """
    def _keep(block: Dict[str, Any]) -> bool:
        t = block.get("type")
        if t in BLOCK_TYPES_WITH_PATH:
            return bool(_get_img_path(block))
        if t == TYPE_TEXT:
            return bool(_get_str(block.get("text")))
        return True

    filtered = [b for b in paper_data if _keep(b)]
    logger.info("过滤空块：%d -> %d", len(paper_data), len(filtered))
    return filtered





# -----------------------------------------------------------------------------
# 文件与批量入口
# -----------------------------------------------------------------------------


def find_content_list_files(root_dir: Optional[Path] = None) -> List[Path]:
    """
    在指定目录下递归查找所有 *_content_list.json。
    默认使用 datasets/paper_parsered。
    """
    root = root_dir or PAPER_PARSERED_DIR
    if not root.exists():
        logger.warning("目录不存在: %s", root)
        return []
    return sorted(root.rglob("*_content_list.json"))


def clean_one_file(
    input_path: Path,
    output_dir: Path = INPUT_CLEANED_DIR,
) -> Optional[Path]:
    """
    清洗单个 *_content_list.json，输出到 output_dir，文件名不变。

    Args:
        input_path: 源 JSON 路径
        output_dir: 输出目录

    Returns:
        输出文件路径；若输入不是列表格式则返回 None
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("读取失败 %s: %s", input_path, e)
        return None

    if not isinstance(data, list):
        logger.warning("跳过非列表格式: %s", input_path)
        return None

    cleaned = clean_paper_json(data)
    cleaned = filter_empty_blocks(cleaned)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=4)

    logger.info("已清洗: %s -> %s", input_path.name, output_path)
    return output_path


def clean_all_papers(
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    批量清洗：遍历 input_dir 下所有 *_content_list.json，写入 output_dir。

    Args:
        input_dir: 论文解析结果根目录，默认 datasets/paper_parsered
        output_dir: 清洗结果目录，默认 datasets/input_cleaned

    Returns:
        成功写入的输出文件路径列表
    """
    input_dir = input_dir or PAPER_PARSERED_DIR
    output_dir = output_dir or INPUT_CLEANED_DIR

    files = find_content_list_files(input_dir)
    if not files:
        logger.warning("未找到任何 *_content_list.json 文件")
        return []

    results: List[Path] = []
    for path in files:
        try:
            out = clean_one_file(path, output_dir=output_dir)
            if out is not None:
                results.append(out)
        except Exception as e:
            logger.exception("清洗失败 %s: %s", path, e)

    logger.info("共处理 %d 个文件，成功 %d 个", len(files), len(results))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    clean_all_papers()
