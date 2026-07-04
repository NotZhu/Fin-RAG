"""FinRAG 文档解析元数据工具"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def utc_now_iso() -> str:
    """
    获取当前 UTC 时间的 ISO 字符串
    Returns:
        带时区信息的 ISO 时间字符串
    """
    return datetime.now(timezone.utc).isoformat()


def compute_content_hash(path: Path) -> str:
    """
    计算文件内容 SHA256 哈希，用于文档去重
    Args:
        path: 待计算文件路径
    Returns:
        sha256: 前缀的内容哈希
    """
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_text(text: str) -> str:
    """
    规范化文本空白，同时保留段落分隔
    Args:
        text: 原始文本
    Returns:
        清洗后的文本
    """
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    lines: list[str] = []
    blank = False # 上一行是否为空行
    for raw_line in normalized.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip() # 统一连续空白
        if line:
            lines.append(line)
            blank = False
        elif not blank:
            lines.append("")
            blank = True
    return "\n".join(lines).strip()


def relative_path(path: Path, data_root: Optional[Path]) -> str:
    """
    计算文件相对数据根目录的 POSIX 路径
    Args:
        path: 文件路径
        data_root: 数据根目录
    Returns:
        相对路径；不在数据根目录内时返回绝对路径
    """
    if data_root is not None:
        try:
            return path.resolve().relative_to(data_root.resolve()).as_posix()
        except ValueError:
            pass
    return path.resolve().as_posix()


def is_path_within(path: Path, root: Path) -> bool:
    """
    判断路径是否位于指定根目录内
    Args:
        path: 待判断路径
        root: 根目录
    Returns:
        路径位于根目录内时返回 True
    """
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def build_base_metadata(
    path: Path,
    text: str,
    file_type: str,
    *,
    knowledge_base_id: str = "default",
    data_root: Optional[Path] = None,
) -> dict:
    """
    为解析出的 LlamaIndex Document 构造统一基础元数据
    Args:
        path: 源文件路径
        text: 文档文本
        file_type: 文件类型
        knowledge_base_id: 资料库 ID
        data_root: 可选数据根目录
    Returns:
        包含 document_id、文件名和类型的元数据字典
    """
    content_hash = compute_content_hash(path)
    document_seed = f"{knowledge_base_id}:{relative_path(path, data_root)}:{content_hash}"
    # 使用文档作用域和内容哈希生成稳定 document_id
    document_id = hashlib.md5(document_seed.encode("utf-8")).hexdigest()
    return {
        "knowledge_base_id": knowledge_base_id,
        "document_id": document_id,
        "source_path": path.resolve().as_posix(),
        "filename": path.name,
        "file_type": file_type,
    }
