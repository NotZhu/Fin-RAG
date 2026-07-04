"""基于 Docling 的文档解析适配器"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from llama_index.core import Document

from finrag.ingestion.metadata import build_base_metadata

logger = logging.getLogger(__name__)


def load_docling_documents(
    path: str | Path,
    *,
    knowledge_base_id: str,
    data_root: Optional[Path] = None,
) -> list[Document]:
    """
    使用 Docling JSON 解析单个支持格式文件，并转换为 LlamaIndex Document
    Args:
        path: 待解析文件路径
        knowledge_base_id: 资料库 ID
        data_root: 可选数据根目录，用于生成稳定相对路径
    Returns:
        解析得到的 LlamaIndex Document 列表；解析失败或无文本时返回空列表
    """
    source = Path(path)
    try:
        reader = _make_docling_reader()
        raw_documents = reader.load_data(str(source))
        documents: list[Document] = []
        for raw_document in raw_documents:
            text = _document_text(raw_document).strip()
            if not text:
                continue
            metadata = build_base_metadata(
                source,
                text,
                source.suffix.lower().lstrip("."),
                knowledge_base_id=knowledge_base_id,
                data_root=data_root,
            )
            metadata["parser_name"] = "docling"
            documents.append(Document(text=text, metadata=metadata))
        return documents
    except Exception as exc:
        logger.warning("Docling 解析失败: %s, error=%s", source, exc)
        return []


def _make_docling_reader() -> Any:
    """延迟创建 DoclingReader，避免导入阶段强依赖扩展包"""
    try:
        from llama_index.readers.docling import DoclingReader
    except Exception as exc:
        raise RuntimeError("Docling JSON 解析需要安装 llama-index-readers-docling 依赖") from exc
    export_type = getattr(getattr(DoclingReader, "ExportType", object), "JSON", "json")
    return DoclingReader(export_type=export_type)


def _document_text(document: Any) -> str:
    """读取 LlamaIndex Document 文本"""
    text = getattr(document, "text", None)
    if text is not None:
        return str(text or "")
    getter = getattr(document, "get_content", None)
    if callable(getter):
        return str(getter() or "")
    return str(document or "")
