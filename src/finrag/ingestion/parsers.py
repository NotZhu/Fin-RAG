"""基于 Docling 的 FinRAG 文档解析入口"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from llama_index.core import Document

from finrag.ingestion.docling_loader import load_docling_documents
from finrag.ingestion.metadata import (
    build_base_metadata,
    compute_content_hash,
    is_path_within,
    normalize_text,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

# 公共解析入口，所有支持格式统一交给 Docling
__all__ = [
    "DocumentRecord",
    "SUPPORTED_SUFFIXES",
    "build_base_metadata",
    "compute_content_hash",
    "is_path_within",
    "load_docling_documents",
    "load_documents",
    "normalize_text",
    "utc_now_iso",
]

# 企业级 RAG 常见文档白名单，所有格式统一交给 Docling 解析。
# 仅开放常见知识库资料和 OCR 图片；模板、宏文件、音视频和长尾出版格式不作为默认上传入口。
SUPPORTED_SUFFIXES = {
    # PDF / Office
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".csv",
    # Markdown / plain text / web
    ".md",
    ".txt",
    ".html",
    ".htm",
    ".json",
    # OCR / scanned images
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}


def load_documents(
    data_path: str | Path,
    *,
    knowledge_base_id: str = "default",
    document_registry: Optional[object] = None,
) -> list[Document]:
    """
    从数据目录或文档注册表加载支持格式文档
    Args:
        data_path: 数据根目录
        knowledge_base_id: 默认资料库 ID
        document_registry: 可选文档生命周期注册表
    Returns:
        Docling 解析得到的 LlamaIndex Document 列表
    """
    root = Path(data_path)
    if not root.exists():
        logger.warning("文档目录不存在: %s", root)
        return []

    docs: list[Document] = []
    if document_registry is not None:
        records = []
        for record in document_registry.records.values():
            path = Path(record.source_path)
            # 仅加载当前知识库且未删除的注册文档
            if record.knowledge_base_id != knowledge_base_id:
                continue
            if record.status == "deleted" or not path.exists():
                continue
            # 跳过可信数据根目录外的源文件
            if not is_path_within(path, root):
                logger.warning("跳过可信目录外的注册文档: %s", path)
                continue
            records.append(record)

        # 按上传时间排序，保证增量重建和全量加载顺序稳定
        for record in sorted(records, key=lambda item: item.upload_time):
            path = Path(record.source_path)
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                parsed_docs = load_docling_documents(
                    path,
                    knowledge_base_id=record.knowledge_base_id,
                    data_root=root,
                )
                for doc in parsed_docs:
                    # 注册表中的生命周期元数据优先于解析器生成值
                    doc.metadata.update(
                        {
                            "knowledge_base_id": record.knowledge_base_id,
                            "document_id": record.document_id,
                            "filename": record.filename,
                            "file_type": record.file_type,
                        }
                    )
                docs.extend(parsed_docs)
        return docs

    # 无注册表时按文件系统递归加载，仍只处理支持后缀
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            docs.extend(
                load_docling_documents(
                    path,
                    knowledge_base_id=knowledge_base_id,
                    data_root=root,
                )
            )
    return docs


@dataclass
class DocumentRecord:
    """文档生命周期注册记录"""

    document_id: str
    source_path: str
    filename: str
    file_type: str
    content_hash: str
    knowledge_base_id: str
    status: str = "uploaded"
    chunk_count: int = 0
    upload_time: str = ""
    last_error: Optional[str] = None

    def to_dict(self) -> dict:
        """
        将文档记录转换为可 JSON 序列化字典
        Returns:
            文档生命周期记录字典
        """
        return asdict(self)
