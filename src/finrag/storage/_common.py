"""Shared helpers for PostgreSQL storage adapters."""

from __future__ import annotations

from datetime import datetime, timezone

from finrag.core.node_schema import TextNode


def _serialize_node(node: TextNode) -> str:
    """
    将 TextNode 序列化为 JSON 字符串
    Args:
        node: 待序列化的文本节点
    Returns:
        节点 JSON 字符串
    """
    return node.model_dump_json()


def _deserialize_node(payload: str | bytes) -> TextNode:
    """
    将 JSON 载荷还原为 TextNode
    Args:
        payload: 字符串或字节形式的节点 JSON
    Returns:
        还原后的 TextNode
    """
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    # 将 JSON 字符串转换为 TextNode
    return TextNode.model_validate_json(payload)


def _document_record_cls():
    """
    延迟导入文档记录类以避免循环依赖
    Returns:
        DocumentRecord 类对象
    """
    from finrag.ingestion.parsers import DocumentRecord

    return DocumentRecord


def utc_now_iso() -> str:
    """
    获取当前 UTC 时间的 ISO 字符串
    Returns:
        UTC ISO 时间字符串
    """
    return datetime.now(timezone.utc).isoformat()
