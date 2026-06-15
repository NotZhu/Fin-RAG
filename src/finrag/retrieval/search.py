"""基于 LlamaIndex 的 FinRAG 检索优化"""

from __future__ import annotations

from typing import Any, Dict, Optional

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters


def build_metadata_filters(filters: Dict[str, Any]) -> Optional[MetadataFilters]:
    """
    将普通字典过滤条件转换为 LlamaIndex MetadataFilters
    Args:
        filters: metadata key 到过滤值的映射
    Returns:
        LlamaIndex MetadataFilters；无有效过滤条件时返回 None
    """
    # 过滤空值和空字符串
    filter_items = []
    # 遍历过滤条件
    for key, value in filters.items():
        if value in (None, ""):
            continue
        # 处理列表值
        if isinstance(value, list):
            # 用 IN 运滤符处理列表值
            filter_items.append(MetadataFilter(key=key, value=value, operator=FilterOperator.IN))
        else:
            # 用 EQ 运滤符处理单值
            filter_items.append(MetadataFilter(key=key, value=value))
    # 将过滤条件转换为 MetadataFilters
    return MetadataFilters(filters=filter_items) if filter_items else None


def _matches_filters(node: TextNode, filters: Dict[str, Any]) -> bool:
    """
    判断节点 metadata 是否满足过滤条件
    Args:
        node: 待判断节点
        filters: metadata 过滤条件
    Returns:
        满足全部过滤条件时返回 True
    """
    metadata = node.metadata or {}
    for key, value in filters.items():
        if value in (None, ""):
            continue
        if isinstance(value, list):
            if metadata.get(key) not in value:
                return False
        elif metadata.get(key) != value:
            return False
    return True
