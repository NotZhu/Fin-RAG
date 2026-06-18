"""Knowledge base scope helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finrag.core.config import validate_knowledge_base_id


@dataclass(frozen=True)
class KnowledgeBaseScope:
    """知识库作用域信息"""
    
    # 知识库 ID
    knowledge_base_id: str
    # 知识库集合名称
    collection_name: str
    # 知识库数据根目录
    source_root: Path
    # 知识库元数据键
    manifest_key: str
    # 运行时缓存数据键
    runtime_cache_key: str

    @classmethod
    def from_config(cls, config: Any, knowledge_base_id: str) -> "KnowledgeBaseScope":
        """
        从配置创建知识库作用域
        Args:
            config: 应用配置
            knowledge_base_id: 知识库 ID
        Returns:
            知识库作用域助手实例
        """
        # 验证知识库 ID
        resolved_id = validate_knowledge_base_id(knowledge_base_id)
        # 从配置中获取知识库集合名称
        collection_base = str(getattr(config, "milvus_collection", "") or "finrag_leaf_nodes").strip()
        # 从知识库 ID 中提取安全的后缀
        safe_suffix = re.sub(r"[^A-Za-z0-9_]+", "_", resolved_id).strip("_") or "default"
        return cls(
            knowledge_base_id=resolved_id,
            collection_name=f"{collection_base}__kb_{safe_suffix}",
            source_root=Path(config.data_path) / resolved_id,
            manifest_key=resolved_id,
            runtime_cache_key=resolved_id,
        )
