"""FinRAG 运行时使用的存储协议"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, runtime_checkable

from finrag.core.node_schema import TextNode


@dataclass(frozen=True) # 冻结数据类，防止修改
class SparseVector:
    """BM25 稀疏向量载荷"""

    indices: List[int] # 稀疏向量非零维度下标
    values: List[float] # 稀疏向量非零维度权重
    token_count: int # 构建向量时使用的 token 数量


@runtime_checkable # 运行时可检查协议，确保实现符合协议
class DocumentRegistryStore(Protocol):
    """文档生命周期注册表存储协议"""

    records: Dict[str, Any] # 以 document_id 为 key 的文档记录缓存

    def list_public(self, knowledge_base_id: Optional[str] = None) -> List[dict]:
        """列出可对外展示的文档记录"""
        ...

    def get(self, document_id: str) -> Any:
        """按文档 ID 获取注册记录"""
        ...

    def find_by_hash(self, content_hash: str, knowledge_base_id: str) -> Optional[Any]:
        """按内容哈希和资料库 ID 查找已有文档"""
        ...

    def upsert_uploaded(
        self,
        *,
        source_path: Path,
        filename: str,
        file_type: str,
        content_hash: str,
        knowledge_base_id: str,
    ) -> Any:
        """插入或复用上传文档记录"""
        ...

    def update_source_path(self, document_id: str, source_path: str) -> None:
        """更新单个文档的源文件路径"""
        ...

    def mark_parsing(self, document_id: str) -> None:
        """将文档标记为解析中"""
        ...

    def mark_indexed(self, document_id: str, chunk_count: int) -> None:
        """将文档标记为已索引并记录叶子分块数量"""
        ...

    def mark_failed(self, document_id: str, error: str) -> None:
        """将文档标记为失败并保存错误信息"""
        ...

    def mark_deleted(self, document_id: str) -> dict:
        """将文档标记为已删除并返回公开状态"""
        ...


@runtime_checkable
class NodeStore(Protocol):
    """层级 TextNode 持久化存储协议"""

    def replace_document_nodes(self, document_id: str, nodes: List[TextNode], knowledge_base_id: str) -> None:
        """替换单个文档的全部层级节点"""
        ...

    def load_all_nodes(self, knowledge_base_id: str) -> List[TextNode]:
        """加载指定知识库的全部层级节点"""
        ...

    def load_leaf_nodes(self, knowledge_base_id: str) -> List[TextNode]:
        """加载指定知识库的所有叶子节点"""
        ...

    def get_node(self, node_id: str) -> Optional[TextNode]:
        """按节点 ID 获取节点"""
        ...

    def delete_document(self, document_id: str, knowledge_base_id: str) -> None:
        """删除指定知识库中单个文档的全部节点"""
        ...

    def clear(self) -> None:
        """清空全部节点"""
        ...

    def count_leaf_nodes(self, document_id: str) -> int:
        """统计单个文档的叶子节点数量"""
        ...


@runtime_checkable
class BM25StateStore(Protocol):
    """BM25 词项和分块词频状态存储协议"""

    def replace_document_chunks(self, knowledge_base_id: str, document_id: str, chunk_token_counts: Dict[str, Dict[str, int]]) -> None:
        """替换单个文档各分块的词频"""
        ...

    def delete_document(self, knowledge_base_id: str, document_id: str) -> None:
        """删除单个文档的 BM25 状态"""
        ...

    def clear(self, knowledge_base_id: str) -> None:
        """清空指定知识库的 BM25 状态"""
        ...

    def build_query_sparse_vector(self, knowledge_base_id: str, tokens: Iterable[str]) -> SparseVector:
        """根据 query token 序列构建稀疏向量"""
        ...

    def build_document_sparse_vector(self, knowledge_base_id: str, tokens: Iterable[str]) -> SparseVector:
        """根据 document token 序列构建 BM25 稀疏向量"""
        ...


@runtime_checkable
class IndexManifestStore(Protocol):
    """索引清单持久化存储协议"""

    def save_manifest(self, manifest: Dict, knowledge_base_id: str) -> None:
        """保存当前索引清单"""
        ...

    def load_manifest(self, knowledge_base_id: str) -> Optional[Dict]:
        """加载当前索引清单"""
        ...
