"""Milvus 索引构建模块"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = 3 # 当前索引 manifest schema 版本
INDEX_TYPE = "LlamaIndexRouter" # 当前索引类型标识
MILVUS_COLLECTION_NAME = "finrag_leaf_nodes" # 默认 Milvus collection 名称
SPARSE_PROVIDER = "milvus_builtin_bm25" # Sparse 向量生成提供方
DOCLING_NODE_STRUCTURE = {
    "parser": "docling_node_parser",
    "hierarchy": ["document", "section", "leaf"],
    "indexed_levels": ["leaf"],
}

PRIMARY_ID_FIELD = "id" # Milvus 主键字段名
DOC_ID_FIELD = "document_id" # 文档 ID 元数据字段名
DENSE_EMBEDDING_FIELD = "dense_embedding" # Dense embedding 向量字段名
TEXT_FIELD = "text" # 节点文本字段名
SPARSE_EMBEDDING_FIELD = "sparse_embedding" # Sparse embedding 向量字段名
LLAMA_NODE_CONTENT_FIELD = "_node_content" # 节点序列化内容字段名
LLAMA_NODE_TYPE_FIELD = "_node_type" # 节点类型字段名
SCALAR_FIELD_NAMES = [ # 写入 Milvus 的标量元数据字段
    "chunk_id",
    "document_id",
    "knowledge_base_id",
    "filename",
    "file_type",
    "page_number",
    "chunk_level",
    "chunk_idx",
    "parent_chunk_id",
    "root_chunk_id",
]
SCALAR_FIELD_TYPE_NAMES = [ # 标量元数据字段对应的 Milvus 类型
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "VARCHAR",
    "INT64",
    "INT64",
    "INT64",
    "VARCHAR",
    "VARCHAR",
]
MILVUS_AUTO_FIELD_NAMES = {DOC_ID_FIELD} # Milvus 自动维护或不可直接写入的字段
MILVUS_DYNAMIC_METADATA_FIELD_NAMES = {"page_number"} # 允许作为动态 metadata 写入的字段
MILVUS_OUTPUT_FIELD_NAMES = [ # Milvus 查询结果需要返回的字段
    *SCALAR_FIELD_NAMES,
    LLAMA_NODE_CONTENT_FIELD,
    LLAMA_NODE_TYPE_FIELD,
    TEXT_FIELD,
]
MANIFEST_COMPARE_KEYS = [ # 判断现有索引是否可复用时参与比较的 manifest 字段
    "schema_version",
    "index_type",
    "knowledge_base_id",
    "embedding",
    "milvus",
    "node_structure",
]
DENSE_INDEX_CONFIG = { # Dense 向量索引配置
    "index_type": "HNSW", # 使用 HNSW 索引类型
    "metric_type": "IP", # 使用内积（IP）作为相似度度量
    "params": {"M": 16, "efConstruction": 200}, # HNSW 索引参数
}
DENSE_SEARCH_CONFIG = { # Dense 向量检索配置
    "ef": 64,
}
SPARSE_INDEX_CONFIG = { # Sparse 向量索引配置
    "index_type": "SPARSE_INVERTED_INDEX",
    "metric_type": "BM25", # Milvus 2.5 built-in BM25 function 对应的 sparse index 度量
}
KNOWN_EMBEDDING_DIMENSIONS = { # 常用 embedding 模型到向量维度的映射
    "BAAI/bge-m3": 1024,
}
EMBED_BATCH_SIZE = 10 # embedding 批量请求大小


class OpenAICompatibleEmbedding(BaseEmbedding):
    """OpenAI-compatible embedding endpoint 的 LlamaIndex embedding 模型"""
    api_key: str = Field(default="", exclude=True)
    base_url: str = ""
    embed_dim: Optional[int] = None
    timeout: float = 60.0

    def _embedding_endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/embeddings"

    def _request_embeddings(self, texts: List[str]) -> List[List[float]]:
        payload = json.dumps({"model": self.model_name, "input": texts}).encode("utf-8")
        request = urlrequest.Request(
            self._embedding_endpoint(),
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible embedding 请求失败: HTTP {exc.code} {body}") from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"OpenAI-compatible embedding 请求失败: {exc}") from exc
        try:
            data = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("OpenAI-compatible embedding 返回了非法 JSON") from exc
        rows = data.get("data")
        if not isinstance(rows, list):
            raise RuntimeError("OpenAI-compatible embedding 响应缺少 data 列表")
        rows = sorted(rows, key=lambda row: int(row.get("index", 0)) if isinstance(row, dict) else 0)
        embeddings: List[List[float]] = []
        for row in rows:
            embedding = row.get("embedding") if isinstance(row, dict) else None
            if not isinstance(embedding, list) or not embedding:
                raise RuntimeError("OpenAI-compatible embedding 响应缺少有效 embedding")
            embeddings.append([float(value) for value in embedding])
        if len(embeddings) != len(texts):
            raise RuntimeError("OpenAI-compatible embedding 返回数量与输入数量不一致")
        return embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._request_embeddings([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._request_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._request_embeddings(texts)


class IndexConstructionModule:
    """构建和加载 Milvus 索引"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        collection_name: str = MILVUS_COLLECTION_NAME,
        milvus_uri: Optional[str] = None,
        milvus_host: str = "localhost",
        milvus_port: int = 19530,
        embedding_base_url: str = "",
        embedding_api_key: str = "",
        embed_model: Optional[BaseEmbedding] = None,
        manifest_store: Optional[Any] = None,
        enable_sparse: bool = True,
        rrf_k: int = 60,
    ):
        """
        初始化 Milvus 索引构建模块
        Args:
            model_name: embedding 模型名称
            collection_name: Milvus collection 名称
            milvus_uri: 可选 Milvus URI，未提供时由 host 和 port 生成
            milvus_host: Milvus 主机名
            milvus_port: Milvus 端口
            embedding_base_url: OpenAI 兼容 embedding endpoint
            embedding_api_key: OpenAI 兼容 embedding API key
            embed_model: 可直接注入的 LlamaIndex embedding 模型
            manifest_store: 索引清单持久化存储
            enable_sparse: 是否启用 Milvus 内置 BM25 sparse 向量
            rrf_k: 混合排序的 RRF 参数
        """
        self.model_name = model_name # 嵌入模型名称
        self.collection_name = collection_name # Milvus collection 名称
        self.milvus_uri = milvus_uri or f"http://{milvus_host}:{int(milvus_port)}" # Milvus URI
        self.manifest_store = manifest_store # 索引清单持久化存储
        self.embedding_base_url = str(embedding_base_url or os.getenv("EMBEDDING_BASE_URL", "")).strip()
        self.embedding_api_key = str(embedding_api_key or os.getenv("EMBEDDING_API_KEY", "")).strip()
        self.enable_sparse = bool(enable_sparse) # 是否启用 Milvus 内置 BM25 sparse 向量
        self.sparse_embedding_function = self._build_sparse_embedding_function() if self.enable_sparse else None # Milvus 内置 BM25 sparse 向量函数
        self.rrf_k = int(rrf_k) # RRF 算法参数
        self.embed_model = embed_model or self._build_embedding_model() # LlamaIndex embedding 模型实例
        self.embedding_dimensions = self._infer_embedding_dimensions(self.embed_model) or self._known_embedding_dimensions(self.model_name) # 向量维度
        if self.embedding_dimensions is None:
            raise RuntimeError(f"无法推断模型 {self.model_name!r} 的 embedding 维度")
        self.vector_store: Optional[BasePydanticVectorStore] = None # 向量存储实例
        self.storage_context: Optional[StorageContext] = None # 存储上下文实例
        self.index: Optional[VectorStoreIndex] = None # 索引实例实例

    @staticmethod
    def _build_sparse_embedding_function() -> BM25BuiltInFunction:
        """构建 Milvus 内置 BM25 sparse 向量函数"""
        return BM25BuiltInFunction(
            input_field_names=TEXT_FIELD,
            output_field_names=SPARSE_EMBEDDING_FIELD,
        )

    def _build_embedding_model(self) -> BaseEmbedding:
        """
        根据 OpenAI 兼容配置创建 embedding 模型
        Returns:
            LlamaIndex embedding 模型实例
        """
        if not self.embedding_base_url or not self.embedding_api_key:
            raise RuntimeError(
                "embedding_model="
                f"{self.model_name!r} 需要配置 EMBEDDING_BASE_URL 和 EMBEDDING_API_KEY"
            )
        return OpenAICompatibleEmbedding(
            model_name=self.model_name,
            api_key=self.embedding_api_key,
            base_url=self.embedding_base_url,
            embed_dim=self._known_embedding_dimensions(self.model_name),
            embed_batch_size=EMBED_BATCH_SIZE,
        )

    @staticmethod
    def _infer_embedding_dimensions(embed_model: BaseEmbedding) -> Optional[int]:
        """
        从 embedding 模型属性中推断向量维度
        Args:
            embed_model: LlamaIndex embedding 模型实例
        Returns:
            维度整数，无法推断时返回 None
        """
        for attr_name in ("embed_dim", "embedding_dimensions", "dimensions"):
            value = getattr(embed_model, attr_name, None)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _known_embedding_dimensions(model_name: str) -> Optional[int]:
        """
        返回 LlamaIndex wrapper 不暴露维度属性时的已知模型默认维度
        Args:
            model_name: embedding 模型名称
        Returns:
            默认维度，未知模型返回 None
        """
        return KNOWN_EMBEDDING_DIMENSIONS.get(str(model_name))

    def build_manifest(
        self,
        *,
        knowledge_base_id: str = "",
        index_ids: list[str] | None = None,
        document_count: int = 0,
        node_count: int = 0,
        last_persist_ms: float = 0.0,
    ) -> Dict[str, Any]:
        """构建当前索引配置清单"""
        manifest: Dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "index_type": INDEX_TYPE,
            "knowledge_base_id": str(knowledge_base_id or ""),
            "index_ids": list(index_ids or ["finrag-auto-merge"]),
            "embedding": {
                "model": self.model_name,
                "dimensions": self.embedding_dimensions,
            },
            "milvus": {
                "collection": self.collection_name,
                "dense_embedding_field": DENSE_EMBEDDING_FIELD,
                "sparse_enabled": self.enable_sparse,
                "sparse_provider": SPARSE_PROVIDER if self.enable_sparse else None,
                "scalar_fields": _milvus_scalar_field_names(),
                "dense_index": dict(DENSE_INDEX_CONFIG),
                "sparse_index": dict(SPARSE_INDEX_CONFIG) if self.enable_sparse else None,
                "rrf_k": self.rrf_k,
            },
            "node_structure": dict(DOCLING_NODE_STRUCTURE),
            "document_count": int(document_count),
            "node_count": int(node_count),
            "last_persist_ms": round(float(last_persist_ms), 2),
        }
        return manifest

    def load_manifest(self, knowledge_base_id: str) -> Optional[Dict[str, Any]]:
        """
        从持久化存储读取索引清单
        Args:
            knowledge_base_id: 知识库 ID
        Returns:
            索引清单字典，读取失败或不存在时返回 None
        """
        if self.manifest_store is not None:
            try:
                return self.manifest_store.load_manifest(knowledge_base_id)
            except Exception as exc:
                logger.warning("读取 PostgreSQL 索引清单失败: %s", exc)
                return None
        return None

    def save_manifest(self, manifest: Dict[str, Any]) -> None:
        """
        保存索引清单到持久化存储
        Args:
            manifest: 待保存的索引清单字典
        """
        if self.manifest_store is not None:
            self.manifest_store.save_manifest(manifest, str(manifest.get("knowledge_base_id") or ""))
            return
        raise RuntimeError("持久化 manifest 需要 PostgreSQL index manifest store")

    def manifest_matches(self, expected_manifest: Dict[str, Any]) -> bool:
        """
        判断已保存清单是否匹配预期清单
        Args:
            expected_manifest: 由当前配置构建的预期清单
        Returns:
            匹配时返回 True，否则返回 False
        """
        knowledge_base_id = str(expected_manifest.get("knowledge_base_id") or "")
        actual = self.load_manifest(knowledge_base_id)
        if not actual:
            return False
        return all(actual.get(key) == expected_manifest.get(key) for key in MANIFEST_COMPARE_KEYS)

    def init_collection(self, *, reset: bool = False) -> BasePydanticVectorStore:
        """
        初始化或重建 Milvus 向量存储
        Args:
            reset: 是否覆盖重建 collection
        Returns:
            当前向量存储实例
        """
        # 初始化或重建 Milvus 向量存储
        self.vector_store = self._build_vector_store(reset=reset)
        return self.vector_store

    def load_index(
        self,
        expected_manifest: Optional[Dict[str, Any]] = None,
        *,
        storage_context: Optional[StorageContext] = None,
    ) -> Optional[VectorStoreIndex]:
        """
        从已有向量存储加载 LlamaIndex 索引句柄
        Args:
            expected_manifest: 可选预期清单，不匹配时拒绝加载
            storage_context: 可选外部存储上下文
        Returns:
            加载成功的 VectorStoreIndex，不匹配时返回 None
        """
        # 检查清单清单是否匹配预期清单
        if expected_manifest is not None:
            knowledge_base_id = str(expected_manifest.get("knowledge_base_id") or "")
            if self.load_manifest(knowledge_base_id) is not None and not self.manifest_matches(expected_manifest):
                return None
        if self.vector_store is None:
            self.init_collection(reset=False)
        # 基于当前向量存储和外部存储上下文创建存储上下文 StorageContext
        self.storage_context = self._storage_context_with_vector_store(storage_context)
        # 从已有的 Milvus 向量库对象恢复一个 LlamaIndex 索引句柄
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store, # 当前向量存储实例
            storage_context=self.storage_context, # 存储上下文实例
            embed_model=self.embed_model, # 嵌入模型实例
        )
        return self.index

    def build_vector_index(
        self,
        chunks: List[TextNode],
        *,
        storage_context: Optional[StorageContext] = None,
        reset: bool = True,
    ) -> VectorStoreIndex:
        """
        使用叶子节点重建向量索引
        Args:
            chunks: 待写入向量索引的叶子节点
            storage_context: 可选外部存储上下文
            reset: 是否重建底层 collection
        Returns:
            新构建的 VectorStoreIndex
        """
        # 初始化或重建 Milvus 向量存储
        self.init_collection(reset=reset)
        # 构建存储上下文，根据外部存储上下文和当前向量存储
        self.storage_context = self._storage_context_with_vector_store(storage_context)
        # 构建向量索引
        self.index = VectorStoreIndex(
            chunks,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
        )
        return self.index

    def delete_vectors_by_document_id(self, document_id: str) -> None:
        """
        删除指定文档的向量记录
        Args:
            document_id: 待删除的文档 ID
        """
        # 确保向量存储已初始化
        self._ensure_vector_store()
        # 构建删除过滤器，根据文档 ID 匹配所有向量记录
        filters = MetadataFilters(
            # 只匹配 metadata 里 DOC_ID_FIELD 等于当前 document_id 的向量记录
            filters=[
                MetadataFilter(key=DOC_ID_FIELD, value=document_id)
            ]
        )
        try:
            # 删除所有匹配的向量记录
            self.vector_store.delete_nodes(filters=filters)
        except NotImplementedError:
            self.vector_store.delete(document_id)

    def upsert_leaf_nodes(self, leaf_nodes: List[TextNode], *, storage_context: Optional[StorageContext] = None) -> VectorStoreIndex:
        """
        增量写入叶子节点并刷新索引句柄
        Args:
            leaf_nodes: 待写入的叶子节点列表
            storage_context: 可选外部存储上下文
        Returns:
            刷新后的 VectorStoreIndex
        """
        self._ensure_vector_store()
        # 构建存储上下文，根据外部存储上下文和当前向量存储
        self.storage_context = self._storage_context_with_vector_store(storage_context)
        # 从叶子节点中提取所有文档 ID
        document_ids = {str(node.metadata.get(DOC_ID_FIELD)) for node in leaf_nodes if node.metadata.get(DOC_ID_FIELD)}
        # 删除所有旧向量索引
        for document_id in document_ids:
            self.delete_vectors_by_document_id(document_id)
        if leaf_nodes:
            # 构建新索引句柄
            VectorStoreIndex(
                leaf_nodes,
                storage_context=self.storage_context,
                embed_model=self.embed_model,
            )
        # 从已有的 vector_store 构造一个索引对象
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
        )
        return self.index

    def clear_index(self, *, storage_context: Optional[StorageContext] = None) -> VectorStoreIndex:
        """
        清空向量索引并返回空索引句柄
        Args:
            storage_context: 可选外部存储上下文
        Returns:
            清空后的 VectorStoreIndex
        """
        # 重置索引状态，确保从新构建
        self.init_collection(reset=True)
        # 从空向量存储创建新索引句柄
        self.storage_context = self._storage_context_with_vector_store(storage_context)
        # 创建新索引句柄
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store, # 向量存储实例
            storage_context=self.storage_context, # 存储上下文实例
            embed_model=self.embed_model, # 嵌入模型实例
        )
        return self.index

    def _build_vector_store(self, *, reset: bool) -> BasePydanticVectorStore:
        """
        构造 Milvus 或内存向量存储实例
        Args:
            reset: 是否覆盖重建 collection
        Returns:
            LlamaIndex 向量存储实例
        """
        # 获取 Milvus scalar 字段定义列表
        scalar_fields = _milvus_scalar_field_specs()
        # 构建 Milvus 向量存储参数字典
        kwargs = {
            "uri": self.milvus_uri, # Milvus URI
            "collection_name": self.collection_name, # Milvus collection 名称
            "overwrite": bool(reset), # 是否覆盖重建 collection
            "upsert_mode": True, # 启用 upsert 模式以支持增量更新
            "enable_dense": True, # 启用密集向量索引
            "enable_sparse": self.enable_sparse, # 是否启用稀疏向量索引
            "dim": self.embedding_dimensions, # Dense embedding 维度
            "embedding_field": DENSE_EMBEDDING_FIELD, # Dense embedding 字段名
            "sparse_embedding_field": SPARSE_EMBEDDING_FIELD, # Sparse embedding 字段名
            "sparse_embedding_function": self.sparse_embedding_function, # Sparse embedding 嵌入函数
            "doc_id_field": DOC_ID_FIELD, # 文档 ID 字段名
            "text_key": TEXT_FIELD, # 文本字段名
            "similarity_metric": "IP", # 相似度度量
            "index_config": dict(DENSE_INDEX_CONFIG), # 密集向量索引配置
            "sparse_index_config": dict(SPARSE_INDEX_CONFIG), # 稀疏向量索引配置
            "search_config": dict(DENSE_SEARCH_CONFIG), # 密集向量搜索配置
            "hybrid_ranker": "RRFRanker", # 混合排名器
            "hybrid_ranker_params": {"k": self.rrf_k}, # RRF 混合排名参数
            "scalar_field_names": [field_name for field_name, _ in scalar_fields], # scalar 字段名列表
            "scalar_field_types": [field_type for _, field_type in scalar_fields], # scalar 字段类型列表
            "output_fields": list(MILVUS_OUTPUT_FIELD_NAMES), # 搜索结果中返回的字段列表
        }
        # 加载 Milvus 向量存储类 MilvusVectorStore
        vector_store_cls = _load_milvus_vector_store_class()
        # 确保当前线程拥有 asyncio event loop
        _ensure_event_loop_for_current_thread()
        # 返回 MilvusVectorStore 实例对象
        return vector_store_cls(**kwargs)

    def _ensure_vector_store(self) -> None:
        """
        确保向量存储实例已初始化
        """
        if self.vector_store is None:
            self.init_collection(reset=False)

    def _storage_context_with_vector_store(self, storage_context: Optional[StorageContext]) -> StorageContext:
        """
        基于当前向量存储创建 LlamaIndex 存储上下文
        Args:
            storage_context: 可选外部存储上下文，用于复用 docstore
        Returns:
            绑定当前向量存储的 StorageContext
        """
        # 从外部存储上下文获取文档存储实例
        docstore: Optional[BaseDocumentStore] = storage_context.docstore if storage_context is not None else None
        # 创建并返回 StorageContext 实例对象
        return StorageContext.from_defaults(vector_store=self.vector_store, docstore=docstore)

def _load_milvus_vector_store_class() -> Any:
    """
    延迟加载 LlamaIndex Milvus 向量存储类
    Returns:
        MilvusVectorStore 类对象
    """
    try:
        from llama_index.vector_stores.milvus import MilvusVectorStore

        return MilvusVectorStore
    except Exception as exc:
        raise RuntimeError(
            "Milvus 索引需要 llama-index-vector-stores-milvus，使用前请先安装项目依赖"
        ) from exc


def _ensure_event_loop_for_current_thread() -> None:
    """
    确保当前线程拥有 asyncio event loop，兼容 FastAPI 同步路由的 AnyIO worker thread
    """
    try:
        # 异步任务调度器
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def _milvus_scalar_field_types() -> List[Any]:
    """
    解析 Milvus 标量字段类型
    Returns:
        pymilvus DataType 列表，pymilvus 不可用时返回类型名列表
    """
    try:
        from pymilvus import DataType

        type_map = {
            "VARCHAR": DataType.VARCHAR,
            "INT64": DataType.INT64,
        }
        return [type_map[type_name] for type_name in SCALAR_FIELD_TYPE_NAMES]
    except Exception:
        return list(SCALAR_FIELD_TYPE_NAMES)


def _milvus_scalar_field_names() -> List[str]:
    """
    返回需要显式声明为 Milvus scalar 字段的元数据名称
    Returns:
        去除 LlamaIndex 自动字段和可空动态字段后的字段名列表
    """
    excluded = MILVUS_AUTO_FIELD_NAMES | MILVUS_DYNAMIC_METADATA_FIELD_NAMES
    return [field_name for field_name in SCALAR_FIELD_NAMES if field_name not in excluded]


def _milvus_scalar_field_specs() -> List[tuple[str, Any]]:
    """
    返回 Milvus scalar 字段名称和类型定义
    Returns:
        (字段名, 字段类型) 列表
    """
    # 过滤掉 LlamaIndex 自动字段和可空动态字段
    excluded = MILVUS_AUTO_FIELD_NAMES | MILVUS_DYNAMIC_METADATA_FIELD_NAMES
    return [
        (field_name, field_type)
        for field_name, field_type in zip(SCALAR_FIELD_NAMES, _milvus_scalar_field_types())
        if field_name not in excluded
    ]