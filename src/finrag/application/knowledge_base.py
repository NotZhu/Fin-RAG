"""Knowledge base initialization and rebuild orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

from llama_index.core import VectorStoreIndex

from finrag.application.llamaindex_engines import KnowledgeBaseUnavailableError
from finrag.core.node_schema import TextNode
from finrag.generation import GenerationIntegrationModule
from finrag.indexing import BM25SparseEmbeddingFunction, DataPreparationModule, IndexConstructionModule


class KnowledgeBaseService:
    """知识库服务类"""

    def __init__(self, system: Any):
        self.system = system

    def initialize_system(self) -> None:
        """
        初始化知识库相关模块
        """
        system = self.system
        # 初始化数据准备模块
        system.data_module = DataPreparationModule(
            system.config.data_path, # 数据路径
            knowledge_base_id=system.config.knowledge_base_id, # 知识库 ID
            chunk_size=system.config.chunk_size, # 分块大小
            chunk_overlap=system.config.chunk_overlap, # 分块重叠
            document_registry=system.document_registry, # 文档注册表
            docstore=getattr(system, "llama_docstore", None), # 可选 LlamaIndex docstore adapter
        )
        # 初始化索引构造模块
        system.index_module = IndexConstructionModule(
            model_name=system.config.embedding_model, # 嵌入模型名称
            collection_name=system.config.milvus_collection, # Milvus 集合名称
            milvus_host=system.config.milvus_host, # Milvus 主机地址
            milvus_port=system.config.milvus_port, # Milvus 端口号
            manifest_store=system.manifest_store, # 知识库清单存储模块
            sparse_embedding_function=BM25SparseEmbeddingFunction(system.bm25_store) if system.bm25_store is not None else None, # 可选的 BM25 稀疏嵌入函数
            rrf_k=system.config.rrf_k, # RRF 算法参数
        )
        # 如果启用了语义分块，配置嵌入模型
        if getattr(system.config, "use_semantic_chunking", False):
            system.data_module._use_semantic_chunking = True # 启用语义分块
            embed_model = getattr(system.index_module, "embed_model", None)
            if embed_model is not None:
                system.data_module._embed_model = embed_model # 配置嵌入模型
        # 初始化生成模块
        system.generation_module = GenerationIntegrationModule(
            model_name=system.config.llm_model, # LLM 模型名称
            temperature=system.config.temperature, # 温度参数
            max_tokens=system.config.max_tokens, # 最大令牌数
        )

    def build_knowledge_base(self) -> None:
        """构建知识库索引：优先从 docstore + Milvus 恢复，失败则全量重建。"""
        system = self.system
        with system._write_lock:
            system._ensure_modules()
            assert system.data_module is not None
            assert system.index_module is not None
            Path(system.config.data_path).mkdir(parents=True, exist_ok=True)
            # 构建预期清单
            expected_manifest = system._build_expected_manifest()
            # 从数据 store 加载所有叶子节点
            leaf_nodes: List[TextNode] = system._load_leaf_nodes_from_docstore()
            # 如果有叶子节点且清单匹配，尝试加载索引
            if leaf_nodes and system.index_module.manifest_matches(expected_manifest):
                # 向量检索索引对象
                vector_index: VectorStoreIndex = system.index_module.load_index(
                    expected_manifest, storage_context=system.data_module.storage_context
                )
                if vector_index is not None:
                    # 刷新检索索引
                    system._refresh_retrieval(vector_index, leaf_nodes)
                    return
            system._full_rebuild_locked()

    def ensure_knowledge_base_ready(self) -> None:
        """确保知识库已初始化并可用，不可用时先检查 Milvus 连通性再构建"""
        system = self.system
        # 如果知识库和数据模块都已初始化，直接返回可用
        if system.knowledge_query_engine is not None and system.data_module is not None:
            return
        with system._write_lock:
            if system.knowledge_query_engine is not None and system.data_module is not None:
                return
            # 检查 Milvus 连通性
            self._check_milvus()
            if system.data_module is None or system.index_module is None or system.generation_module is None:
                # 初始化知识库相关模块
                self.initialize_system()
            # 构建知识库索引
            self.build_knowledge_base()

    def _check_milvus(self) -> None:
        """检查 Milvus 连通性"""
        system = self.system
        from pymilvus import connections, exceptions as milvus_exc

        config = system.config
        uri = f"http://{config.milvus_host}:{config.milvus_port}" # Milvus 连接 URI
        alias = f"finrag_health_{id(self)}" # 连接别名
        try:
            # 尝试连接 Milvus
            connections.connect(alias=alias, uri=uri, timeout=3)
        except milvus_exc.MilvusException as exc:
            raise KnowledgeBaseUnavailableError(
                f"Milvus 不可连接: {uri} — {exc}",
                code="milvus_unavailable",
            )
        except Exception as exc:
            raise KnowledgeBaseUnavailableError(
                f"Milvus 连通性检查失败: {uri} — {exc}",
                code="milvus_unavailable",
            )
        finally:
            try:
                # 断开 Milvus 连接
                connections.disconnect(alias=alias)
            except Exception:
                pass

    def rebuild_from_sources(self) -> dict:
        """
        从数据源重建知识库索引
        """
        system = self.system
        with system._write_lock:
            system._ensure_modules()
            system._full_rebuild_locked(sync_source_registry=True)
            manifest = system.index_module.load_manifest() if system.index_module is not None else {}
            schema_version = int((manifest or {}).get("schema_version", 0) or 0)
            document_count = len(system.data_module.documents) if system.data_module is not None else 0
            chunk_count = len(system.data_module.chunks) if system.data_module is not None else 0
            return {
                "document_count": document_count,
                "chunk_count": chunk_count,
                "manifest_schema_version": schema_version,
            }
