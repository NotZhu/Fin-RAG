"""Knowledge base initialization and rebuild orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

from llama_index.core import VectorStoreIndex

from finrag.application.knowledge_base_scope import KnowledgeBaseScope
from finrag.application.llamaindex_engines import KnowledgeBaseUnavailableError
from finrag.application.runtime import KnowledgeBaseRuntime
from finrag.core.node_schema import TextNode
from finrag.generation import GenerationIntegrationModule
from finrag.indexing import DataPreparationModule, IndexConstructionModule


class KnowledgeBaseService:
    """知识库服务类"""

    def __init__(self, system: Any):
        self.system = system

    def _scope(self, knowledge_base_id: str | None = None) -> KnowledgeBaseScope:
        """根据知识库 ID 获取知识库作用域"""
        system = self.system
        # 解析知识库 ID，使用配置默认值或提供值
        resolved_id = knowledge_base_id or system.config.knowledge_base_id
        # 构建知识库作用域
        scope_builder = getattr(system, "knowledge_base_scope", None)
        if callable(scope_builder):
            return scope_builder(resolved_id)
        # 如果未提供作用域构建函数，默认使用配置中的知识库作用域
        return KnowledgeBaseScope.from_config(system.config, resolved_id)

    def initialize_system(self, knowledge_base_id: str | None = None) -> None:
        """
        初始化知识库相关模块
        Args:
            knowledge_base_id: 知识库 ID，未提供时使用配置默认值
        """
        system = self.system
        scope = self._scope(knowledge_base_id)
        runtime_cache_key = scope.runtime_cache_key
        if getattr(system, "kb_runtimes", None) is not None and runtime_cache_key in system.kb_runtimes:
            runtime = system.kb_runtimes[runtime_cache_key]
            if hasattr(system, "_activate_runtime"):
                system._activate_runtime(runtime)
            return

        # 初始化数据准备模块
        data_module = DataPreparationModule(
            system.config.data_path, # 数据路径
            knowledge_base_id=scope.knowledge_base_id, # 知识库 ID
            document_registry=system.document_registry, # 文档注册表
            docstore=getattr(system, "llama_docstore", None), # 可选 LlamaIndex docstore adapter
        )
        # 初始化索引构造模块
        index_module = IndexConstructionModule(
            model_name=system.config.embedding_model, # 嵌入模型名称
            collection_name=scope.collection_name, # Milvus 集合名称
            milvus_host=system.config.milvus_host, # Milvus 主机地址
            milvus_port=system.config.milvus_port, # Milvus 端口号
            manifest_store=system.manifest_store, # 知识库清单存储模块
            rrf_k=system.config.rrf_k, # RRF 算法参数
        )
        # 初始化生成模块
        generation_module = GenerationIntegrationModule(
            model_name=system.config.llm_model, # LLM 模型名称
            temperature=system.config.temperature, # 温度参数
            max_tokens=system.config.max_tokens, # 最大令牌数
        )
        # 初始化运行时模块
        runtime = KnowledgeBaseRuntime(
            scope=scope, # 知识库作用域
            data_module=data_module, # 数据准备模块
            index_module=index_module, # 索引构造模块
            generation_module=generation_module, # 生成模块
        )
        # 缓存运行时模块
        if getattr(system, "kb_runtimes", None) is not None:
            system.kb_runtimes[runtime_cache_key] = runtime
        # 激活运行时模块
        if hasattr(system, "_activate_runtime"):
            system._activate_runtime(runtime)
        else:
            system.data_module = data_module
            system.index_module = index_module
            system.generation_module = generation_module

    def build_knowledge_base(self, knowledge_base_id: str | None = None) -> None:
        """构建知识库索引：优先从 docstore + Milvus 恢复，失败则全量重建"""
        system = self.system
        with system._write_lock:
            knowledge_base_id = self._scope(knowledge_base_id).knowledge_base_id
            system._ensure_modules(knowledge_base_id)
            assert system.data_module is not None
            assert system.index_module is not None
            Path(system.config.data_path).mkdir(parents=True, exist_ok=True)
            # 配置知识库作用域
            knowledge_base_id = system._configure_knowledge_base_scope_locked(knowledge_base_id)
            # 构建预期清单
            expected_manifest = system._build_expected_manifest(knowledge_base_id)
            # 从数据 store 加载所有叶子节点
            leaf_nodes: List[TextNode] = system._load_leaf_nodes_from_docstore(knowledge_base_id)
            # 如果有叶子节点且清单匹配，尝试加载索引
            if leaf_nodes and system.index_module.manifest_matches(expected_manifest):
                # 向量检索索引对象
                vector_index: VectorStoreIndex = system.index_module.load_index(
                    expected_manifest, storage_context=system.data_module.storage_context
                )
                if vector_index is not None:
                    # 刷新检索索引
                    system._refresh_retrieval(vector_index, leaf_nodes, knowledge_base_id)
                    return
            system._full_rebuild_locked(knowledge_base_id)

    def ensure_knowledge_base_ready(self, knowledge_base_id: str | None = None) -> None:
        """确保知识库已初始化并可用，不可用时先检查 Milvus 连通性再构建"""
        system = self.system
        scope = self._scope(knowledge_base_id)
        runtime = getattr(system, "kb_runtimes", {}).get(scope.runtime_cache_key)
        # 如果知识库和数据模块都已初始化，直接返回可用
        if runtime is not None and runtime.knowledge_query_engine is not None and runtime.data_module is not None:
            system._activate_runtime(runtime)
            return
        with system._write_lock:
            runtime = getattr(system, "kb_runtimes", {}).get(scope.runtime_cache_key)
            if runtime is not None and runtime.knowledge_query_engine is not None and runtime.data_module is not None:
                system._activate_runtime(runtime)
                return
            # 检查 Milvus 连通性
            self._check_milvus()
            runtime = getattr(system, "kb_runtimes", {}).get(scope.runtime_cache_key)
            if runtime is None or runtime.data_module is None or runtime.index_module is None or runtime.generation_module is None:
                # 确保知识库相关模块已初始化
                if hasattr(system, "_ensure_modules"):
                    system._ensure_modules(scope.knowledge_base_id)
                else:
                    self.initialize_system(scope.knowledge_base_id)
            # 构建知识库索引
            self.build_knowledge_base(scope.knowledge_base_id)

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

    def rebuild_from_sources(self, knowledge_base_id: str) -> dict:
        """
        从数据源重建知识库索引
        """
        if not str(knowledge_base_id or "").strip():
            raise ValueError("knowledge_base_id is required")
        system = self.system
        with system._write_lock:
            knowledge_base_id = self._scope(knowledge_base_id).knowledge_base_id
            system._ensure_modules(knowledge_base_id)
            knowledge_base_id = system._configure_knowledge_base_scope_locked(knowledge_base_id)
            try:
                system._full_rebuild_locked(knowledge_base_id, sync_source_registry=True)
            except Exception as exc:
                if hasattr(system, "_mark_rebuild_documents_failed"):
                    system._mark_rebuild_documents_failed(knowledge_base_id, system._format_exception(exc))
                if hasattr(system, "_touch_knowledge_base"):
                    system._touch_knowledge_base(knowledge_base_id)
                raise
            if hasattr(system, "_touch_knowledge_base"):
                system._touch_knowledge_base(knowledge_base_id)
            manifest = system.index_module.load_manifest(knowledge_base_id) if system.index_module is not None else {}
            schema_version = int((manifest or {}).get("schema_version", 0) or 0)
            document_count = len(system.data_module.documents) if system.data_module is not None else 0
            chunk_count = len(system.data_module.chunks) if system.data_module is not None else 0
            return {
                "document_count": document_count,
                "chunk_count": chunk_count,
                "manifest_schema_version": schema_version,
            }
