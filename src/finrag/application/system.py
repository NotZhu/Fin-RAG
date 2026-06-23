"""FinRAG 系统编排"""

from __future__ import annotations

import logging
import shutil
from collections import Counter
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

from finrag.core.config import PROJECT_ROOT, RAGConfig
from finrag.core.node_schema import TextNode
from finrag.generation import GenerationIntegrationModule
from finrag.application.document_lifecycle import DocumentLifecycleService
from finrag.application.knowledge_base import KnowledgeBaseService
from finrag.application.knowledge_base_scope import KnowledgeBaseScope
from finrag.application.llamaindex_engines import build_knowledge_engines, build_top_router
from finrag.application.qa_pipeline import QAPipelineService
from finrag.application.runtime import KnowledgeBaseRuntime
from finrag.application.source_files import ManagedSourceFileService
from finrag.indexing import (
    DataPreparationModule,
    IndexConstructionModule,
)
from finrag.storage import (
    KnowledgeBaseArchivedError,
    KnowledgeBaseNotFoundError,
    PostgreSQLBM25StateStore,
    PostgreSQLIndexManifestStore,
    PostgreSQLKnowledgeBaseRegistry,
    PostgreSQLLlamaIndexDocumentStore,
    ProtectedKnowledgeBaseError,
)
from finrag.ingestion import DocumentRecord, PostgreSQLDocumentRegistry, compute_content_hash, is_path_within
from finrag.ingestion.parsers import utc_now_iso
from finrag.retrieval import build_reranker
from finrag.retrieval.tokenization import tokenize_chinese_text


def _load_environment(project_root: Path = PROJECT_ROOT) -> None:
    """
    从项目根目录或当前工作目录加载 .env 配置
    Args:
        project_root: 项目根路径
    Returns:
        无返回值，环境变量写入当前进程
    """
    project_env = project_root / ".env"
    if project_env.exists():
        load_dotenv(project_env)
        return
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(cwd_env)


_load_environment() # 加载环境变量
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s") # 配置日志记录
logging.getLogger("jieba").setLevel(logging.WARNING) # 设置 jieba 日志级别为 WARNING
logger = logging.getLogger(__name__)


class FinRAGSystem:
    """金融资料库 RAG 系统"""

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        初始化 FinRAG 系统编排器
        Args:
            config: 可选运行配置，未提供时从环境变量读取
        """
        # 初始化系统配置
        self.config = config or RAGConfig.from_env()
        # 初始化系统模块
        self.data_module: Optional[DataPreparationModule] = None
        # 索引构建模块
        self.index_module: Optional[IndexConstructionModule] = None
        # 生成模块
        self.generation_module: Optional[GenerationIntegrationModule] = None

        # 知识查询引擎
        self.knowledge_query_engine: Optional[Any] = None
        # 自动合并检索器
        self.auto_merge_retriever: Optional[Any] = None
        # 混合检索器
        self.hybrid_retriever: Optional[Any] = None
        # 顶部路由引擎
        self.router_engine: Optional[Any] = None
        # 每个知识库独立的运行时模块和检索引擎缓存
        self.kb_runtimes: Dict[str, KnowledgeBaseRuntime] = {}
        # 当前激活的知识库运行时 key
        self._active_runtime_key: Optional[str] = None


        # 文档注册表
        self.document_registry = PostgreSQLDocumentRegistry(self.config.database_url)
        # LlamaIndex 文档存储适配器
        self.llama_docstore = PostgreSQLLlamaIndexDocumentStore(self.config.database_url)
        # BM25 状态存储
        self.bm25_store = PostgreSQLBM25StateStore(self.config.database_url)
        # 索引清单存储
        self.manifest_store = PostgreSQLIndexManifestStore(self.config.database_url)
        # 知识库注册表
        self.knowledge_base_registry = PostgreSQLKnowledgeBaseRegistry(self.config.database_url)
        # 确保默认知识库存在
        self.knowledge_base_registry.ensure_default(self.config.knowledge_base_id)
        # 源文件管理服务
        self.source_files = ManagedSourceFileService(self.config, self.document_registry)
        
        # 知识库服务
        self.knowledge_base = self._default_knowledge_base()
        # 问答管道服务
        self.qa_pipeline = self._default_qa_pipeline()
        # 文档生命周期服务
        self.document_lifecycle = self._default_document_lifecycle()
        
        # 串行化索引写操作，避免 Milvus、NodeStore 和注册表状态交叉写入
        self._write_lock = RLock()
        # 检索结果 reranker 模块
        self.reranker = build_reranker(
            self.config.reranker_provider,
            self.config.reranker_model,
            self.config.reranker_endpoint,
            self.config.reranker_api_key,
            self.config.reranker_top_n,
        )

    def initialize_system(self, knowledge_base_id: str | None = None) -> None:
        """
        初始化数据准备、索引构建和生成模块
        Args:
            knowledge_base_id: 知识库 ID，未提供时使用配置默认值
        Returns:
            无返回值，初始化后的模块写入实例属性
        """
        if knowledge_base_id is None:
            self.knowledge_base.initialize_system()
            return
        self.knowledge_base.initialize_system(knowledge_base_id)

    def build_knowledge_base(self, knowledge_base_id: str | None = None) -> None:
        """
        加载文档、构建层级节点、创建或复用向量索引，并初始化检索模块
        Args:
            knowledge_base_id: 知识库 ID，未提供时使用配置默认值
        Returns:
            无返回值，资料库状态写入实例属性
        """
        if knowledge_base_id is None:
            self.knowledge_base.build_knowledge_base()
            return
        self.knowledge_base.build_knowledge_base(knowledge_base_id)

    def ensure_knowledge_base_ready(self, knowledge_base_id: str | None = None) -> None:
        """
        确保 knowledge 问答所需的数据、索引和检索模块已经可用
        Args:
            knowledge_base_id: 知识库 ID，未提供时使用配置默认值
        """
        if knowledge_base_id is None:
            self.knowledge_base.ensure_knowledge_base_ready()
            return
        # 确保知识库已激活
        self._ensure_knowledge_base_active(knowledge_base_id)
        # 确保知识库已构建
        self.knowledge_base.ensure_knowledge_base_ready(knowledge_base_id)

    def rebuild_from_sources(self, knowledge_base_id: str) -> dict:
        """
        从源文档强制全量重建 PostgreSQL 节点/BM25 状态和 Milvus collection
        Args:
            knowledge_base_id: 知识库 ID
        Returns:
            重建摘要，供 CLI 和运维任务展示
        """
        # 确保知识库已激活
        self._ensure_knowledge_base_active(knowledge_base_id)
        return self.knowledge_base.rebuild_from_sources(knowledge_base_id)

    def ready(self, knowledge_base_id: str | None = None) -> dict:
        """
        返回系统是否完成检索模块初始化及当前资料库统计
        Args:
            knowledge_base_id: 可选知识库 ID，提供时返回该知识库的运行时和文档统计
        Returns:
            包含 ready、status、文档数和节点数的状态字典
        """
        if knowledge_base_id is None:
            stats = self.get_statistics()
            retrieval_ready = self.knowledge_query_engine is not None
        else:
            scope = self.knowledge_base_scope(knowledge_base_id)
            runtime = self._active_runtime(scope.knowledge_base_id)
            documents = self.document_registry.list_public(scope.knowledge_base_id)
            stats = {
                "total_documents": len(documents),
                "total_chunks": sum(int(document.get("chunk_count") or 0) for document in documents),
            }
            retrieval_ready = (
                runtime is not None
                and runtime.knowledge_query_engine is not None
                and runtime.data_module is not None
            )
        return {
            "ready": retrieval_ready,
            "status": "ready" if retrieval_ready else "not_ready",
            "total_documents": int(stats.get("total_documents", 0) or 0),
            "total_chunks": int(stats.get("total_chunks", 0) or 0),
            "last_error": None,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        汇总资料库统计，优先使用文档注册表补齐从节点存储恢复时缺失的文档计数
        Returns:
            文档、节点和文件类型统计字典
        """
        # 优先使用数据准备模块的统计信息
        stats = self.data_module.get_statistics() if self.data_module is not None else {}
        payload = dict(stats)
        # 从文档注册表获取公开文档记录
        documents = self.document_registry.list_public()
        if documents:
            # 统计文档类型分布，并补齐文档总数和类型分布信息
            file_types = Counter(str(document.get("file_type") or "unknown") for document in documents)
            payload["total_documents"] = len(documents)
            payload["file_types"] = dict(file_types)
        else:
            # 如果没有文档记录，补齐默认统计信息
            payload.setdefault("total_documents", 0)
            payload.setdefault("file_types", {})
        # 兜底统计信息
        payload.setdefault("total_chunks", 0)
        payload.setdefault("avg_chunk_size", 0)
        return payload

    def list_documents(self, knowledge_base_id: str) -> List[dict]:
        """
        列出文档注册表中的公开文档记录
        Args:
            knowledge_base_id: 知识库 ID
        Returns:
            文档状态字典列表
        """
        scope = self.knowledge_base_scope(knowledge_base_id)
        return self.document_registry.list_public(scope.knowledge_base_id)

    def knowledge_base_scope(self, knowledge_base_id: str) -> KnowledgeBaseScope:
        """
        获取指定知识库的作用域信息
        Args:
            knowledge_base_id: 知识库 ID
        Returns:
            当前知识库的路径、collection 和缓存 key 信息
        """
        return KnowledgeBaseScope.from_config(self.config, knowledge_base_id)

    def list_knowledge_bases(self) -> List[dict]:
        """
        列出所有知识库
        Returns:
            包含文档数量的知识库公开记录
        """
        # 确保知识库注册表与文档注册表同步
        self._sync_knowledge_bases_from_documents()
        # 统计每个知识库的文档数量
        document_counts = Counter(
            str(getattr(record, "knowledge_base_id", "") or "")
            for record in self.document_registry.records.values()
            if getattr(record, "status", "") != "deleted"
        )
        return [
            record.to_dict(document_count=document_counts.get(record.knowledge_base_id, 0))
            for record in self.knowledge_base_registry.list()
        ]

    def create_knowledge_base(self, knowledge_base_id: str) -> dict:
        """
        创建用户指定 ID 的知识库，并准备其源文件目录
        Args:
            knowledge_base_id: 用户输入的知识库 ID，也作为知识库名
        Returns:
            新知识库公开记录
        """
        record = self.knowledge_base_registry.create(knowledge_base_id)
        # 确保知识库源文件目录存在
        self.knowledge_base_scope(record.knowledge_base_id).source_root.mkdir(parents=True, exist_ok=True)
        return record.to_dict(document_count=0)

    def archive_knowledge_base(self, knowledge_base_id: str) -> dict:
        """
        归档知识库，保留文档、索引和源文件，但禁止问答和写入操作
        Args:
            knowledge_base_id: 知识库 ID
        Returns:
            归档后的知识库公开记录
        """
        knowledge_base_id = self.knowledge_base_scope(knowledge_base_id).knowledge_base_id
        self._ensure_not_default_knowledge_base(knowledge_base_id)
        with self._write_lock:
            # 归档知识库
            record = self.knowledge_base_registry.archive(knowledge_base_id)
            # 丢弃知识库运行时缓存
            self._discard_knowledge_base_runtime(knowledge_base_id)
            return record.to_dict(document_count=self._knowledge_base_document_count(knowledge_base_id))

    def restore_knowledge_base(self, knowledge_base_id: str) -> dict:
        """
        恢复已归档知识库
        Args:
            knowledge_base_id: 知识库 ID
        Returns:
            恢复后的知识库公开记录
        """
        knowledge_base_id = self.knowledge_base_scope(knowledge_base_id).knowledge_base_id
        with self._write_lock:
            # 恢复知识库
            record = self.knowledge_base_registry.restore(knowledge_base_id)
            # 确保知识库源文件目录存在
            self.knowledge_base_scope(knowledge_base_id).source_root.mkdir(parents=True, exist_ok=True)
            return record.to_dict(document_count=self._knowledge_base_document_count(knowledge_base_id))

    def delete_knowledge_base(self, knowledge_base_id: str) -> dict:
        """
        删除知识库及其托管源文件、文档记录、BM25、docstore、manifest 和运行时缓存
        Args:
            knowledge_base_id: 知识库 ID
        Returns:
            删除后的知识库公开记录
        """
        knowledge_base_id = self.knowledge_base_scope(knowledge_base_id).knowledge_base_id
        self._ensure_not_default_knowledge_base(knowledge_base_id)
        with self._write_lock:
            # 确保知识库存在
            if self.knowledge_base_registry.get_optional(knowledge_base_id, include_deleted=True) is None:
                raise KnowledgeBaseNotFoundError(knowledge_base_id)
            # 删除知识库中的所有文档
            for record in list(self.document_registry.records.values()):
                if record.knowledge_base_id == knowledge_base_id and record.status != "deleted":
                    self.document_registry.mark_deleted(record.document_id)
            # 删除知识库索引
            if self.bm25_store is not None:
                self.bm25_store.clear(knowledge_base_id)
            # 删除知识库文档存储
            if self.llama_docstore is not None:
                if hasattr(self.llama_docstore, "delete_knowledge_base"):
                    self.llama_docstore.delete_knowledge_base(knowledge_base_id)
                else:
                    for record in list(self.document_registry.records.values()):
                        if record.knowledge_base_id == knowledge_base_id:
                            self.llama_docstore.delete_nodes_by_document(record.document_id, knowledge_base_id)
            # 删除知识库manifest
            if hasattr(self.manifest_store, "delete_manifest"):
                self.manifest_store.delete_manifest(knowledge_base_id)
            # 删除知识库源目录和待处理目录
            self._delete_knowledge_base_source_dirs(knowledge_base_id)
            # 删除丢弃知识库运行时缓存
            self._discard_knowledge_base_runtime(knowledge_base_id)
            record = self.knowledge_base_registry.mark_deleted(knowledge_base_id)
            return record.to_dict(document_count=0)

    def _knowledge_base_document_count(self, knowledge_base_id: str) -> int:
        """
        统计知识库中未删除文档数量
        Args:
            knowledge_base_id: 知识库 ID
        Returns:
            未删除文档数量
        """
        return sum(
            1
            for record in self.document_registry.records.values()
            if record.knowledge_base_id == knowledge_base_id and record.status != "deleted"
        )

    def _ensure_not_default_knowledge_base(self, knowledge_base_id: str) -> None:
        """
        校验知识库是否为默认知识库
        Args:
            knowledge_base_id: 知识库 ID
        """
        knowledge_base_id = self.knowledge_base_scope(knowledge_base_id).knowledge_base_id
        if knowledge_base_id == self.config.knowledge_base_id:
            # 处理默认知识库保护
            raise ProtectedKnowledgeBaseError(knowledge_base_id)

    def _ensure_knowledge_base_active(self, knowledge_base_id: str) -> None:
        """
        校验知识库是否可执行业务操作
        Args:
            knowledge_base_id: 知识库 ID
        """
        record = self.knowledge_base_registry.get_optional(knowledge_base_id, include_deleted=True)
        # 确保知识库存在
        if record is None:
            record = self.knowledge_base_registry.ensure_default(knowledge_base_id)
        # 校验知识库是否已删除
        if record.status == "deleted":
            raise KnowledgeBaseNotFoundError(knowledge_base_id)
        # 校验知识库是否已归档
        if record.status == "archived":
            raise KnowledgeBaseArchivedError(knowledge_base_id)

    def ensure_knowledge_base_active(self, knowledge_base_id: str) -> None:
        """
        对外校验知识库是否可执行业务操作
        Args:
            knowledge_base_id: 知识库 ID
        """
        self._ensure_knowledge_base_active(knowledge_base_id)

    def _discard_knowledge_base_runtime(self, knowledge_base_id: str) -> None:
        """
        丢弃知识库运行时缓存
        Args:
            knowledge_base_id: 知识库 ID
        """
        knowledge_base_id = self.knowledge_base_scope(knowledge_base_id).knowledge_base_id
        scope = self.knowledge_base_scope(knowledge_base_id)
        runtime = getattr(self, "kb_runtimes", {}).pop(scope.runtime_cache_key, None)
        if runtime is not None and getattr(self, "_active_runtime_key", None) == scope.runtime_cache_key:
            self._active_runtime_key = None
            self.data_module = None
            self.index_module = None
            self.generation_module = None
            self.knowledge_query_engine = None
            self.auto_merge_retriever = None
            self.hybrid_retriever = None
            self.router_engine = None

    def _delete_knowledge_base_source_dirs(self, knowledge_base_id: str) -> None:
        """
        删除知识库源目录
        Args:
            knowledge_base_id: 知识库 ID
        """
        data_root = Path(self.config.data_path)
        scope = self.knowledge_base_scope(knowledge_base_id)
        # 删除知识库源目录和待处理目录
        for path in [scope.source_root, data_root / ".pending" / knowledge_base_id]:
            if not is_path_within(path, data_root):
                logger.warning("跳过删除托管目录外的知识库源目录: %s", path)
                continue
            if path.exists():
                shutil.rmtree(path)

    def _sync_knowledge_bases_from_documents(self) -> None:
        """
        将既有文档注册表中的知识库补入知识库注册表
        """
        for record in self.document_registry.records.values():
            if getattr(record, "status", "") == "deleted":
                continue
            knowledge_base_id = str(getattr(record, "knowledge_base_id", "") or "").strip()
            if not knowledge_base_id:
                continue
            # 如果知识库不存在，确保/创建知识库
            if self.knowledge_base_registry.get_optional(knowledge_base_id) is None:
                self.knowledge_base_registry.ensure_default(knowledge_base_id)

    def _default_qa_pipeline(self) -> QAPipelineService:
        """
        创建默认问答 pipeline 服务
        Returns:
            绑定当前系统实例的 QAPipelineService
        """
        return QAPipelineService(self)

    def _default_knowledge_base(self) -> KnowledgeBaseService:
        """
        创建默认知识库服务
        Returns:
            绑定当前系统实例的 KnowledgeBaseService
        """
        return KnowledgeBaseService(self)

    def _default_document_lifecycle(self) -> DocumentLifecycleService:
        """
        创建默认文档生命周期服务
        Returns:
            绑定当前系统实例的 DocumentLifecycleService
        """
        return DocumentLifecycleService(self)

    def _managed_source_files(self) -> ManagedSourceFileService:
        """
        获取源文件服务，并同步可能被测试或调用方替换的文档注册表
        Returns:
            当前系统使用的源文件管理服务
        """
        self.source_files.document_registry = self.document_registry
        return self.source_files

    def prepare_uploaded_file(
        self,
        file_path: Path,
        filename: str,
        knowledge_base_id: str,
    ) -> dict:
        """
        处理上传文件的去重、落盘和文档注册，尚不强制同步建索引
        Args:
            file_path: 上传临时文件路径
            filename: 用户侧原始文件名
            knowledge_base_id: 目标资料库 ID
        Returns:
            文档注册记录的公开字典
        """
        self._ensure_knowledge_base_active(knowledge_base_id)
        return self.document_lifecycle.prepare_uploaded_file(file_path, filename, knowledge_base_id)

    def index_registered_document(self, document_id: str) -> dict:
        """
        对已注册文档执行单文档增量索引，并更新该文档索引状态
        Args:
            document_id: 待索引文档 ID
        Returns:
            更新后的文档公开状态
        """
        try:
            # 获取文档注册记录
            record = self.document_registry.get(document_id)
        except KeyError:
            record = None
        if record is not None:
            # 确保知识库存在
            self._ensure_knowledge_base_active(record.knowledge_base_id)
        return self.document_lifecycle.index_registered_document(document_id)

    def ingest_uploaded_file(
        self,
        file_path: Path,
        filename: str,
        knowledge_base_id: str,
    ) -> dict:
        """
        完成上传文件注册并同步构建索引
        Args:
            file_path: 上传临时文件路径
            filename: 用户侧原始文件名
            knowledge_base_id: 目标资料库 ID
        Returns:
            索引完成后的文档公开状态
        """
        self._ensure_knowledge_base_active(knowledge_base_id)
        return self.document_lifecycle.ingest_uploaded_file(file_path, filename, knowledge_base_id)

    def delete_document(self, document_id: str, knowledge_base_id: str) -> dict:
        """
        删除托管文档源文件，并按 document_id 增量删除索引
        Args:
            document_id: 待删除文档 ID
            knowledge_base_id: 知识库 ID
        Returns:
            删除后的文档公开状态
        """
        self._ensure_knowledge_base_active(knowledge_base_id)
        return self.document_lifecycle.delete_document(
            document_id,
            self.knowledge_base_scope(knowledge_base_id).knowledge_base_id,
        )

    def reindex_document(self, document_id: str, knowledge_base_id: str) -> dict:
        """
        对指定文档重新执行解析和索引构建
        Args:
            document_id: 待重建索引的文档 ID
            knowledge_base_id: 知识库 ID
        Returns:
            重建后的文档公开状态
        """
        self._ensure_knowledge_base_active(knowledge_base_id)
        return self.document_lifecycle.reindex_document(
            document_id,
            self.knowledge_base_scope(knowledge_base_id).knowledge_base_id,
        )

    def ask_question(
        self,
        question: str,
        knowledge_base_id: str,
        return_sources: bool = False,
        return_trace: bool = False,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event: Any = None,
    ):
        """
        执行完整 RAG 问答流程：查询分析、检索、置信度门控、证据扩展和生成
        Args:
            question: 用户问题
            return_sources: 是否返回来源证据
            return_trace: 是否返回调试 trace
            knowledge_base_id: 资料库过滤条件
            event_sink: 可选 SSE 事件回调
            cancel_event: 可选取消信号
        Returns:
            FinRAGResponse 问答响应对象
        """
        scope = self.knowledge_base_scope(knowledge_base_id)
        # 按请求知识库准备并激活运行时，避免复用其他知识库的 router/retriever。
        self.ensure_knowledge_base_ready(scope.knowledge_base_id)
        return self.qa_pipeline.ask_question(
            question,
            return_sources=return_sources,
            return_trace=return_trace,
            knowledge_base_id=scope.knowledge_base_id,
            event_sink=event_sink,
            cancel_event=cancel_event,
        )

    def _activate_runtime(self, runtime: KnowledgeBaseRuntime) -> None:
        """
        将指定知识库运行时映射到系统当前使用的模块属性
        Args:
            runtime: 知识库运行时
        """
        self._active_runtime_key = runtime.scope.runtime_cache_key
        self.data_module = runtime.data_module
        self.index_module = runtime.index_module
        self.generation_module = runtime.generation_module
        self.knowledge_query_engine = runtime.knowledge_query_engine
        self.auto_merge_retriever = runtime.auto_merge_retriever
        self.hybrid_retriever = runtime.hybrid_retriever
        self.router_engine = runtime.router_engine

    def _active_runtime(self, knowledge_base_id: str | None = None) -> Optional[KnowledgeBaseRuntime]:
        """
        获取当前或指定知识库的运行时
        Args:
            knowledge_base_id: 可选知识库 ID
        Returns:
            命中的运行时，不存在时返回 None
        """
        if knowledge_base_id is not None:
            scope = self.knowledge_base_scope(knowledge_base_id)
            return getattr(self, "kb_runtimes", {}).get(scope.runtime_cache_key)
        if getattr(self, "_active_runtime_key", None) is None:
            return None
        return getattr(self, "kb_runtimes", {}).get(self._active_runtime_key)

    def _adopt_current_runtime(self, scope: KnowledgeBaseScope) -> Optional[KnowledgeBaseRuntime]:
        """
        将当前系统模块映射为指定知识库运行时
        Args:
            scope: 知识库作用域
        Returns:
            新建或命中的知识库运行时；模块不完整时返回 None
        """
        if self.data_module is None or self.index_module is None or self.generation_module is None:
            return None
        if getattr(self, "kb_runtimes", None) is None:
            self.kb_runtimes = {}
        runtime = KnowledgeBaseRuntime(
            scope=scope,
            data_module=self.data_module,
            index_module=self.index_module,
            generation_module=self.generation_module,
            knowledge_query_engine=getattr(self, "knowledge_query_engine", None),
            auto_merge_retriever=getattr(self, "auto_merge_retriever", None),
            hybrid_retriever=getattr(self, "hybrid_retriever", None),
            router_engine=getattr(self, "router_engine", None),
        )
        self.kb_runtimes[scope.runtime_cache_key] = runtime
        # 激活新创建的运行时
        self._activate_runtime(runtime)
        return runtime

    def _ensure_modules(self, knowledge_base_id: str | None = None) -> None:
        """
        确保核心模块已初始化，并同步最新文档注册表引用
        Args:
            knowledge_base_id: 知识库 ID，未提供时使用配置默认值
        Returns:
            无返回值
        """
        scope = self.knowledge_base_scope(knowledge_base_id or self.config.knowledge_base_id)
        runtime = getattr(self, "kb_runtimes", {}).get(scope.runtime_cache_key)
        # 如果运行时不存在，尝试映射当前系统模块为指定知识库运行时
        if runtime is None:
            runtime = self._adopt_current_runtime(scope)
        # 如果运行时或其模块为空，说明系统模块未初始化，需要重新初始化
        if runtime is None or runtime.data_module is None or runtime.index_module is None or runtime.generation_module is None:
            self.initialize_system(scope.knowledge_base_id)
            runtime = getattr(self, "kb_runtimes", {}).get(scope.runtime_cache_key)
        # 如果运行时存在，激活该运行时的模块
        if runtime is not None:
            self._activate_runtime(runtime)
        # 断言模块已初始化，否则抛出异常
        assert self.data_module is not None
        # 同步文档注册表引用，确保最新状态被使用
        self.data_module.document_registry = self.document_registry

    def _configure_knowledge_base_scope_locked(self, knowledge_base_id: str) -> str:
        """
        将数据模块和稀疏向量函数切换到指定知识库上下文
        Args:
            knowledge_base_id: 知识库 ID
        Returns:
            校验后的知识库 ID
        """
        scope = self.knowledge_base_scope(knowledge_base_id)
        runtime = getattr(self, "kb_runtimes", {}).get(scope.runtime_cache_key)
        if runtime is not None:
            self._activate_runtime(runtime)
        if self.data_module is not None:
            self.data_module.knowledge_base_id = scope.knowledge_base_id
        if self.index_module is not None:
            self.index_module.collection_name = scope.collection_name
            # 获取稀疏向量函数实例
            sparse_embedding = getattr(self.index_module, "sparse_embedding_function", None)
            # 如果稀疏向量函数支持设置知识库 ID，则调用设置方法
            if hasattr(sparse_embedding, "set_knowledge_base_id"):
                sparse_embedding.set_knowledge_base_id(scope.knowledge_base_id)
        return scope.knowledge_base_id

    def _build_expected_manifest(self, knowledge_base_id: str) -> Dict[str, Any]:
        """基于当前配置和活跃文档构建索引清单"""
        assert self.index_module is not None
        knowledge_base_id = self.knowledge_base_scope(knowledge_base_id).knowledge_base_id
        # 从文档注册表获取所有公开文档
        public_docs = self.document_registry.list_public(knowledge_base_id) if self.document_registry is not None else []
        # 从数据模块获取所有叶子节点
        leaf_nodes = [
            node
            for node in (getattr(self.data_module, "chunks", None) or [])
            if str((node.metadata or {}).get("knowledge_base_id") or "") == knowledge_base_id
        ]
        return self.index_module.build_manifest(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            knowledge_base_id=knowledge_base_id,
            llamaindex_index_store_dir=self.config.llamaindex_index_store_dir, # LlamaIndex 索引存储目录
            index_ids=["finrag-auto-merge"], # 索引 ID 列表
            document_count=len(public_docs), # 文档数量
            node_count=len(leaf_nodes), # 叶子节点数量
        )

    def _full_rebuild_locked(self, knowledge_base_id: str, *, sync_source_registry: bool = False) -> None:
        """
        全量重建索引
        Args:
            knowledge_base_id: 知识库 ID
            sync_source_registry: 是否同步从源注册目录加载文档
        Returns:
            无返回值
        """
        self._ensure_modules(knowledge_base_id)
        assert self.data_module is not None
        assert self.index_module is not None
        # 将数据模块和稀疏向量函数切换到指定知识库上下文
        knowledge_base_id = self._configure_knowledge_base_scope_locked(knowledge_base_id)

        restore_registry = self.data_module.document_registry
        # 保存当前数据路径
        restore_data_path = getattr(self.data_module, "data_path", self.config.data_path)
        # 如果同步从源注册目录加载文档，先清空数据模块的文档注册表
        if sync_source_registry:
            # 标记所有文档为解析中
            self._mark_rebuild_documents_parsing(knowledge_base_id)
            # 刷新知识库更新时间
            self._touch_knowledge_base(knowledge_base_id)
            self.data_module.document_registry = None
            self.data_module.data_path = str(self.knowledge_base_scope(knowledge_base_id).source_root)
        try:
            # 加载文档
            self.data_module.load_documents()
        finally:
            if sync_source_registry:
                # 恢复数据模块的文档注册表
                self.data_module.document_registry = restore_registry
                self.data_module.data_path = restore_data_path
        if sync_source_registry:
            # 从源注册目录加载文档
            self._replace_registry_from_source_documents(
                knowledge_base_id,
                [],
                status="parsing",
                upload_time=utc_now_iso(),
            )
            # 刷新知识库更新时间
            self._touch_knowledge_base(knowledge_base_id)

        # 如果没有文档，直接清空索引
        if not self.data_module.documents:
            # 清空索引后返回的空索引对象
            vector_index = self.index_module.clear_index(storage_context=self.data_module.storage_context)
            # 清空 BM25 索引
            self._replace_bm25_all_locked(knowledge_base_id, [])
            # 保存空清单
            self.index_module.save_manifest(self._build_expected_manifest(knowledge_base_id))
            # 刷新检索
            self._refresh_retrieval(vector_index, [], knowledge_base_id)
            return

        leaf_nodes = self._rebuild_via_pipeline(knowledge_base_id, sync_source_registry=sync_source_registry)
        # 保存索引清单
        self.index_module.save_manifest(self._build_expected_manifest(knowledge_base_id))
        if leaf_nodes:
            # 加载索引
            vector_index = self.index_module.load_index(
                self._build_expected_manifest(knowledge_base_id),
                storage_context=self.data_module.storage_context,
            )
            if vector_index is None:
                # 如果清单不匹配，重新构建索引
                vector_index = self.index_module.build_vector_index(
                    leaf_nodes, storage_context=self.data_module.storage_context, reset=False,
                )
        else:
            # 如果没有叶子节点，清空索引
            vector_index = self.index_module.clear_index(storage_context=self.data_module.storage_context)
        # 刷新检索结果
        self._refresh_retrieval(vector_index, leaf_nodes or [], knowledge_base_id)

    def _rebuild_via_pipeline(self, knowledge_base_id: str, *, sync_source_registry: bool = False) -> list:
        """
        IngestionPipeline 重建索引流程
        Args:
            knowledge_base_id: 知识库 ID
            sync_source_registry: 是否同步从源注册目录加载文档
        Returns:
            新构建的叶子节点列表
        """
        from llama_index.core.node_parser import get_leaf_nodes
        from finrag.indexing.nodes import build_ingestion_pipeline

        self.index_module.init_collection(reset=True)
        # 是否使用语义分块
        use_semantic = getattr(self.config, "use_semantic_chunking", False)
        # 获取嵌入模型
        embed_model = getattr(self.index_module, "embed_model", None)
        if embed_model is None:
            raise RuntimeError("embed_model 未初始化，无法使用 pipeline")

        # 构建 IngestionPipeline
        pipeline = build_ingestion_pipeline(
            self.data_module,
            embed_model,
            use_semantic_chunking=use_semantic,
        )
        # 运行 IngestionPipeline，获取所有节点
        all_nodes = list(pipeline.run(documents=self.data_module.documents, show_progress=True))
        # 从所有节点中提取叶子节点
        leaf_nodes: list = get_leaf_nodes(all_nodes)
        self.data_module.all_nodes = all_nodes
        self.data_module.chunks = leaf_nodes
        docstore = self.llama_docstore or getattr(self.data_module.storage_context, "docstore", None)
        if docstore is not None:
            delete_knowledge_base = getattr(docstore, "delete_knowledge_base", None)
            if callable(delete_knowledge_base):
                delete_knowledge_base(knowledge_base_id)
            if all_nodes:
                docstore.add_documents(all_nodes)
        if leaf_nodes:
            self.index_module.build_vector_index(
                leaf_nodes,
                storage_context=self.data_module.storage_context,
                reset=False,
            )

        # 如果同步从源注册目录加载文档，重写文档注册表
        if sync_source_registry:
            self._replace_registry_from_source_documents(knowledge_base_id, leaf_nodes)
        # 重置 BM25 模型
        self._replace_bm25_all_locked(knowledge_base_id, leaf_nodes)
        return leaf_nodes

    def _replace_registry_from_source_documents(
        self,
        knowledge_base_id: str,
        leaf_nodes: List[TextNode],
        *,
        status: str = "indexed",
        upload_time: Optional[str] = None,
    ) -> None:
        """
        根据源目录加载结果重写文档注册表
        Args:
            knowledge_base_id: 知识库 ID
            leaf_nodes: 全量重建得到的叶子节点列表
        Returns:
            无返回值
        """
        assert self.data_module is not None
        upload_time = upload_time or utc_now_iso()

        # 统计每个文档的分块数量
        chunk_count_by_document = Counter(
            str((node.metadata or {}).get("document_id") or "")
            for node in leaf_nodes
            if (node.metadata or {}).get("document_id")
        )
        # 构建文档记录映射
        records: Dict[str, DocumentRecord] = {}
        for doc in self.data_module.documents:
            metadata = doc.metadata or {}
            document_id = str(metadata.get("document_id") or "")
            source_path = str(metadata.get("source_path") or "")
            if not document_id or not source_path or document_id in records:
                continue
            path = Path(source_path)
            records[document_id] = DocumentRecord(
                document_id=document_id,
                source_path=source_path,
                filename=str(metadata.get("filename") or path.name),
                file_type=str(metadata.get("file_type") or path.suffix.lower().lstrip(".")),
                content_hash=compute_content_hash(path),
                knowledge_base_id=str(metadata.get("knowledge_base_id") or knowledge_base_id),
                status=status,
                chunk_count=int(chunk_count_by_document.get(document_id, 0)) if status == "indexed" else 0,
                upload_time=upload_time,
            )
        # 保留其他知识库的文档记录
        preserved_records = {
            document_id: record
            for document_id, record in self.document_registry.records.items()
            if record.knowledge_base_id != knowledge_base_id
        }
        self.document_registry.records = {**preserved_records, **records}
        self.document_registry.save()

    def _mark_rebuild_documents_parsing(self, knowledge_base_id: str) -> None:
        """
        将重建开始前已有的文档标记为解析中
        Args:
            knowledge_base_id: 知识库 ID
        Returns:
            无返回值
        """
        changed = False
        for record in list(self.document_registry.records.values()):
            if record.knowledge_base_id == knowledge_base_id and record.status != "deleted":
                record.status = "parsing"
                record.chunk_count = 0
                record.last_error = None
                changed = True
        if changed:
            self.document_registry.save()

    def _mark_rebuild_documents_failed(self, knowledge_base_id: str, error: str) -> None:
        """
        将重建过程中处于解析中的文档标记为失败
        Args:
            knowledge_base_id: 知识库 ID
            error: 失败原因
        Returns:
            无返回值
        """
        for record in list(self.document_registry.records.values()):
            if record.knowledge_base_id == knowledge_base_id and record.status == "parsing":
                self.document_registry.mark_failed(record.document_id, error)

    def _touch_knowledge_base(self, knowledge_base_id: str) -> None:
        """
        刷新知识库更新时间，兼容测试替身未实现 touch 的情况
        Args:
            knowledge_base_id: 知识库 ID
        Returns:
            无返回值
        """
        touch = getattr(self.knowledge_base_registry, "touch", None)
        if callable(touch):
            touch(knowledge_base_id)

    def _replace_bm25_all_locked(self, knowledge_base_id: str, leaf_nodes: List[TextNode]) -> None:
        """
        用当前全部叶子节点重写 BM25 稀疏状态
        Args:
            knowledge_base_id: 知识库 ID
            leaf_nodes: 当前资料库全部叶子节点
        Returns:
            无返回值
        """
        if self.bm25_store is None:
            return
        # 清空 BM25 稀疏状态
        self.bm25_store.clear(knowledge_base_id)
        # 按文档 ID 分组叶子节点
        leaf_nodes_by_document: Dict[str, List[TextNode]] = {}
        for node in leaf_nodes:
            # 过滤出当前知识库的叶子节点
            if str((node.metadata or {}).get("knowledge_base_id") or "") != knowledge_base_id:
                continue
            document_id = str((node.metadata or {}).get("document_id") or "")
            if document_id:
                leaf_nodes_by_document.setdefault(document_id, []).append(node)
        # 逐文档写入 BM25 稀疏状态，确保同一文档的分块写入操作连续，避免跨文档交叉写入导致的状态不一致问题
        for document_id, document_leaf_nodes in leaf_nodes_by_document.items():
            self._replace_bm25_document_chunks_locked(knowledge_base_id, document_id, document_leaf_nodes)

    def _replace_bm25_document_chunks_locked(self, knowledge_base_id: str, document_id: str, leaf_nodes: List[TextNode]) -> None:
        """
        重写单个文档的 BM25 分块词频状态
        Args:
            knowledge_base_id: 知识库 ID
            document_id: 待更新的文档 ID
            leaf_nodes: 该文档的叶子节点列表
        Returns:
            无返回值
        """
        if self.bm25_store is None:
            return
        # 初始化分块词频字典，键为分块 ID，值为词频字典，键为词，值为词频
        chunk_token_counts: Dict[str, Dict[str, int]] = {}
        # 遍历该文档的所有叶子节点，统计每个分块的词频
        for node in leaf_nodes:
            metadata = node.metadata or {}
            chunk_id = str(metadata.get("chunk_id") or node.node_id)
            tokens = tokenize_chinese_text(node.get_content())
            chunk_token_counts[chunk_id] = dict(Counter(tokens))
        # 重写该文档的 BM25 分块词频状态
        self.bm25_store.replace_document_chunks(knowledge_base_id, document_id, chunk_token_counts)

    def _ensure_incremental_index_ready_locked(self, knowledge_base_id: str) -> None:
        """
        确保增量索引状态与当前文档注册表一致
        """
        self._ensure_modules(knowledge_base_id)
        assert self.data_module is not None
        assert self.index_module is not None
        # 切换到指定知识库上下文
        knowledge_base_id = self._configure_knowledge_base_scope_locked(knowledge_base_id)

        # 构建预期索引清单
        expected_manifest = self._build_expected_manifest(knowledge_base_id)

        # 如果索引清单与预期清单不匹配，执行全量重建
        if self.index_module.load_manifest(knowledge_base_id) is not None and not self.index_module.manifest_matches(expected_manifest):
            self._full_rebuild_locked(knowledge_base_id)
            return
        
        # 加载当前资料库全部叶子节点
        leaf_nodes = self._load_leaf_nodes_from_docstore(knowledge_base_id)
        # 加载当前索引状态
        vector_index = self.index_module.load_index(expected_manifest, storage_context=self.data_module.storage_context)
        # 如果索引状态为空且有叶子节点，构建索引
        if vector_index is None and leaf_nodes:
            vector_index = self.index_module.build_vector_index(
                leaf_nodes, storage_context=self.data_module.storage_context, reset=True,
            )
        # 保存当前索引状态
        self.index_module.save_manifest(self._build_expected_manifest(knowledge_base_id))
        if vector_index is not None:
            self._refresh_retrieval(vector_index, leaf_nodes, knowledge_base_id)

    def _index_document_locked(self, document_id: str, *, retire_replacements: bool) -> dict:
        """
        对单个注册文档执行解析、向量写入、docstore 写入和检索刷新
        Args:
            document_id: 待索引的文档 ID
            retire_replacements: 是否删除旧索引条目，保留新索引条目
        Returns:
            公开文档记录
        """
        # 获取文档记录
        record = self.document_registry.get(document_id)
        # 确保该文档所属知识库的模块已初始化
        self._ensure_modules(record.knowledge_base_id)
        assert self.data_module is not None
        assert self.index_module is not None
        # 切换到指定知识库上下文
        knowledge_base_id = self._configure_knowledge_base_scope_locked(record.knowledge_base_id)
        # 解析文档内容，生成节点
        all_nodes, leaf_nodes = self.data_module.chunk_single_document(record)
        if not all_nodes:
            raise ValueError(f"文档 {document_id!r} 未生成可索引节点")

        self._ensure_incremental_index_ready_locked(knowledge_base_id)

        # 加载当前资料库全部节点
        existing_nodes = self.llama_docstore.load_all_nodes(knowledge_base_id) if self.llama_docstore is not None else []
        # 筛选出当前文档的所有节点
        previous_document_nodes = [node for node in existing_nodes if (node.metadata or {}).get("document_id") == document_id]
        # 筛选出其他文档的所有节点
        stored_nodes = [node for node in existing_nodes if (node.metadata or {}).get("document_id") != document_id]
        # 合并所有节点
        self.data_module.load_prepared_nodes(stored_nodes + all_nodes)
        if self.llama_docstore is not None:
            # 写入当前文档所有节点到 docstore
            self.llama_docstore.add_documents(all_nodes)
        # 重写当前文档的 BM25 分块词频状态
        self._replace_bm25_document_chunks_locked(knowledge_base_id, document_id, leaf_nodes)

        try:
            # 删除当前文档的所有向量索引条目
            self.index_module.delete_vectors_by_document_id(document_id)
            # 写入当前文档所有叶子节点到索引
            vector_index = self.index_module.upsert_leaf_nodes(leaf_nodes, storage_context=self.data_module.storage_context)
        except Exception:
            if self.bm25_store is not None:
                # 删除当前文档的所有 BM25 索引条目
                self.bm25_store.delete_document(knowledge_base_id, document_id)
            if self.llama_docstore is not None and previous_document_nodes:
                # 写入当前文档所有节点到 docstore
                self.llama_docstore.add_documents(previous_document_nodes)
            elif self.llama_docstore is not None:
                # 删除当前文档的所有节点store 条目
                self.llama_docstore.delete_nodes_by_document(document_id, knowledge_base_id)
            raise
        # 刷新索引
        self._reload_from_store_and_refresh_locked(knowledge_base_id, vector_index)
        # 标记文档为已索引
        self.document_registry.mark_indexed(document_id, chunk_count=len(leaf_nodes))
        if retire_replacements:
            # 删除旧索引条目
            self._retire_replaced_documents_locked(record)
        # 返回公开文档记录
        return self._public_document(document_id)

    def _delete_document_index_entries_locked(self, knowledge_base_id: str, document_id: str) -> None:
        """删除指定 document_id 的向量、BM25 和 docstore 条目"""
        assert self.index_module is not None
        # 切换到指定知识库上下文
        knowledge_base_id = self._configure_knowledge_base_scope_locked(knowledge_base_id)
        # 删除向量索引条目
        self.index_module.delete_vectors_by_document_id(document_id)
        # 删除 BM25 索引条目
        if self.bm25_store is not None:
            self.bm25_store.delete_document(knowledge_base_id, document_id)
        # 删除 docstore 条目
        if self.llama_docstore is not None:
            self.llama_docstore.delete_nodes_by_document(document_id, knowledge_base_id)

    def _load_leaf_nodes_from_docstore(self, knowledge_base_id: str) -> List[TextNode]:
        """从 llama_docstore 恢复全部层级节点并返回叶子节点"""
        assert self.data_module is not None
        # 切换到指定知识库上下文
        knowledge_base_id = self._configure_knowledge_base_scope_locked(knowledge_base_id)

        if self.llama_docstore is not None:
            all_nodes = self.llama_docstore.load_all_nodes(knowledge_base_id)
            if all_nodes:
                # 返回所有叶子节点
                return self.data_module.load_prepared_nodes(all_nodes)
        return []

    def _reload_from_store_and_refresh_locked(self, knowledge_base_id: str, vector_index: Optional[Any] = None) -> Any:
        """
        从 docstore 重新加载叶子节点并刷新检索引擎
        Args:
            knowledge_base_id: 知识库 ID
            vector_index: 可选的向量索引条目，用于更新索引
        Returns:
            更新后的向量索引条目
        """
        assert self.data_module is not None
        assert self.index_module is not None
        # 切换到指定知识库上下文
        knowledge_base_id = self._configure_knowledge_base_scope_locked(knowledge_base_id)
        
        # 从 docstore 加载所有叶子节点
        leaf_nodes = self._load_leaf_nodes_from_docstore(knowledge_base_id)
        # 构建索引清单
        manifest = self._build_expected_manifest(knowledge_base_id)
        # 加载索引
        loaded_index = self.index_module.load_index(manifest, storage_context=self.data_module.storage_context)

        if loaded_index is not None:
            vector_index = loaded_index
        elif vector_index is None:
            if leaf_nodes:
                # 构建向量索引
                vector_index = self.index_module.build_vector_index(
                    leaf_nodes, storage_context=self.data_module.storage_context, reset=True,
                )
            else:
                # 清空索引，返回空索引句柄
                vector_index = self.index_module.clear_index(storage_context=self.data_module.storage_context)
        # 保存索引清单
        self.index_module.save_manifest(manifest)
        # 刷新检索引擎
        self._refresh_retrieval(vector_index, leaf_nodes, knowledge_base_id)
        return vector_index

    def _retire_replaced_documents_locked(self, indexed_record: Any) -> None:
        """
        新同名文档索引成功后，删除同资料库同名旧文档的文件、向量和节点
        Args:
            indexed_record: 新索引成功的文档记录
        Returns:
            无返回值
        """
        # 查找所有同资料库同名旧文档的记录
        replaced_records = [
            record
            for record in self.document_registry.records.values()
            if record.document_id != indexed_record.document_id
            and record.knowledge_base_id == indexed_record.knowledge_base_id
            and record.filename == indexed_record.filename
            and record.status != "deleted"
        ]
        if not replaced_records:
            return
        for record in replaced_records:
            # 删除旧文档的所有向量索引
            self._delete_document_index_entries_locked(record.knowledge_base_id, record.document_id)
            # 删除旧文档的托管文件
            self._managed_source_files().delete_managed_source_file(record.source_path)
            # 标记旧文档已删除
            self.document_registry.mark_deleted(record.document_id)
        # 刷新检索状态
        self._reload_from_store_and_refresh_locked(indexed_record.knowledge_base_id)

    def _refresh_retrieval(self, vector_index: Any, leaf_nodes: List[TextNode], knowledge_base_id: str) -> None:
        """
        用当前向量索引和叶子节点重建全部 LlamaIndex 检索和查询引擎
        Args:
            vector_index: 向量索引对象
            leaf_nodes: 叶子节点列表
            knowledge_base_id: 知识库 ID
        Returns:
            无返回值
        """
        assert self.data_module is not None
        scope = self.knowledge_base_scope(knowledge_base_id)
        runtime = getattr(self, "kb_runtimes", {}).get(scope.runtime_cache_key)
        # 如果没有叶子节点，清空所有检索器和引擎
        if not leaf_nodes:
            if runtime is not None:
                runtime.hybrid_retriever = None
                runtime.auto_merge_retriever = None
                runtime.knowledge_query_engine = None
                runtime.router_engine = None
                self._activate_runtime(runtime)
            else:
                self.hybrid_retriever = None
                self.auto_merge_retriever = None
                self.knowledge_query_engine = None
                self.router_engine = None
            return
        # 构建知识引擎组装器
        engines = build_knowledge_engines(
            vector_index=vector_index, # 向量索引对象
            storage_context=self.data_module.storage_context, # 存储上下文
            config=self.config, # 系统配置
            reranker=self.reranker, # 重排序模型
            llm=getattr(self.generation_module, "llm", None) if self.generation_module is not None else None, # LLM 模型
        )
        llm = getattr(self.generation_module, "llm", None) if self.generation_module is not None else None
        router_engine = build_top_router(system=self, llm=llm, knowledge_base_id=scope.knowledge_base_id)
        if runtime is not None:
            runtime.hybrid_retriever = engines.hybrid_retriever
            runtime.auto_merge_retriever = engines.auto_merge_retriever
            runtime.knowledge_query_engine = engines.knowledge_query_engine
            runtime.router_engine = router_engine
            self._activate_runtime(runtime)
        else:
            self.hybrid_retriever = engines.hybrid_retriever
            self.auto_merge_retriever = engines.auto_merge_retriever
            self.knowledge_query_engine = engines.knowledge_query_engine
            self.router_engine = router_engine

    def _public_document(self, document_id: str) -> dict:
        """
        获取指定文档对外展示的注册表记录
        Args:
            document_id: 文档 ID
        Returns:
            文档公开状态字典
        """
        for record in self.document_registry.list_public():
            if record["document_id"] == document_id:
                return record
        return self.document_registry.get(document_id).to_dict()

    @staticmethod
    def _format_exception(exc: Exception) -> str:
        """
        将异常格式化为适合写入文档注册表的错误文本
        Args:
            exc: 捕获到的异常
        Returns:
            包含异常类型和消息的字符串
        """
        return f"{exc.__class__.__name__}: {exc}"

    def run_interactive(self):
        """
        启动命令行交互式问答循环
        Returns:
            无返回值，结果直接输出到终端
        """
        print("=" * 60)
        print("FinRAG 金融资料库问答系统")
        print("=" * 60)
        self.initialize_system()
        self.build_knowledge_base()
        while True:
            try:
                question = input("\n您的问题: ").strip()
                if question.lower() in {"退出", "quit", "exit"}:
                    break
                if not question:
                    print("请输入问题内容，或输入'退出'结束")
                    continue
                response = self.ask_question(question, return_sources=True, knowledge_base_id=self.config.knowledge_base_id)
                print(response.answer)
            except (KeyboardInterrupt, EOFError):
                break
