"""文档加载、层级节点构建和证据窗口"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from llama_index.core import Document, StorageContext
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.schema import BaseNode, NodeRelationship, RelatedNodeInfo, TransformComponent
from finrag.core.node_schema import TextNode
from finrag.ingestion.parsers import SUPPORTED_SUFFIXES, ParserRegistry, is_path_within, load_documents as load_financial_documents

logger = logging.getLogger(__name__)

class DataPreparationModule:
    """负责从数据源加载文档、构建层级节点、生成检索索引和构造生成上下文窗口的模块"""

    def __init__(
        self,
        data_path: str,
        *,
        knowledge_base_id: str = "default",
        chunk_size: int = 300,
        chunk_overlap: int = 60,
        document_registry: Any = None,
        docstore: Optional[Any] = None,
    ):
        """
        初始化数据准备模块
        Args:
            data_path: 文档数据目录路径
            knowledge_base_id: 默认资料库 ID
            chunk_size: 文档切分块大小
            chunk_overlap: 文档切分块重叠大小
            document_registry: 可选文档注册表
            docstore: 可选 LlamaIndex docstore adapter（PostgreSQLLlamaIndexDocumentStore）
        """
        self.data_path = data_path # 文档数据目录路径
        self.knowledge_base_id = knowledge_base_id # 资料库 ID
        self.chunk_size = int(chunk_size) # 文档切分块大小
        self.chunk_overlap = int(chunk_overlap) # 文档切分块重叠大小
        self.document_registry = document_registry # 文档注册表
        self._docstore = docstore # 可选 LlamaIndex docstore adapter
        self.documents: List[Document] = [] # 已加载的 Document 列表
        self.all_nodes: List[TextNode] = [] # 构建的全部层级节点列表
        self.chunks: List[TextNode] = [] # 仅用于检索的叶子节点列表
        # 初始化 StorageContext 容器
        self.storage_context: StorageContext = self._make_storage_context()

    def load_documents(self) -> List[Document]:
        """
        从数据目录或文档注册表加载金融资料文档
        Returns:
            LlamaIndex Document 列表
        """
        logger.info("正在从 %s 加载金融资料文档...", self.data_path)
        self.documents = load_financial_documents(
            self.data_path, # 文档数据目录路径
            knowledge_base_id=self.knowledge_base_id, # 资料库 ID
            document_registry=self.document_registry, # 文档注册表
        )
        # 重置已加载状态
        self._reset_loaded_state()
        logger.info("成功加载 %s 个父文档", len(self.documents))
        return self.documents

    def _make_storage_context(self) -> StorageContext:
        """根据配置创建 StorageContext"""
        # 如果提供了 docstore，创建一个定了 docstore 的 StorageContext
        if self._docstore is not None:
            return StorageContext.from_defaults(docstore=self._docstore)
        # 如果没有提供 docstore，创建一个默认的 StorageContext
        return StorageContext.from_defaults()

    def _reset_loaded_state(self) -> None:
        """清空已构建节点、叶子分块和存储上下文"""
        self.all_nodes = []
        self.chunks = []
        self.storage_context = self._make_storage_context()

    def load_record_documents(self, record: Any) -> List[Document]:
        """
        只加载一个文档注册记录对应的源文件
        Args:
            record: DocumentRecord 文档注册记录
        Returns:
            该文档解析得到的 LlamaIndex Document 列表
        """
        path = Path(record.source_path)
        # 检查文档是否已删除或源文件不存在
        if record.status == "deleted" or not path.exists():
            return []
        # 检查文档是否在可信目录内
        source_root = self._trusted_root_for_path(path)
        if source_root is None:
            logger.warning("跳过可信目录外的注册文档: %s", path)
            return []
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            return []
        # 加载文档内容
        parsed_docs = ParserRegistry.default().load(
            path,
            knowledge_base_id=record.knowledge_base_id,
            data_root=source_root,
        )
        # 更新文档元数据
        for doc in parsed_docs:
            doc.metadata.update(
                {
                    "document_id": record.document_id,
                    "filename": record.filename,
                    "file_type": record.file_type,
                    "knowledge_base_id": record.knowledge_base_id,
                }
            )
        return parsed_docs

    def _trusted_root_for_path(self, path: Path) -> Optional[Path]:
        """
        返回包含指定路径的可信源文件根目录
        Args:
            path: 待检查的文档源路径
        Returns:
            命中的可信根目录，未命中时返回 None
        """
        root = Path(self.data_path)
        return root if is_path_within(path, root) else None

    def chunk_single_document(self, record: Any) -> tuple[List[TextNode], List[TextNode]]:
        """
        只切分一个文档注册记录，返回全部层级节点和叶子节点
        Args:
            record: DocumentRecord 文档注册记录
        Returns:
            (all_nodes, leaf_nodes)
        """
        # 加载文档内容
        documents = self.load_record_documents(record)
        if not documents:
            return [], []
        # 构建层级节点
        return self._build_hierarchical_nodes(documents)

    def load_prepared_nodes(self, all_nodes: List[TextNode]) -> List[TextNode]:
        """从已持久化的层级节点恢复当前数据准备模块状态"""
        self.storage_context = self._make_storage_context()
        # 将所有节点添加到存储上下文的文档存储中
        self.storage_context.docstore.add_documents(all_nodes)
        self.all_nodes = list(all_nodes)
        # 保存所有叶子节点
        self.chunks = get_leaf_nodes(all_nodes)
        return self.chunks

    def _build_hierarchical_nodes(self, documents: List[Document]) -> tuple[List[TextNode], List[TextNode]]:
        """
        根据传入文档构建层级节点，不直接修改模块状态
        Args:
            documents: 待切分的 Document 列表
        Returns:
            (all_nodes, leaf_nodes)
        """
        # 检查是否使用语义分块器
        use_semantic = getattr(self, "_use_semantic_chunking", False)
        if use_semantic:
            all_nodes = self._chunk_with_semantic_splitter(documents)
        # 否则使用默认的层级分块器
        else:
            # 初始化层级分块器
            parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=[max(self.chunk_size * 4, self.chunk_size), max(self.chunk_size * 2, self.chunk_size), self.chunk_size],
                chunk_overlap=self.chunk_overlap,
                include_metadata=False, # 生成节点时，节点文本内容不包含文档元数据
            )
            all_nodes = [
                node
                for node in parser.get_nodes_from_documents(documents)
                if isinstance(node, TextNode)
            ]
        # 为层级节点重写稳定 chunk_id，并补充父子层级和溯源元数据
        self._assign_finrag_metadata(all_nodes)
        # 返回全部层级节点和叶子节点
        return all_nodes, get_leaf_nodes(all_nodes)

    def _chunk_with_semantic_splitter(self, documents: List[Document]) -> List[TextNode]:
        """
        使用语义分块器对文档进行分块，返回全部层级节点
        Args:
            documents: 待切分的 Document 列表
        Returns:
            该文档解析得到的 LlamaIndex TextNode 列表
        """
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        # 初始化语义分块器
        parser = SemanticSplitterNodeParser.from_defaults(
            embed_model=getattr(self, "_embed_model", None),
            breakpoint_percentile_threshold=95, # 断点阈值，用于确定分块位置
            buffer_size=1, # 语义比较时的上下文窗口大小
        )
        nodes: List[TextNode] = []
        for doc in documents:
            # 对每个文档进行分块
            for node in parser.get_nodes_from_documents([doc]):
                if isinstance(node, TextNode):
                    nodes.append(node)
        return nodes

    def _assign_finrag_metadata(self, all_nodes: List[TextNode]) -> None:
        """
        为层级节点重写稳定 chunk_id，并补充父子层级和溯源元数据
        Args:
            all_nodes: HierarchicalNodeParser 生成的全部层级节点
        Returns:
            无返回值，直接修改节点 ID、relationships 和 metadata
        """
        # 旧节点 ID 映射到新节点 ID
        old_to_new: Dict[str, str] = {}
        # 旧节点 ID 映射到新节点层级
        level_by_old_id: Dict[str, int] = {}
        # 每个文档每个层级的计数器
        counters: Dict[tuple[str, int], int] = defaultdict(int) # 默认值为 0
        for node in all_nodes:
            level = self._relationship_level(node)
            old_id = node.node_id
            metadata = self._node_metadata(node) # 合并节点自身 metadata 与 SOURCE 关系中的文档 metadata
            document_id = str(metadata.get("document_id") or "")
            chunk_idx = counters[(document_id, level)]
            counters[(document_id, level)] += 1
            old_to_new[old_id] = self._make_chunk_id(document_id, level, chunk_idx, node.text)
            level_by_old_id[old_id] = level

        # 重写节点 ID
        for node in all_nodes:
            old_id = node.node_id
            node.node_id = old_to_new[old_id]
        # 重写节点关系中的 node_id 引用
        for node in all_nodes:
            node.relationships = self._remap_relationships(node.relationships, old_to_new)

        # 构建 node_id 到节点对象的映射
        nodes_by_id = {node.node_id: node for node in all_nodes}
        # 构建 leaf chunk ID 到索引的映射
        leaf_chunk_idx = {node.node_id: index for index, node in enumerate(get_leaf_nodes(all_nodes))}
        # 构建每个文档每个层级的索引计数器
        level_counters: Dict[tuple[str, int], int] = defaultdict(int)
        for node in all_nodes:
            metadata = self._node_metadata(node)
            document_id = str(metadata.get("document_id") or "")
            # 从旧节点 ID 映射中获取旧层级
            old_level = level_by_old_id.get(next((old for old, new in old_to_new.items() if new == node.node_id), ""), 3)
            # 从节点关系中获取新层级，优先级高于旧层级
            level = self._relationship_level(node) or old_level
            parent_id = node.parent_node.node_id if node.parent_node else node.node_id
            root_id = self._root_node_id(node, nodes_by_id)
            if level == 3:
                chunk_idx = leaf_chunk_idx[node.node_id]
            else:
                chunk_idx = level_counters[(document_id, level)]
                level_counters[(document_id, level)] += 1
            metadata.update(
                {
                    "document_id": metadata.get("document_id", document_id), # 文档ID
                    "knowledge_base_id": metadata.get("knowledge_base_id", self.knowledge_base_id),
                    "chunk_id": node.node_id, # 节点ID
                    "parent_chunk_id": parent_id, # 父节点ID
                    "root_chunk_id": root_id, # 根节点ID
                    "chunk_level": level, # 节点层级
                    "chunk_idx": chunk_idx, # 同层级内的序号
                }
            )
            node.metadata = metadata

    @staticmethod
    def _relationship_level(node: TextNode) -> int:
        """
        根据节点关系判断 FinRAG 层级编号
        Args:
            node: 待判断的 TextNode
        Returns:
            L1/L2/L3 对应的整数层级
        """
        if node.child_nodes: # 有子节点，为 L2 层
            return 2 if node.parent_node else 1
        return 3

    @staticmethod
    def _node_metadata(node: TextNode) -> Dict[str, Any]:
        """
        合并节点自身 metadata 与 SOURCE 关系中的文档 metadata
        Args:
            node: 待读取元数据的 TextNode
        Returns:
            合并后的元数据字典
        """
        metadata = dict(node.metadata or {})
        source = node.relationships.get(NodeRelationship.SOURCE) # SOURCE 关系通常指向原始文档节点
        if isinstance(source, RelatedNodeInfo):
            metadata = {**dict(source.metadata or {}), **metadata} # 用 SOURCE 中的 metadata 更新节点自身 metadata，后者优先覆盖同名字段
        return metadata

    @staticmethod
    def _remap_relationships(relationships: Dict[NodeRelationship, Any], old_to_new: Dict[str, str]) -> Dict[NodeRelationship, Any]:
        """
        将 LlamaIndex 节点关系中的旧 node_id 替换为 FinRAG 稳定 chunk_id
        Args:
            relationships: 原始节点关系字典
            old_to_new: 旧 node_id 到新 chunk_id 的映射
        Returns:
            重写后的关系字典
        """
        remapped: Dict[NodeRelationship, Any] = {}
        for relationship, value in relationships.items():
            if isinstance(value, list):
                remapped[relationship] = [DataPreparationModule._remap_related_node(info, old_to_new) for info in value]
            else:
                remapped[relationship] = DataPreparationModule._remap_related_node(value, old_to_new)
        return remapped

    @staticmethod
    def _remap_related_node(info: RelatedNodeInfo, old_to_new: Dict[str, str]) -> RelatedNodeInfo:
        """
        重写单个 RelatedNodeInfo 中的 node_id
        Args:
            info: 原始关系节点信息
            old_to_new: 旧 node_id 到新 chunk_id 的映射
        Returns:
            使用新 node_id 的 RelatedNodeInfo
        """
        node_id = old_to_new.get(info.node_id, info.node_id)
        return RelatedNodeInfo(node_id=node_id, node_type=info.node_type, metadata=info.metadata, hash=info.hash)

    @staticmethod
    def _root_node_id(node: TextNode, nodes_by_id: Dict[str, TextNode]) -> str:
        """
        沿 PARENT 关系向上查找当前节点所属根节点
        Args:
            node: 当前节点
            nodes_by_id: node_id 到节点对象的映射
        Returns:
            根节点 ID；关系缺失时返回可到达的最高层节点 ID
        """
        current = node
        seen: Set[str] = set()
        while current.parent_node and current.parent_node.node_id not in seen:
            seen.add(current.node_id)
            parent = nodes_by_id.get(current.parent_node.node_id)
            if parent is None:
                break
            current = parent
        return current.node_id

    @staticmethod
    def _make_chunk_id(document_id: str, chunk_level: int, chunk_idx: int, text: str, scope: str = "") -> str:
        """
        根据文档、层级、序号和文本生成稳定 chunk_id
        Args:
            document_id: 文档 ID
            chunk_level: 节点层级
            chunk_idx: 节点序号
            text: 节点文本
            scope: 可选命名空间
        Returns:
            chunk- 前缀的哈希 ID
        """
        digest = hashlib.md5(f"{document_id}:{scope}:{chunk_level}:{chunk_idx}:{text}".encode("utf-8")).hexdigest()
        return f"chunk-{digest}"

    def get_statistics(self) -> Dict[str, Any]:
        """
        汇总当前已加载文档和叶子分块的统计信息
        Returns:
            文档数、chunk 数、文件类型和平均 chunk 长度
        """
        file_types: Dict[str, int] = {}
        for doc in self.documents:
            metadata = doc.metadata or {}
            file_type = metadata.get("file_type") or "unknown"
            file_types[file_type] = file_types.get(file_type, 0) + 1
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "file_types": file_types,
            "avg_chunk_size": sum(len(node.text) for node in self.chunks) / len(self.chunks) if self.chunks else 0,
        }


class FinRAGMetadataTransform(TransformComponent):
    """
    将 FinRAG 元数据赋值到层级节点的转换函数
    """

    _data_module: Any = PrivateAttr()

    def __init__(self, data_module: Any):
        super().__init__()
        self._data_module = data_module

    @classmethod
    def class_name(cls) -> str:
        return "finrag_metadata_transform"

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        # 赋值 FinRAG 元数据到层级节点
        node_list = list(nodes)
        self._data_module._assign_finrag_metadata(node_list)
        return node_list


def build_ingestion_pipeline(
    data_module: Any,
    embed_model: Any,
    vector_store: Any,
    docstore: Any,
    *,
    use_semantic_chunking: bool = False,
) -> Any:
    """
    构建 LlamaIndex IngestionPipeline 用于 FinRAG 索引流程
    Args:
        data_module: 数据模块，包含文档加载、分块和元数据处理
        embed_model: 嵌入模型，用于将文本转换为向量表示
        vector_store: 向量存储，用于存储和检索向量表示
        docstore: 文档存储，用于存储文档元数据
        use_semantic_chunking: 是否使用语义分块（默认 False）
    Returns:
        构建好的 IngestionPipeline
    """
    from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy

    # 分块大小
    chunk_sizes = [
        max(data_module.chunk_size * 4, data_module.chunk_size),
        max(data_module.chunk_size * 2, data_module.chunk_size),
        data_module.chunk_size,
    ]
    # 转换链
    transformations: list[Any] = []

    # 使用语义分块
    if use_semantic_chunking:
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        # 添加语义分块节点解析器
        transformations.append(
            SemanticSplitterNodeParser.from_defaults(
                embed_model=embed_model,
                breakpoint_percentile_threshold=95, # 语义分块阈值
                buffer_size=1, # 上下文窗口大小
            )
        )
    # 默认使用层级分块
    else:
        # 添加层级分块节点解析器
        transformations.append(
            HierarchicalNodeParser.from_defaults(
                chunk_sizes=chunk_sizes,
                chunk_overlap=data_module.chunk_overlap,
                include_metadata=False, # 生成节点时，节点文本内容不包含文档元数据
            )
        )
    # 添加 FinRAG 元数据赋值转换函数
    transformations.append(FinRAGMetadataTransform(data_module))
    # 添加嵌入模型
    transformations.append(embed_model)

    return IngestionPipeline(
        transformations=transformations, # 转换链
        vector_store=vector_store, # 向量存储
        docstore=docstore, # 文档存储
        docstore_strategy=DocstoreStrategy.UPSERTS, # 文档存储策略，更新或插入文档
    )
