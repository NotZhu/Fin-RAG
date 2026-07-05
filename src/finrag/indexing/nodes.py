"""文档加载、层级节点构建和证据窗口"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from llama_index.core import Document, StorageContext
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.node_parser import get_leaf_nodes
from llama_index.core.schema import BaseNode, NodeRelationship, RelatedNodeInfo, TransformComponent
from finrag.core.node_schema import TextNode
from finrag.ingestion.docling_loader import load_docling_documents
from finrag.ingestion.parsers import SUPPORTED_SUFFIXES, is_path_within, load_documents as load_financial_documents

logger = logging.getLogger(__name__)


class DataPreparationModule:
    """负责从数据源加载文档、构建层级节点、生成检索索引和构造生成上下文窗口的模块"""

    def __init__(
        self,
        data_path: str,
        *,
        knowledge_base_id: str = "default",
        document_registry: Any = None,
        docstore: Optional[Any] = None,
    ):
        """
        初始化数据准备模块
        Args:
            data_path: 文档数据目录路径
            knowledge_base_id: 默认资料库 ID
            document_registry: 可选文档注册表
            docstore: 可选 LlamaIndex docstore adapter（PostgreSQLLlamaIndexDocumentStore）
        """
        self.data_path = data_path
        self.knowledge_base_id = knowledge_base_id
        self.document_registry = document_registry
        self._docstore = docstore
        self.documents: List[Document] = []
        self.all_nodes: List[TextNode] = []
        self.chunks: List[TextNode] = []
        self.storage_context: StorageContext = self._make_storage_context()

    def load_documents(self) -> List[Document]:
        """
        从数据目录或文档注册表加载金融资料文档
        Returns:
            LlamaIndex Document 列表
        """
        logger.info("正在从 %s 加载金融资料文档...", self.data_path)
        self.documents = load_financial_documents(
            self.data_path,
            knowledge_base_id=self.knowledge_base_id,
            document_registry=self.document_registry,
        )
        self._reset_loaded_state()
        logger.info("成功加载 %s 个父文档", len(self.documents))
        return self.documents

    def _make_storage_context(self) -> StorageContext:
        """根据配置创建 StorageContext"""
        if self._docstore is not None:
            return StorageContext.from_defaults(docstore=self._docstore)
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
        # 使用 Docling 统一解析注册文档内容
        parsed_docs = load_docling_documents(
            path,
            knowledge_base_id=record.knowledge_base_id,
            data_root=source_root,
        )
        # 使用注册表中的生命周期元数据覆盖解析器生成值
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
        leaf_source_nodes = self._parse_docling_leaf_nodes(documents)
        all_nodes = [node for node in HierarchyBuilder()(leaf_source_nodes) if isinstance(node, TextNode)]
        if not all_nodes:
            return [], []
        self._assign_node_metadata(all_nodes)
        return all_nodes, get_leaf_nodes(all_nodes)

    def _parse_docling_leaf_nodes(self, documents: List[Document]) -> List[TextNode]:
        """
        使用 DoclingNodeParser 将 Docling JSON Document 转为叶子节点
        Args:
            documents: 待切分的 Document 列表
        Returns:
            DoclingNodeParser 生成的叶子节点列表
        """
        parser = _make_docling_node_parser()
        nodes: List[TextNode] = []
        # 从所有文档中获取所有节点
        for node in parser.get_nodes_from_documents(documents):
            # 过滤出文本节点
            if isinstance(node, TextNode):
                nodes.append(node)
        return nodes

    def _assign_node_metadata(self, all_nodes: List[TextNode]) -> None:
        """
        为层级节点重写稳定 chunk_id，并补充父子层级和溯源元数据
        Args:
            all_nodes: 层级构造器生成的全部节点
        Returns:
            无返回值，直接修改节点 ID、relationships 和 metadata
        """
        old_to_new: Dict[str, str] = {}
        level_by_old_id: Dict[str, int] = {}
        counters: Dict[tuple[str, int], int] = defaultdict(int)
        for node in all_nodes:
            level = self._relationship_level(node)
            old_id = node.node_id
            metadata = self._node_metadata(node)
            document_id = str(metadata.get("document_id") or "")
            chunk_idx = counters[(document_id, level)]
            counters[(document_id, level)] += 1
            old_to_new[old_id] = self._make_chunk_id(document_id, level, chunk_idx, node.text)
            level_by_old_id[old_id] = level

        for node in all_nodes:
            old_id = node.node_id
            node.node_id = old_to_new[old_id]
        for node in all_nodes:
            node.relationships = self._remap_relationships(node.relationships, old_to_new)

        new_to_old = {new_id: old_id for old_id, new_id in old_to_new.items()}
        nodes_by_id = {node.node_id: node for node in all_nodes}
        leaf_chunk_idx = {node.node_id: index for index, node in enumerate(get_leaf_nodes(all_nodes))}
        level_counters: Dict[tuple[str, int], int] = defaultdict(int)
        for node in all_nodes:
            metadata = self._node_metadata(node)
            document_id = str(metadata.get("document_id") or "")
            old_level = level_by_old_id.get(new_to_old.get(node.node_id, ""), 3)
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
                    "document_id": metadata.get("document_id", document_id),
                    "knowledge_base_id": metadata.get("knowledge_base_id", self.knowledge_base_id),
                    "chunk_id": node.node_id,
                    "parent_chunk_id": parent_id,
                    "root_chunk_id": root_id,
                    "chunk_level": level,
                    "chunk_idx": chunk_idx,
                }
            )
            node.metadata = metadata

    @staticmethod
    def _relationship_level(node: TextNode) -> int:
        """
        根据节点关系判断层级编号
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
        将 LlamaIndex 节点关系中的旧 node_id 替换为稳定 chunk_id
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


class HierarchyBuilder(TransformComponent):
    """根据 Docling 叶子节点构建文档/章节/叶子层级"""

    @classmethod
    def class_name(cls) -> str:
        return "hierarchy_builder"

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        """
        将 Docling 叶子节点按文档和章节组织为三层结构
        Args:
            nodes: DoclingNodeParser 生成的叶子节点
        Returns:
            root、section 和 leaf 组成的节点列表
        """
        leaf_nodes = [node for node in nodes if isinstance(node, TextNode)]
        if not leaf_nodes:
            return []
        return self.build(leaf_nodes)

    def build(self, leaf_nodes: Sequence[TextNode]) -> List[TextNode]:
        """
        将叶子节点按文档和章节组织为三层结构
        Args:
            leaf_nodes: DoclingNodeParser 生成的叶子节点
        Returns:
            root、section 和 leaf 组成的节点列表
        """
        documents: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        for leaf in leaf_nodes:
            raw_metadata = dict(leaf.metadata or {})
            metadata = self._clean_metadata(raw_metadata)
            section_title = self._section_title(raw_metadata)
            metadata["section_title"] = section_title
            page_number = self._page_number(raw_metadata)
            if page_number is not None:
                metadata["page_number"] = page_number
            leaf.metadata = metadata
            leaf.relationships = {
                relationship: value
                for relationship, value in leaf.relationships.items()
                if relationship not in {NodeRelationship.PARENT, NodeRelationship.CHILD, NodeRelationship.SOURCE}
            }

            document_id = str(metadata.get("document_id") or "")
            document_entry = documents.setdefault(
                document_id,
                {
                    "metadata": self._document_metadata(metadata),
                    "sections": OrderedDict(),
                },
            )
            sections = document_entry["sections"]
            section_entry = sections.setdefault(
                section_title,
                {
                    "metadata": self._section_metadata(metadata),
                    "leaves": [],
                },
            )
            section_entry["leaves"].append(leaf)

        all_nodes: List[TextNode] = []
        for document_entry in documents.values():
            section_pairs: List[tuple[TextNode, List[TextNode]]] = []
            for section_title, section_entry in document_entry["sections"].items():
                section_leaves = list(section_entry["leaves"])
                section_node = TextNode(
                    text=self._join_text([section_title, *[leaf.text for leaf in section_leaves]]),
                    metadata=dict(section_entry["metadata"]),
                )
                for leaf in section_leaves:
                    leaf.relationships[NodeRelationship.PARENT] = section_node.as_related_node_info()
                section_node.relationships[NodeRelationship.CHILD] = [leaf.as_related_node_info() for leaf in section_leaves]
                section_pairs.append((section_node, section_leaves))
            root_node = TextNode(
                text=self._join_text([section.text for section, _ in section_pairs]),
                metadata=dict(document_entry["metadata"]),
            )
            for section_node, _section_leaves in section_pairs:
                section_node.relationships[NodeRelationship.PARENT] = root_node.as_related_node_info()
            root_node.relationships[NodeRelationship.CHILD] = [section.as_related_node_info() for section, _ in section_pairs]

            all_nodes.append(root_node)
            for section_node, section_leaves in section_pairs:
                all_nodes.append(section_node)
                all_nodes.extend(section_leaves)
        return all_nodes

    @staticmethod
    def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """清理元数据，只保留需要持久化的基础元数据"""
        clean: Dict[str, Any] = {}
        for key in ("knowledge_base_id", "document_id", "source_path", "filename", "file_type", "parser_name"):
            value = metadata.get(key)
            if value is not None:
                clean[key] = value
        clean.setdefault("parser_name", "docling")
        return clean

    @staticmethod
    def _document_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """提取文档级元数据"""
        return {key: value for key, value in metadata.items() if key not in {"page_number", "section_title"}}

    @staticmethod
    def _section_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """提取章节级元数据"""
        section_metadata = dict(metadata)
        section_metadata.pop("page_number", None)
        return section_metadata

    @staticmethod
    def _section_title(metadata: Dict[str, Any]) -> str:
        """读取最近章节标题"""
        # 优先使用显式章节标题
        section_title = metadata.get("section_title")
        if section_title:
            return str(section_title)
        # 如果没有显式章节标题，尝试使用 headings 列表中的最后一个非空标题
        headings = metadata.get("headings")
        if isinstance(headings, list):
            for heading in reversed(headings):
                if str(heading or "").strip():
                    return str(heading).strip()
        if headings:
            return str(headings).strip()
        return "全文"

    @staticmethod
    def _page_number(metadata: Dict[str, Any]) -> Optional[int]:
        """读取该节点第一个可用页码"""
        # 优先尝试直接获取页码字段
        for key in ("page_number", "page_no", "page"):
            value = metadata.get(key)
            page_number = HierarchyBuilder._to_int(value)
            if page_number is not None:
                return page_number
        # 如果没有直接页码字段，尝试从 doc_items 中获取
        for item in HierarchyBuilder._iter_doc_items(metadata.get("doc_items")):
            # 溯源信息
            provenance = item.get("prov") or item.get("provenance") or []
            # 处理单个字典情况
            if isinstance(provenance, dict):
                provenance = [provenance]
            for entry in provenance:
                if not isinstance(entry, dict):
                    continue
                for key in ("page_no", "page_number", "page"):
                    page_number = HierarchyBuilder._to_int(entry.get(key))
                    if page_number is not None:
                        return page_number
        return None

    @staticmethod
    def _iter_doc_items(value: Any) -> List[dict]:
        """规范化 Docling doc_items 列表"""
        if isinstance(value, dict):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return []

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        """将页码转换为整数"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _join_text(parts: Sequence[str]) -> str:
        """合并节点文本"""
        return "\n\n".join(str(part or "").strip() for part in parts if str(part or "").strip())


class MetadataTransform(TransformComponent):
    """
    将节点元数据赋值到层级节点的转换函数
    """

    _data_module: Any = PrivateAttr()

    def __init__(self, data_module: Any):
        super().__init__()
        self._data_module = data_module

    @classmethod
    def class_name(cls) -> str:
        return "metadata_transform"

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        # 赋值元数据到层级节点
        node_list = list(nodes)
        self._data_module._assign_node_metadata(node_list)
        return node_list


def _make_docling_node_parser() -> Any:
    """延迟创建 DoclingNodeParser，避免导入阶段强依赖扩展包"""
    try:
        from llama_index.node_parser.docling import DoclingNodeParser
    except Exception as exc:
        raise RuntimeError("Docling 分块需要安装 llama-index-node-parser-docling 依赖") from exc
    return DoclingNodeParser()


def build_ingestion_pipeline(
    data_module: Any,
    embed_model: Any,
) -> Any:
    """
    构建 LlamaIndex IngestionPipeline 用于 FinRAG 索引流程
    Args:
        data_module: 数据模块，包含文档加载、分块和元数据处理
        embed_model: 嵌入模型，用于将文本转换为向量表示
    Returns:
        构建好的 IngestionPipeline
    """
    from llama_index.core.ingestion import IngestionPipeline

    return IngestionPipeline(
        transformations=[
            _make_docling_node_parser(),
            HierarchyBuilder(),
            MetadataTransform(data_module),
            embed_model,
        ],
    )
