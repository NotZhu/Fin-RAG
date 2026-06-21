"""LlamaIndex 文档存储类"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore.types import BaseDocumentStore, RefDocInfo

from finrag.core.node_schema import TextNode

from ._common import _deserialize_node, utc_now_iso
from .db import Database, execute, executemany


class PostgreSQLLlamaIndexDocumentStore(BaseDocumentStore):
    """
    PostgreSQL 后端的 LlamaIndex BaseDocumentStore adapter 文档存储适配器
    """

    def __init__(self, database_url: str):
        self.db = Database(database_url)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """
        确保数据库表结构存在
        """
        
        with self.db.connect() as conn:
            execute(
                conn,
                # 创建分块表 finrag_chunks，包含分块 ID、文档 ID、知识库 ID、分块等级、分块索引、分块内容、更新时间等字段
                """
                CREATE TABLE IF NOT EXISTS finrag_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    knowledge_base_id TEXT NOT NULL,
                    chunk_level INTEGER NOT NULL,
                    chunk_idx INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )
            # 创建索引 idx_finrag_chunks_document，用于快速查询指定文档的所有分块
            execute(conn, "CREATE INDEX IF NOT EXISTS idx_finrag_chunks_document ON finrag_chunks(document_id)")
            # 创建索引 idx_finrag_chunks_kb_document，用于快速查询指定知识库的所有分块
            execute(conn, "CREATE INDEX IF NOT EXISTS idx_finrag_chunks_kb_document ON finrag_chunks(knowledge_base_id, document_id)")
            # 创建索引 idx_finrag_chunks_level，用于快速查询指定分块等级的所有分块
            execute(conn, "CREATE INDEX IF NOT EXISTS idx_finrag_chunks_level ON finrag_chunks(chunk_level)")
        
        with self.db.connect() as conn:
            execute(
                conn,
                # 创建参考文档表 finrag_ref_docs，包含参考文档 ID、节点 ID 列表的 JSON 字符串、文档元数据的 JSON 字符串等字段
                """
                CREATE TABLE IF NOT EXISTS finrag_ref_docs (
                    ref_doc_id TEXT PRIMARY KEY,
                    node_ids_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """,
            )

            execute(
                conn,
                # 创建 LlamaIndex 哈希表；接口里的 doc_id 在本实现中存为节点/分块 ID
                """
                CREATE TABLE IF NOT EXISTS finrag_llama_doc_hashes (
                    chunk_id TEXT PRIMARY KEY,
                    doc_hash TEXT NOT NULL
                )
                """,
            )

    @property
    def docs(self) -> Dict[str, BaseNode]:
        """
        获取所有节点的字典，键为节点 ID，值为节点对象
        """
        return {node.node_id: node for node in self._load_nodes()}


    def load_all_nodes(self, knowledge_base_id: str) -> List[TextNode]:
        """
        从数据库加载指定知识库的所有节点
        Args:
            knowledge_base_id: 知识库 ID
        Returns:
            所有节点的列表
        """
        return self._load_nodes(knowledge_base_id)

    def load_leaf_nodes(self, knowledge_base_id: str) -> List[TextNode]:
        """
        从数据库加载指定知识库的叶子节点
        Args:
            knowledge_base_id: 知识库 ID
        Returns:
            叶子节点列表
        """
        return [
            node
            for node in self.load_all_nodes(knowledge_base_id)
            if int((node.metadata or {}).get("chunk_level", 0) or 0) == 3
        ]

    def _load_nodes(self, knowledge_base_id: Optional[str] = None) -> List[TextNode]:
        """
        从数据库加载节点；knowledge_base_id 为空时用于 BaseDocumentStore 的全量 docs 属性
        """
        where_clause = ""
        params: tuple[str, ...] = ()
        if knowledge_base_id is not None:
            where_clause = "WHERE knowledge_base_id = %s"
            params = (knowledge_base_id,)
        with self.db.connect() as conn:
            rows = execute(
                conn,
                # 在 finrag_chunks 表中查询所有分块内容，按文档 ID、分块等级、分块索引、分块 ID 排序
                f"SELECT payload FROM finrag_chunks {where_clause} ORDER BY document_id, chunk_level, chunk_idx, chunk_id",
                params,
            ).fetchall()
        return [_deserialize_node(row[0]) for row in rows]

    def get_node(self, node_id: str) -> Optional[TextNode]:
        """
        根据节点 ID 获取节点对象
        Args:
            node_id: 节点 ID
        Returns:
            节点对象，如果不存在则返回 None
        """
        with self.db.connect() as conn:
            row = execute(conn, "SELECT payload FROM finrag_chunks WHERE chunk_id = %s", (node_id,)).fetchone()
        if row is None:
            return None
        return _deserialize_node(row[0])

    def replace_document_nodes(self, document_id: str, nodes: List[TextNode], knowledge_base_id: str) -> None:
        """
        替换指定知识库中单个文档的所有节点
        Args:
            document_id: 文档 ID
            nodes: 新节点列表
            knowledge_base_id: 知识库 ID
        """
        self.delete_nodes_by_document(document_id, knowledge_base_id)
        self.add_documents(nodes)

    def delete_nodes_by_document(self, document_id: str, knowledge_base_id: str) -> None:
        """
        删除指定文档的所有分块、参考文档和哈希记录
        Args:
            document_id: 文档 ID
            knowledge_base_id: 知识库 ID
        """
        with self.db.connect() as conn:
            # 删除指定文档的所有哈希记录
            execute(
                conn,
                """
                DELETE FROM finrag_llama_doc_hashes
                WHERE chunk_id IN (
                    SELECT chunk_id FROM finrag_chunks
                    WHERE knowledge_base_id = %s AND document_id = %s
                )
                """,
                (knowledge_base_id, document_id),
            )
            # 删除指定文档的所有分块记录
            execute(
                conn,
                "DELETE FROM finrag_chunks WHERE knowledge_base_id = %s AND document_id = %s",
                (knowledge_base_id, document_id),
            )
            # 删除指定文档的所有参考文档记录
            execute(conn, "DELETE FROM finrag_ref_docs WHERE ref_doc_id = %s", (document_id,))

    def delete_knowledge_base(self, knowledge_base_id: str) -> None:
        """
        删除指定知识库的所有分块、参考文档和哈希记录
        Args:
            knowledge_base_id: 知识库 ID
        """
        with self.db.connect() as conn:
            # 删除指定知识库的所有分块记录
            execute(
                conn,
                """
                DELETE FROM finrag_llama_doc_hashes
                WHERE chunk_id IN (
                    SELECT chunk_id FROM finrag_chunks
                    WHERE knowledge_base_id = %s
                )
                """,
                (knowledge_base_id,),
            )
            # 删除指定知识库的所有参考文档记录
            execute(
                conn,
                """
                DELETE FROM finrag_ref_docs
                WHERE ref_doc_id IN (
                    SELECT DISTINCT document_id FROM finrag_chunks
                    WHERE knowledge_base_id = %s
                )
                """,
                (knowledge_base_id,),
            )
            # 删除指定知识库的所有分块记录
            execute(conn, "DELETE FROM finrag_chunks WHERE knowledge_base_id = %s", (knowledge_base_id,))

    def _chunk_ids_for_document(self, document_id: str, knowledge_base_id: str) -> List[str]:
        """
        在 finrag_chunks 表中获取指定文档的所有分块 ID 列表
        Args:
            document_id: 文档 ID
            knowledge_base_id: 知识库 ID
        Returns:
            所有分块 ID 列表
        """
        with self.db.connect() as conn:
            rows = execute(
                conn,
                "SELECT chunk_id FROM finrag_chunks WHERE knowledge_base_id = %s AND document_id = %s",
                (knowledge_base_id, document_id),
            ).fetchall()
        return [str(row[0]) for row in rows]

    # BaseDocumentStore interface

    def persist(self, persist_path: str = "", fs: Optional[Any] = None) -> None:
        """
        持久化数据库到指定路径
        Args:
            persist_path: 持久化路径
            fs: 文件系统对象，用于持久化到文件系统
        """
        return None

    def add_documents(
        self,
        docs: Iterable[BaseNode],
        allow_update: bool = True,
        batch_size: int = 2048,
        store_text: bool = True,
    ) -> None:
        """
        向数据库添加节点
        Args:
            docs: 节点迭代器，每个节点是一个 BaseNode 对象
            allow_update: 是否允许更新已存在的节点
            batch_size: 批量大小，一次插入的节点数量
            store_text: 是否存储节点文本内容
        """
        nodes = list(docs)
        if not nodes:
            return
        now = utc_now_iso()
        rows = []
        # 遍历所有节点，将每个节点转换为分块记录
        for node in nodes:
            metadata = dict(node.metadata or {})
            chunk_id = str(metadata.get("chunk_id") or node.node_id)
            document_id = self._ref_doc_id(node)
            rows.append(
                (
                    chunk_id,
                    document_id,
                    str(metadata.get("knowledge_base_id") or "finance"),
                    int(metadata.get("chunk_level", 0) or 0),
                    int(metadata.get("chunk_idx", 0) or 0),
                    node.model_dump_json(),
                    now,
                )
            )
        with self.db.connect() as conn:
            # 如果不允许更新已存在的节点，检查是否存在已存在的分块 ID
            if not allow_update:
                # 已存在的分块 ID 列表集合
                existing_ids = self._existing_chunk_ids(conn, [row[0] for row in rows])
                if existing_ids:
                    raise ValueError(f"Chunks already exist: {', '.join(sorted(existing_ids))}")
            # 批量插入或更新分块记录
            executemany(
                conn,
                """
                INSERT INTO finrag_chunks (
                    chunk_id, document_id, knowledge_base_id, chunk_level,
                    chunk_idx, payload, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    knowledge_base_id = EXCLUDED.knowledge_base_id,
                    chunk_level = EXCLUDED.chunk_level,
                    chunk_idx = EXCLUDED.chunk_idx,
                    payload = EXCLUDED.payload,
                    updated_at = EXCLUDED.updated_at
                """,
                rows,
            )
        # 批量插入或更新参考文档记录
        self._upsert_ref_doc_info(nodes)

    async def async_add_documents(
        self,
        docs: Iterable[BaseNode],
        allow_update: bool = True,
        batch_size: int = 2048,
        store_text: bool = True,
    ) -> None:
        """
        异步向数据库添加节点
        Args:
            docs: 节点迭代器，每个节点是一个 BaseNode 对象
            allow_update: 是否允许更新已存在的节点
            batch_size: 批量大小，一次插入的节点数量
            store_text: 是否存储节点文本内容
        """
        self.add_documents(docs, allow_update=allow_update, batch_size=batch_size, store_text=store_text)

    def get_document(self, doc_id: str, raise_error: bool = True) -> Optional[BaseNode]:
        """
        获取节点
        Args:
            doc_id: 节点 ID
            raise_error: 是否在节点不存在时抛出异常
        Returns:
            节点对象，如果存在则返回，否则返回 None
        """
        node = self.get_node(doc_id)
        if node is None and raise_error:
            raise ValueError(f"Document {doc_id} not found")
        return node

    async def aget_document(self, doc_id: str, raise_error: bool = True) -> Optional[BaseNode]:
        """
        异步获取节点
        Args:
            doc_id: 节点 ID
            raise_error: 是否在节点不存在时抛出异常
        Returns:
            节点对象，如果存在则返回，否则返回 None
        """
        return self.get_document(doc_id, raise_error=raise_error)

    def delete_document(self, doc_id: str, raise_error: bool = True) -> None:
        """
        删除节点
        Args:
            doc_id: 节点 ID
            raise_error: 是否在节点不存在时抛出异常
        """
        if not self.document_exists(doc_id):
            if raise_error:
                raise ValueError(f"Document {doc_id} not found")
            return
        with self.db.connect() as conn:
            # 删除分块记录和分块哈希记录
            execute(conn, "DELETE FROM finrag_chunks WHERE chunk_id = %s", (doc_id,))
            execute(conn, "DELETE FROM finrag_llama_doc_hashes WHERE chunk_id = %s", (doc_id,))
        #  获取节点
        node = self.get_document(doc_id, raise_error=False)
        #  获取参考文档 ID
        ref_doc_id = self._ref_doc_id(node) if node is not None else ""
        # 如果有参考文档 ID，更新参考文档记录
        if ref_doc_id:
            info = self.get_ref_doc_info(ref_doc_id)
            if info is not None:
                # 从参考文档记录中删除该节点 ID
                info.node_ids = [node_id for node_id in info.node_ids if node_id != doc_id]
                # 更新后的参考文档记录
                self._write_ref_doc_info(ref_doc_id, info)

    async def adelete_document(self, doc_id: str, raise_error: bool = True) -> None:
        """
        异步删除节点
        Args:
            doc_id: 节点 ID
            raise_error: 是否在节点不存在时抛出异常
        """
        self.delete_document(doc_id, raise_error=raise_error)

    def document_exists(self, doc_id: str) -> bool:
        """
        检查节点是否存在
        Args:
            doc_id: 节点 ID
        Returns:
            如果节点存在则返回 True，否则返回 False
        """
        return self.get_node(doc_id) is not None

    async def adocument_exists(self, doc_id: str) -> bool:
        """
        异步检查节点是否存在
        Args:
            doc_id: 节点 ID
        Returns:
            如果节点存在则返回 True，否则返回 False
        """
        return self.document_exists(doc_id)

    def set_document_hash(self, doc_id: str, doc_hash: str) -> None:
        """
        设置节点哈希值
        Args:
            doc_id: 节点 ID
            doc_hash: 节点哈希值
        """
        self.set_document_hashes({doc_id: doc_hash})

    async def aset_document_hash(self, doc_id: str, doc_hash: str) -> None:
        """
        异步设置节点哈希值
        Args:
            doc_id: 节点 ID
            doc_hash: 节点哈希值
        """
        self.set_document_hash(doc_id, doc_hash)

    def set_document_hashes(self, doc_hashes: Dict[str, str]) -> None:
        """
        设置节点哈希值
        Args:
            doc_hashes: 节点 ID 到哈希值的映射字典
        """
        if not doc_hashes:
            return
        rows = [(str(chunk_id), str(doc_hash)) for chunk_id, doc_hash in doc_hashes.items()]
        with self.db.connect() as conn:
            executemany(
                conn,
                """
                INSERT INTO finrag_llama_doc_hashes (chunk_id, doc_hash)
                VALUES (%s, %s)
                ON CONFLICT (chunk_id) DO UPDATE SET doc_hash = EXCLUDED.doc_hash
                """,
                rows,
            )

    async def aset_document_hashes(self, doc_hashes: Dict[str, str]) -> None:
        """
        异步设置节点哈希值
        Args:
            doc_hashes: 节点 ID 到哈希值的映射字典
        """
        self.set_document_hashes(doc_hashes)

    def get_document_hash(self, doc_id: str) -> Optional[str]:
        """
        获取节点哈希值
        Args:
            doc_id: 节点 ID
        Returns:
            节点哈希值，如果存在则返回，否则返回 None
        """
        with self.db.connect() as conn:
            row = execute(conn, "SELECT doc_hash FROM finrag_llama_doc_hashes WHERE chunk_id = %s", (doc_id,)).fetchone()
        return str(row[0]) if row else None

    async def aget_document_hash(self, doc_id: str) -> Optional[str]:
        """
        异步获取节点哈希值
        Args:
            doc_id: 节点 ID
        Returns:
            节点哈希值，如果存在则返回，否则返回 None
        """
        return self.get_document_hash(doc_id)

    def get_all_document_hashes(self) -> Dict[str, str]:
        """
        获取所有节点的哈希值
        Returns:
            节点 ID 到哈希值的映射字典
        """
        with self.db.connect() as conn:
            rows = execute(conn, "SELECT chunk_id, doc_hash FROM finrag_llama_doc_hashes ORDER BY chunk_id").fetchall()
        return {str(row[0]): str(row[1]) for row in rows}

    async def aget_all_document_hashes(self) -> Dict[str, str]:
        """
        异步获取所有节点的哈希值
        Returns:
            节点 ID 到哈希值的映射字典
        """
        return self.get_all_document_hashes()

    def get_all_ref_doc_info(self) -> Optional[Dict[str, RefDocInfo]]:
        """
        获取所有参考文档信息
        Returns:
            参考文档 ID 到参考文档信息对象的映射字典
        """
        with self.db.connect() as conn:
            rows = execute(conn, "SELECT ref_doc_id, node_ids_json, metadata_json FROM finrag_ref_docs ORDER BY ref_doc_id").fetchall()
        return {
            str(row[0]): RefDocInfo(node_ids=list(json.loads(row[1])), metadata=dict(json.loads(row[2])))
            for row in rows
        }

    async def aget_all_ref_doc_info(self) -> Optional[Dict[str, RefDocInfo]]:
        """
        异步获取所有参考文档信息
        Returns:
            参考文档 ID 到参考文档信息对象的映射字典
        """
        return self.get_all_ref_doc_info()

    def get_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        """
        获取参考文档信息
        Args:
            ref_doc_id: 参考文档 ID
        Returns:
            参考文档信息对象，包含该参考文档 ID 对应的节点 ID 列表和元数据字典
        """
        with self.db.connect() as conn:
            row = execute(
                conn,
                "SELECT node_ids_json, metadata_json FROM finrag_ref_docs WHERE ref_doc_id = %s",
                (ref_doc_id,),
            ).fetchone()
        if row is None:
            return None
        return RefDocInfo(node_ids=list(json.loads(row[0])), metadata=dict(json.loads(row[1])))

    async def aget_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        """
        异步获取参考文档信息
        Args:
            ref_doc_id: 参考文档 ID
        Returns:
            参考文档信息对象，包含该参考文档 ID 对应的节点 ID 列表和元数据字典
        """
        return self.get_ref_doc_info(ref_doc_id)

    def delete_ref_doc(self, ref_doc_id: str, raise_error: bool = True) -> None:
        """
        删除参考文档
        Args:
            ref_doc_id: 参考文档 ID
            raise_error: 如果参考文档不存在是否抛出 ValueError 异常
        """
        info = self.get_ref_doc_info(ref_doc_id)
        if info is None:
            if raise_error:
                raise ValueError(f"Ref doc {ref_doc_id} not found")
            return
        with self.db.connect() as conn:
            for node_id in info.node_ids:
                # 在 finrag_chunks 表中删除所有与参考文档 ID 关联的节点
                execute(conn, "DELETE FROM finrag_chunks WHERE chunk_id = %s", (str(node_id),))
                # 在 finrag_llama_doc_hashes 表中删除所有与参考文档 ID 关联的节点
                execute(conn, "DELETE FROM finrag_llama_doc_hashes WHERE chunk_id = %s", (str(node_id),))
            # 在 finrag_ref_docs 表中删除参考文档
            execute(conn, "DELETE FROM finrag_ref_docs WHERE ref_doc_id = %s", (ref_doc_id,))

    async def adelete_ref_doc(self, ref_doc_id: str, raise_error: bool = True) -> None:
        """
        异步删除参考文档
        Args:
            ref_doc_id: 参考文档 ID
            raise_error: 如果参考文档不存在是否抛出 ValueError 异常
        Returns:
            None
        """
        self.delete_ref_doc(ref_doc_id, raise_error=raise_error)

    @staticmethod
    def _ref_doc_id(node: Optional[BaseNode]) -> str:
        """
        从节点元数据中提取参考文档 ID
        Args:
            node: 节点对象
        Returns:
            参考文档 ID
        """
        if node is None:
            return ""
        metadata = dict(node.metadata or {})
        return str(getattr(node, "ref_doc_id", None) or metadata.get("document_id") or metadata.get("ref_doc_id") or node.node_id)

    @staticmethod
    def _ref_doc_metadata(nodes: List[BaseNode], ref_doc_id: str) -> Dict[str, Any]:
        """
        从节点列表中提取参考文档元数据
        Args:
            nodes: 节点列表
            ref_doc_id: 参考文档 ID
        Returns:
            参考文档元数据字典
        """
        for node in nodes:
            # 如果节点的参考文档 ID 与目标参考文档 ID 匹配，则返回节点的元数据字典
            if PostgreSQLLlamaIndexDocumentStore._ref_doc_id(node) == ref_doc_id:
                metadata = dict(node.metadata or {})
                metadata.setdefault("document_id", ref_doc_id)
                return metadata
        return {"document_id": ref_doc_id}

    def _upsert_ref_doc_info(self, nodes: List[BaseNode]) -> None:
        """
        批量插入或更新参考文档信息
        Args:
            nodes: 要插入或更新的节点列表
        """
        # 键为参考文档 ID，值为该参考文档 ID 对应的节点列表的字典
        by_ref_doc: Dict[str, List[BaseNode]] = {}
        for node in nodes:
            by_ref_doc.setdefault(self._ref_doc_id(node), []).append(node)
        for ref_doc_id, ref_nodes in by_ref_doc.items():
            # 获取已存在的参考文档信息
            existing = self.get_ref_doc_info(ref_doc_id)
            # 该参考文档对应的已存在的节点 ID 列表
            existing_ids = list(existing.node_ids) if existing is not None else []
            for node in ref_nodes:
                # 将不存在于已存在的节点 ID 列表中的节点 ID 添加到已存在的节点 ID 列表中
                if node.node_id not in existing_ids:
                    existing_ids.append(node.node_id)
            # 从节点列表中提取参考文档元数据
            metadata = self._ref_doc_metadata(ref_nodes, ref_doc_id)
            # 插入或更新参考文档信息
            self._write_ref_doc_info(ref_doc_id, RefDocInfo(node_ids=existing_ids, metadata=metadata))

    def _write_ref_doc_info(self, ref_doc_id: str, info: RefDocInfo) -> None:
        """
        插入或更新参考文档信息
        Args:
            ref_doc_id: 参考文档 ID
            info: 参考文档信息对象，包含该参考文档 ID 对应的节点 ID 列表和元数据字典
        """
        if not info.node_ids:
            # 如果参考文档对应的节点 ID 列表为空，则在 finrag_ref_docs 表中删除该记录
            with self.db.connect() as conn:
                execute(conn, "DELETE FROM finrag_ref_docs WHERE ref_doc_id = %s", (ref_doc_id,))
            return
        with self.db.connect() as conn:
            # 插入或更新参考文档信息
            # 如果参考文档 ID 已存在，则更新节点 ID 列表和元数据
            # 如果不存在，则插入新记录
            execute(
                conn,
                """
                INSERT INTO finrag_ref_docs (ref_doc_id, node_ids_json, metadata_json)
                VALUES (%s, %s, %s)
                ON CONFLICT (ref_doc_id) DO UPDATE SET
                    node_ids_json = EXCLUDED.node_ids_json,
                    metadata_json = EXCLUDED.metadata_json
                """,
                (
                    ref_doc_id,
                    json.dumps(list(info.node_ids), ensure_ascii=False),
                    json.dumps(dict(info.metadata), ensure_ascii=False, sort_keys=True),
                ),
            )

    @staticmethod
    def _existing_chunk_ids(conn: Any, chunk_ids: List[str]) -> set[str]:
        """
        在 finrag_chunks 表中获取指定分块 ID 列表中已存在的分块 ID 列表集合
        Args:
            conn: 数据库连接对象
            chunk_ids: 分块 ID 列表
        Returns:
            已存在的分块 ID 列表
        """
        if not chunk_ids:
            return set()
        rows = execute(
            conn,
            f"SELECT chunk_id FROM finrag_chunks WHERE chunk_id IN ({','.join(['%s'] * len(chunk_ids))})",
            tuple(chunk_ids),
        ).fetchall()
        return {str(row[0]) for row in rows}
