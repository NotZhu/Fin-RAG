"""文档注册表类"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._common import _document_record_cls, utc_now_iso
from .db import Database, execute, executemany


class PostgreSQLDocumentRegistry:
    """基于 PostgreSQL 的文档注册表"""

    def __init__(self, database_url: str):
        """
        初始化 PostgreSQL 文档注册表
        Args:
            database_url: PostgreSQL 数据库连接串
        """
        # 初始化数据库连接
        self.db = Database(database_url)
        # 文档记录缓存，以 document_id 为键，DocumentRecord 对象为值
        self.records: Dict[str, Any] = {}
        # 确保数据库表结构存在
        self._ensure_schema()
        # 从数据库加载现有记录到内存缓存
        self._load()

    def _ensure_schema(self) -> None:
        """
        确保文档注册表所需表和索引存在
        """
        with self.db.connect() as conn: # 连接数据库并获取连接对象
            # 创建文档注册表
            execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS finrag_documents (
                    document_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    knowledge_base_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    upload_time TEXT NOT NULL,
                    last_error TEXT
                )
                """,
            )
            # 创建索引
            execute(conn, "CREATE INDEX IF NOT EXISTS idx_finrag_documents_hash ON finrag_documents(content_hash, knowledge_base_id)")

    def _load(self) -> None:
        """
        从数据库加载文档注册记录到内存缓存
        """
        with self.db.connect() as conn: # 连接数据库并获取连接对象
            # 查询所有文档注册记录
            rows = execute(
                conn,
                """
                SELECT document_id, source_path, filename, file_type, content_hash,
                       knowledge_base_id, status, chunk_count, upload_time, last_error
                FROM finrag_documents
                ORDER BY upload_time, document_id
                """,
            ).fetchall()
        document_record_cls = _document_record_cls()
        # 从查询结果构建文档记录缓存
        self.records = {
            row[0]: document_record_cls(
                document_id=row[0],
                source_path=row[1],
                filename=row[2],
                file_type=row[3],
                content_hash=row[4],
                knowledge_base_id=row[5],
                status=row[6],
                chunk_count=int(row[7] or 0),
                upload_time=row[8] or "",
                last_error=row[9],
            )
            for row in rows
        }

    def save(self) -> None:
        """
        将内存中的文档注册记录全量保存到数据库
        """
        rows = [self._record_row(record) for record in self.records.values()]
        with self.db.connect() as conn: # 连接数据库并获取连接对象
            # 清空数据库中的文档注册表，然后批量插入当前内存缓存中的记录
            execute(conn, "DELETE FROM finrag_documents")
            executemany(
                conn,
                """
                INSERT INTO finrag_documents (
                    document_id, source_path, filename, file_type, content_hash,
                    knowledge_base_id, status, chunk_count, upload_time, last_error
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )

    @staticmethod
    def _record_row(record: Any) -> tuple[Any, ...]:
        """
        将文档注册记录转换为数据库写入参数
        Args:
            record: DocumentRecord 注册记录
        Returns:
            可用于 INSERT/UPSERT 的参数元组
        """
        return (
            record.document_id,
            record.source_path,
            record.filename,
            record.file_type,
            record.content_hash,
            record.knowledge_base_id,
            record.status,
            int(record.chunk_count or 0),
            record.upload_time,
            record.last_error,
        )

    def _save_record(self, record: Any) -> None:
        """
        保存单条文档注册记录，不影响数据库中的其他文档
        Args:
            record: DocumentRecord 注册记录
        """
        with self.db.connect() as conn: # 连接数据库并获取连接对象
            # 插入或更新记录
            execute(
                conn,
                """
                INSERT INTO finrag_documents (
                    document_id, source_path, filename, file_type, content_hash,
                    knowledge_base_id, status, chunk_count, upload_time, last_error
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (document_id) DO UPDATE SET
                    source_path = EXCLUDED.source_path,
                    filename = EXCLUDED.filename,
                    file_type = EXCLUDED.file_type,
                    content_hash = EXCLUDED.content_hash,
                    knowledge_base_id = EXCLUDED.knowledge_base_id,
                    status = EXCLUDED.status,
                    chunk_count = EXCLUDED.chunk_count,
                    upload_time = EXCLUDED.upload_time,
                    last_error = EXCLUDED.last_error
                """,
                self._record_row(record),
            )

    def list(self) -> List[dict]:
        """
        列出包含内部字段的全部文档记录
        Returns:
            按上传时间排序的文档记录字典列表
        """
        return [record.to_dict() for record in sorted(self.records.values(), key=lambda item: item.upload_time)]

    def list_public(self, knowledge_base_id: str | None = None) -> List[dict]:
        """
        列出对外展示的非删除文档记录
        Args:
            knowledge_base_id: 可选知识库 ID，传入时只返回该知识库文档
        Returns:
            隐藏 source_path 和 content_hash 等内部字段后的文档列表
        """
        # 公开字段，不包含内部字段
        public_fields = {
            "document_id", # 文档 ID
            "filename", # 原始文件名
            "file_type", # 文件类型
            "knowledge_base_id", # 资料库 ID
            "status", # 文档状态
            "chunk_count", # 分块计数
            "upload_time", # 上传时间
            "last_error", # 最后错误信息
        }
        return [
            # 只保留公开字段
            {key: value for key, value in record.to_dict().items() if key in public_fields}
            # 按 upload_time 升序，并过滤掉 status 为 deleted 的记录，仅返回指定知识库文档
            for record in sorted(self.records.values(), key=lambda item: item.upload_time)
            if record.status != "deleted"
            and (knowledge_base_id is None or record.knowledge_base_id == knowledge_base_id)
        ]

    def get(self, document_id: str) -> Any:
        """
        按文档 ID 获取注册记录
        Args:
            document_id: 文档 ID
        Returns:
            DocumentRecord 注册记录
        """
        return self.records[document_id]

    def find_by_hash(self, content_hash: str, knowledge_base_id: str) -> Optional[Any]:
        """
        按内容哈希和资料库 ID 查找未删除文档
        Args:
            content_hash: 文件内容哈希
            knowledge_base_id: 资料库 ID
        Returns:
            命中的 DocumentRecord，未命中时返回 None
        """
        for record in self.records.values():
            if record.content_hash == content_hash and record.knowledge_base_id == knowledge_base_id and record.status != "deleted":
                return record
        return None

    def upsert_uploaded(
        self,
        *,
        source_path: Path,
        filename: str,
        file_type: str,
        content_hash: str,
        knowledge_base_id: str,
    ) -> Any:
        """
        插入上传文档记录，或复用相同哈希的既有记录
        Args:
            source_path: 上传临时文件路径
            filename: 原始文件名
            file_type: 文件类型
            content_hash: 文件内容哈希
            knowledge_base_id: 资料库 ID
        Returns:
            新增或复用的 DocumentRecord
        """
        # 查找是否有相同哈希的既有记录
        existing = self.find_by_hash(content_hash, knowledge_base_id)
        # 如果有相同哈希的既有记录，直接返回
        if existing is not None:
            return existing
        # 创建新记录
        record = _document_record_cls()(
            document_id=uuid.uuid4().hex,
            source_path=str(source_path),
            filename=filename,
            file_type=file_type,
            content_hash=content_hash,
            knowledge_base_id=knowledge_base_id,
            status="uploaded",
            upload_time=utc_now_iso(),
        )
        self.records[record.document_id] = record
        # 保存新记录到数据库
        self._save_record(record)
        return record

    def update_source_path(self, document_id: str, source_path: str) -> None:
        """
        更新单个文档的源文件路径
        Args:
            document_id: 文档 ID
            source_path: 新的源文件路径
        """
        self.records[document_id].source_path = source_path
        self._save_record(self.records[document_id])

    def mark_parsing(self, document_id: str) -> None:
        """
        将文档状态标记为解析中
        Args:
            document_id: 文档 ID
        """
        self.records[document_id].status = "parsing"
        self.records[document_id].last_error = None
        # 保存更新后的记录同步到数据库
        self._save_record(self.records[document_id])

    def mark_indexed(self, document_id: str, chunk_count: int) -> None:
        """
        将文档状态标记为已索引
        Args:
            document_id: 文档 ID
            chunk_count: 该文档生成的叶子分块数量
        """
        self.records[document_id].status = "indexed"
        self.records[document_id].chunk_count = int(chunk_count)
        self.records[document_id].last_error = None
        self._save_record(self.records[document_id])

    def mark_failed(self, document_id: str, error: str) -> None:
        """
        将文档状态标记为失败并保存错误信息
        Args:
            document_id: 文档 ID
            error: 错误信息
        """
        self.records[document_id].status = "failed"
        self.records[document_id].last_error = error
        self._save_record(self.records[document_id])

    def mark_deleted(self, document_id: str) -> dict:
        """
        将文档状态标记为已删除
        Args:
            document_id: 文档 ID
        Returns:
            更新后的完整文档记录字典
        """
        self.records[document_id].status = "deleted"
        self._save_record(self.records[document_id])
        return self.records[document_id].to_dict()
