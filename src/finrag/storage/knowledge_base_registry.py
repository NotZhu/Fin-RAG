"""知识库注册表"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from finrag.core.config import validate_knowledge_base_id

from ._common import utc_now_iso
from .db import Database, execute


class DuplicateKnowledgeBaseError(ValueError):
    """知识库 ID 已注册"""


class KnowledgeBaseNotFoundError(KeyError):
    """知识库不存在"""


class KnowledgeBaseArchivedError(ValueError):
    """知识库已归档，不允许执行业务写入或问答操作"""


class ProtectedKnowledgeBaseError(ValueError):
    """受保护知识库不允许执行危险操作"""


@dataclass(frozen=True)
class KnowledgeBaseRecord:
    """知识库元数据"""

    knowledge_base_id: str
    created_at: str
    updated_at: str
    status: str = "active"
    archived_at: Optional[str] = None
    deleted_at: Optional[str] = None

    def to_dict(self, *, document_count: int = 0) -> Dict[str, Any]:
        return {
            "knowledge_base_id": self.knowledge_base_id,
            "document_count": int(document_count or 0),
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "archived_at": self.archived_at,
            "deleted_at": self.deleted_at,
        }


class PostgreSQLKnowledgeBaseRegistry:
    """知识库注册表"""

    def __init__(self, database_url: str):
        self.db = Database(database_url)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self.db.connect() as conn:
            execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS finrag_knowledge_bases (
                    knowledge_base_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    archived_at TEXT,
                    deleted_at TEXT
                )
                """,
            )

    def ensure_default(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        """确保知识库存在"""
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        # 检查知识库是否存在
        existing = self.get_optional(knowledge_base_id, include_deleted=True)
        if existing is not None:
            # 如果知识库已删除，恢复它
            if existing.status == "deleted":
                now = utc_now_iso()
                with self.db.connect() as conn:
                    # 恢复知识库
                    execute(
                        conn,
                        """
                        UPDATE finrag_knowledge_bases
                        SET status = 'active', archived_at = NULL, deleted_at = NULL, updated_at = %s
                        WHERE knowledge_base_id = %s
                        """,
                        (now, knowledge_base_id),
                    )
                return self.get(knowledge_base_id)
            return existing
        now = utc_now_iso()
        with self.db.connect() as conn:
            # 插入知识库元数据
            execute(
                conn,
                """
                INSERT INTO finrag_knowledge_bases (knowledge_base_id, created_at, updated_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (knowledge_base_id) DO NOTHING
                """,
                (knowledge_base_id, now, now),
            )
        return self.get(knowledge_base_id)

    def create(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        """创建知识库"""
        resolved_id = validate_knowledge_base_id(knowledge_base_id)
        # 如果知识库已存在，抛出异常
        if self.get_optional(resolved_id, include_deleted=True) is not None:
            raise DuplicateKnowledgeBaseError(resolved_id)
        now = utc_now_iso()
        with self.db.connect() as conn:
            execute(
                conn,
                """
                INSERT INTO finrag_knowledge_bases (knowledge_base_id, created_at, updated_at)
                VALUES (%s, %s, %s)
                """,
                (resolved_id, now, now),
            )
        return self.get(resolved_id)

    def list(self, *, include_deleted: bool = False) -> List[KnowledgeBaseRecord]:
        """列出所有知识库"""
        where_clause = "" if include_deleted else "WHERE status != 'deleted'"
        with self.db.connect() as conn:
            rows = execute(
                conn,
                f"""
                SELECT knowledge_base_id, created_at, updated_at
                     , status, archived_at, deleted_at
                FROM finrag_knowledge_bases
                {where_clause}
                ORDER BY created_at, knowledge_base_id
                """,
            ).fetchall()
        return [self._record(row) for row in rows]

    def get(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        """获取知识库元数据"""
        record = self.get_optional(knowledge_base_id)
        if record is None:
            raise KnowledgeBaseNotFoundError(knowledge_base_id)
        return record

    def get_optional(self, knowledge_base_id: str, *, include_deleted: bool = False) -> Optional[KnowledgeBaseRecord]:
        """获取知识库元数据"""
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        deleted_filter = "" if include_deleted else "AND status != 'deleted'"
        with self.db.connect() as conn:
            # 查询知识库元数据
            row = execute(
                conn,
                f"""
                SELECT knowledge_base_id, created_at, updated_at
                     , status, archived_at, deleted_at
                FROM finrag_knowledge_bases
                WHERE knowledge_base_id = %s
                {deleted_filter}
                """,
                (knowledge_base_id,),
            ).fetchone()
        # 转换为知识库元数据
        return self._record(row) if row is not None else None

    def archive(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        """归档知识库，保留数据和索引"""
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        self.get(knowledge_base_id)
        now = utc_now_iso()
        with self.db.connect() as conn:
            # 归档知识库
            execute(
                conn,
                """
                UPDATE finrag_knowledge_bases
                SET status = 'archived', archived_at = %s, deleted_at = NULL, updated_at = %s
                WHERE knowledge_base_id = %s
                """,
                (now, now, knowledge_base_id),
            )
        return self.get(knowledge_base_id)

    def restore(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        """恢复归档知识库"""
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        record = self.get_optional(knowledge_base_id, include_deleted=True)
        if record is None or record.status == "deleted":
            raise KnowledgeBaseNotFoundError(knowledge_base_id)
        now = utc_now_iso()
        with self.db.connect() as conn:
            # 恢复知识库
            execute(
                conn,
                """
                UPDATE finrag_knowledge_bases
                SET status = 'active', archived_at = NULL, deleted_at = NULL, updated_at = %s
                WHERE knowledge_base_id = %s
                """,
                (now, knowledge_base_id),
            )
        return self.get(knowledge_base_id)

    def mark_deleted(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        """将知识库标记为已删除"""
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        record = self.get_optional(knowledge_base_id, include_deleted=True)
        if record is None:
            raise KnowledgeBaseNotFoundError(knowledge_base_id)
        now = utc_now_iso()
        with self.db.connect() as conn:
            # 标记知识库为已删除
            execute(
                conn,
                """
                UPDATE finrag_knowledge_bases
                SET status = 'deleted', deleted_at = %s, updated_at = %s
                WHERE knowledge_base_id = %s
                """,
                (now, now, knowledge_base_id),
            )
        deleted = self.get_optional(knowledge_base_id, include_deleted=True)
        if deleted is None:
            raise KnowledgeBaseNotFoundError(knowledge_base_id)
        return deleted

    @staticmethod
    def _record(row: Any) -> KnowledgeBaseRecord:
        """将数据库行转换为知识库元数据"""
        return KnowledgeBaseRecord(
            knowledge_base_id=str(row[0]),
            created_at=str(row[1] or ""),
            updated_at=str(row[2] or ""),
            status=str(row[3] or "active"),
            archived_at=str(row[4]) if row[4] is not None else None,
            deleted_at=str(row[5]) if row[5] is not None else None,
        )
