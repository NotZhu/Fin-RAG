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


@dataclass(frozen=True)
class KnowledgeBaseRecord:
    """知识库元数据"""

    knowledge_base_id: str
    created_at: str
    updated_at: str

    def to_dict(self, *, document_count: int = 0) -> Dict[str, Any]:
        return {
            "knowledge_base_id": self.knowledge_base_id,
            "document_count": int(document_count or 0),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
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
                    updated_at TEXT NOT NULL
                )
                """,
            )

    def ensure_default(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        """确保知识库存在"""
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        # 检查知识库是否存在
        existing = self.get_optional(knowledge_base_id)
        if existing is not None:
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
        if self.get_optional(resolved_id) is not None:
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

    def list(self) -> List[KnowledgeBaseRecord]:
        """列出所有知识库"""
        with self.db.connect() as conn:
            rows = execute(
                conn,
                """
                SELECT knowledge_base_id, created_at, updated_at
                FROM finrag_knowledge_bases
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

    def get_optional(self, knowledge_base_id: str) -> Optional[KnowledgeBaseRecord]:
        """获取知识库元数据"""
        knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        with self.db.connect() as conn:
            # 查询知识库元数据
            row = execute(
                conn,
                """
                SELECT knowledge_base_id, created_at, updated_at
                FROM finrag_knowledge_bases
                WHERE knowledge_base_id = %s
                """,
                (knowledge_base_id,),
            ).fetchone()
        # 转换为知识库元数据
        return self._record(row) if row is not None else None

    @staticmethod
    def _record(row: Any) -> KnowledgeBaseRecord:
        """将数据库行转换为知识库元数据"""
        return KnowledgeBaseRecord(
            knowledge_base_id=str(row[0]),
            created_at=str(row[1] or ""),
            updated_at=str(row[2] or ""),
        )
