"""索引清单存储类"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ._common import utc_now_iso
from .db import Database, execute


class PostgreSQLIndexManifestStore:
    """将当前索引清单持久化到 PostgreSQL"""

    def __init__(self, database_url: str):
        """
        初始化索引清单存储
        Args:
            database_url: PostgreSQL 数据库连接串
        """
        self.db = Database(database_url)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """
        确保索引清单表存在
        """
        with self.db.connect() as conn: # 连接数据库并获取连接对象
            # 创建索引清单表，包含 ID、JSON 载荷和更新时间字段
            execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS finrag_index_manifest (
                    knowledge_base_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )

    def save_manifest(self, manifest: Dict[str, Any], knowledge_base_id: str) -> None:
        """
        保存当前索引清单
        Args:
            manifest: 待保存的索引清单字典
            knowledge_base_id: 知识库 ID
        """
        payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
        with self.db.connect() as conn:
            execute(
                conn,
                """
                INSERT INTO finrag_index_manifest (knowledge_base_id, payload, updated_at)
                VALUES (%s, %s, %s)
                ON CONFLICT (knowledge_base_id) DO UPDATE SET
                    payload = EXCLUDED.payload,
                    updated_at = EXCLUDED.updated_at
                """,
                (knowledge_base_id, payload, utc_now_iso()),
            )

    def load_manifest(self, knowledge_base_id: str) -> Optional[Dict[str, Any]]:
        """
        加载当前索引清单
        Args:
            knowledge_base_id: 知识库 ID
        Returns:
            索引清单字典，不存在时返回 None
        """
        with self.db.connect() as conn: # 连接数据库并获取连接对象
            # 查询指定知识库的索引清单记录
            row = execute(
                conn,
                "SELECT payload FROM finrag_index_manifest WHERE knowledge_base_id = %s",
                (knowledge_base_id,),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0]) # 解析 JSON 字符串为字典

    def delete_manifest(self, knowledge_base_id: str) -> None:
        """
        删除指定知识库的索引清单
        Args:
            knowledge_base_id: 知识库 ID
        """
        with self.db.connect() as conn:
            # 删除指定知识库的索引清单记录
            execute(conn, "DELETE FROM finrag_index_manifest WHERE knowledge_base_id = %s", (knowledge_base_id,))
