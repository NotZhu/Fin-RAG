"""PostgreSQL-backed BM25 sparse-vector state store."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

from .db import Database, execute
from .protocols import SparseVector


BM25_K1 = 1.5 # BM25 词频饱和参数
BM25_B = 0.75 # BM25 文档长度归一化参数


class PostgreSQLBM25StateStore:
    """用于 Milvus 稀疏向量的 PostgreSQL BM25 词项状态存储"""

    def __init__(self, database_url: str):
        """
        初始化 BM25 状态存储
        Args:
            database_url: PostgreSQL 数据库连接串
        """
        self.db = Database(database_url)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """
        确保 BM25 词项、分块和词频表存在
        """
        with self.db.connect() as conn: # 连接数据库并获取连接对象
            # 创建 BM25 词项表，包含 词项 ID、词项文本和文档频率
            execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS bm25_terms (
                    term_id SERIAL PRIMARY KEY,
                    knowledge_base_id TEXT NOT NULL,
                    term TEXT NOT NULL,
                    df INTEGER NOT NULL DEFAULT 0,
                    UNIQUE (knowledge_base_id, term)
                )
                """,
            )
            # 创建 BM25 分块表，包含 分块 ID、文档 ID和 token 数量
            execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS bm25_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    knowledge_base_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    token_count INTEGER NOT NULL
                )
                """,
            )
            # 创建 BM25 词频表，包含 分块 ID、词项 ID和词频
            execute(
                conn,
                """
                CREATE TABLE IF NOT EXISTS bm25_term_counts (
                    knowledge_base_id TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    term_id INTEGER NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY (knowledge_base_id, chunk_id, term_id)
                )
                """,
            )
            # 创建 BM25 词项表索引，加速词项文本查询
            execute(conn, "CREATE UNIQUE INDEX IF NOT EXISTS idx_bm25_terms_kb_term ON bm25_terms(knowledge_base_id, term)")
            # 创建 BM25 分块表索引，加速文档 ID 查询
            execute(conn, "CREATE INDEX IF NOT EXISTS idx_bm25_chunks_kb_document ON bm25_chunks(knowledge_base_id, document_id)")
            execute(conn, "CREATE INDEX IF NOT EXISTS idx_bm25_term_counts_kb_chunk ON bm25_term_counts(knowledge_base_id, chunk_id)")

    def replace_document_chunks(self, knowledge_base_id: str, document_id: str, chunk_token_counts: Dict[str, Dict[str, int]]) -> None:
        """
        替换单个文档各分块的 BM25 词频状态
        Args:
            knowledge_base_id: 知识库 ID
            document_id: 文档 ID
            chunk_token_counts: chunk_id 到 token 词频映射的字典
        """
        # 先删除该文档的所有分块词频记录
        self.delete_document(knowledge_base_id, document_id)
        # 初始化受影响的词项 ID 集合，用于刷新词项文档频率
        affected_term_ids = set()
        with self.db.connect() as conn: # 连接数据库并获取连接对象
            # 遍历每个分块的词频记录
            for chunk_id, token_counts in chunk_token_counts.items():
                # 过滤掉无效的 token项（空字符串或非整数）
                clean_counts = {str(term): int(count) for term, count in token_counts.items() if term and int(count) > 0}
                # 插入该分块的 token 数量 到 BM25 分块表
                execute(
                    conn,
                    """
                    INSERT INTO bm25_chunks (chunk_id, knowledge_base_id, document_id, token_count)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (chunk_id, knowledge_base_id, document_id, sum(clean_counts.values())),
                )
                # 插入该分块的所有词项词频记录
                for term, count in clean_counts.items():
                    term_id = self._ensure_term(conn, knowledge_base_id, term)
                    affected_term_ids.add(term_id)
                    # 插入该分块的词项词频记录 到 BM25 词频表
                    execute(
                        conn,
                        """
                        INSERT INTO bm25_term_counts (knowledge_base_id, chunk_id, term_id, count)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (knowledge_base_id, chunk_id, term_id, count),
                    )
            # 刷新词项文档频率
            self._refresh_term_dfs(conn, knowledge_base_id, affected_term_ids)

    def delete_document(self, knowledge_base_id: str, document_id: str) -> None:
        """
        删除指定文档的 BM25 状态并刷新词项文档频率
        Args:
            knowledge_base_id: 知识库 ID
            document_id: 文档 ID
        """
        with self.db.connect() as conn: # 连接数据库并获取连接对象
            # 在 BM25 词频表中查询指定文档的所有词项 ID 列表
            rows = execute(
                conn,
                """
                SELECT DISTINCT btc.term_id
                FROM bm25_term_counts btc
                JOIN bm25_chunks bc ON bc.knowledge_base_id = btc.knowledge_base_id
                    AND bc.chunk_id = btc.chunk_id
                WHERE bc.knowledge_base_id = %s
                AND bc.document_id = %s
                """,
                (knowledge_base_id, document_id),
            ).fetchall()
            # 提取所有词项 ID，用于刷新词项文档频率
            affected_term_ids = {int(row[0]) for row in rows}
            # 在 BM25 词频表中删除指定文档的所有词项词频记录
            execute(
                conn,
                """
                DELETE FROM bm25_term_counts
                WHERE knowledge_base_id = %s
                AND chunk_id IN (
                    SELECT chunk_id FROM bm25_chunks
                    WHERE knowledge_base_id = %s AND document_id = %s
                )
                """,
                (knowledge_base_id, knowledge_base_id, document_id),
            )
            # 删除指定文档的所有分块记录
            execute(
                conn,
                "DELETE FROM bm25_chunks WHERE knowledge_base_id = %s AND document_id = %s",
                (knowledge_base_id, document_id)
            )
            # 刷新词项文档频率
            self._refresh_term_dfs(conn, knowledge_base_id, affected_term_ids)

    def clear(self, knowledge_base_id: str) -> None:
        """
        清空全部 BM25 分块词频并重置文档频率
        """
        with self.db.connect() as conn: # 连接数据库并获取连接对象
            # 清空指定知识库的分块和词频表，并将该知识库词项表中的文档频率重置为 0
            execute(conn, "DELETE FROM bm25_term_counts WHERE knowledge_base_id = %s", (knowledge_base_id,))
            execute(conn, "DELETE FROM bm25_chunks WHERE knowledge_base_id = %s", (knowledge_base_id,))
            execute(conn, "UPDATE bm25_terms SET df = 0 WHERE knowledge_base_id = %s", (knowledge_base_id,))

    def build_query_sparse_vector(self, knowledge_base_id: str, tokens: Iterable[str]) -> SparseVector:
        """
        根据查询 token 构造稀疏向量，query 侧使用二值权重
        Args:
            knowledge_base_id: 知识库 ID
            tokens: 已分词的查询 token 序列
        Returns:
            SparseVector 稀疏向量载荷
        """
        token_counts = self._token_counts(tokens)
        if not token_counts:
            return SparseVector(indices=[], values=[], token_count=0)
        with self.db.connect() as conn:
            rows = execute(
                conn,
                f"""
                SELECT term_id, term FROM bm25_terms
                WHERE knowledge_base_id = %s
                AND term IN ({','.join(['%s'] * len(token_counts))})
                """,
                (knowledge_base_id, *tuple(token_counts.keys())),
            ).fetchall()
        by_term = {str(row[1]): int(row[0]) for row in rows}
        indices = [by_term[term] for term in token_counts if term in by_term]
        values = [1.0 for term in token_counts if term in by_term]
        return SparseVector(indices=indices, values=values, token_count=sum(token_counts.values()))

    def build_document_sparse_vector(self, knowledge_base_id: str, tokens: Iterable[str]) -> SparseVector:
        """
        根据 document token 构造标准 BM25 稀疏向量
        Args:
            knowledge_base_id: 知识库 ID
            tokens: 已分词的文档 token 序列
        Returns:
            SparseVector 稀疏向量载荷
        """
        token_counts = self._token_counts(tokens)
        if not token_counts:
            return SparseVector(indices=[], values=[], token_count=0)
        document_length = sum(token_counts.values())
        with self.db.connect() as conn:
            total_chunks, avgdl = self._collection_stats(conn, knowledge_base_id)
            rows = execute(
                conn,
                f"""
                SELECT term_id, term, df FROM bm25_terms
                WHERE knowledge_base_id = %s
                AND term IN ({','.join(['%s'] * len(token_counts))})
                """,
                (knowledge_base_id, *tuple(token_counts.keys())),
            ).fetchall()
        by_term = {str(row[1]): (int(row[0]), int(row[2] or 0)) for row in rows}
        avgdl = avgdl or float(document_length or 1)
        indices: List[int] = []
        values: List[float] = []
        for term, term_frequency in token_counts.items():
            if term not in by_term:
                continue
            term_id, document_frequency = by_term[term]
            idf = math.log(1.0 + ((total_chunks - document_frequency + 0.5) / (document_frequency + 0.5))) if total_chunks else 0.0
            denominator = term_frequency + BM25_K1 * (1.0 - BM25_B + BM25_B * (document_length / avgdl))
            weight = idf * ((term_frequency * (BM25_K1 + 1.0)) / denominator) if denominator else 0.0
            if weight:
                indices.append(term_id)
                values.append(float(weight))
        return SparseVector(indices=indices, values=values, token_count=document_length)

    @staticmethod
    def _token_counts(tokens: Iterable[str]) -> Dict[str, int]:
        """
        统计 token 词频并清理空白 token
        Args:
            tokens: 原始 token 序列
        Returns:
            小写 token 到出现次数的映射
        """
        token_counts: Dict[str, int] = {}
        for token in tokens:
            token = str(token).strip().lower()
            if token:
                token_counts[token] = token_counts.get(token, 0) + 1
        return token_counts

    @staticmethod
    def _collection_stats(conn: Any, knowledge_base_id: str) -> tuple[int, float]:
        """
        读取 BM25 分块集合统计信息
        Args:
            conn: 当前数据库连接
        Returns:
            分块数量和平均 token 数
        """
        row = execute(
            conn,
            "SELECT COUNT(*), COALESCE(AVG(token_count), 0) FROM bm25_chunks WHERE knowledge_base_id = %s",
            (knowledge_base_id,),
        ).fetchone()
        return int(row[0] or 0), float(row[1] or 0.0)

    def _ensure_term(self, conn: Any, knowledge_base_id: str, term: str) -> int:
        """
        确保词项存在并返回 term_id
        Args:
            conn: 当前数据库连接
            term: 词项文本
        Returns:
            词项 ID
        """
        # 检查该词项是否已存在
        row = execute(
            conn,
            "SELECT term_id FROM bm25_terms WHERE knowledge_base_id = %s AND term = %s",
            (knowledge_base_id, term),
        ).fetchone()
        # 如果该词项已存在，直接返回其 ID
        if row is not None:
            return int(row[0])
        # 如果该词项不存在，插入新记录
        execute(conn, "INSERT INTO bm25_terms (knowledge_base_id, term, df) VALUES (%s, %s, 0)", (knowledge_base_id, term))
        # 返回新插入的词项 ID
        row = execute(
            conn,
            "SELECT term_id FROM bm25_terms WHERE knowledge_base_id = %s AND term = %s",
            (knowledge_base_id, term),
        ).fetchone()
        return int(row[0])

    def _refresh_term_dfs(self, conn: Any, knowledge_base_id: str, term_ids: Iterable[int]) -> None:
        """
        刷新指定词项的文档频率
        Args:
            conn: 当前数据库连接
            term_ids: 待刷新词项 ID 序列
        """
        for term_id in set(int(term_id) for term_id in term_ids):
            # 统计该词项在当前知识库所有分块中的出现次数
            row = execute(
                conn,
                """
                SELECT COUNT(DISTINCT chunk_id)
                FROM bm25_term_counts
                WHERE knowledge_base_id = %s AND term_id = %s
                """,
                (knowledge_base_id, term_id),
            ).fetchone()
            # 更新该词项的文档频率
            execute(
                conn,
                "UPDATE bm25_terms SET df = %s WHERE knowledge_base_id = %s AND term_id = %s",
                (int(row[0] or 0), knowledge_base_id, term_id),
            )
