from __future__ import annotations

import os

import pytest

TABLES = (
    "bm25_term_counts",
    "bm25_chunks",
    "bm25_terms",
    "finrag_index_manifest",
    "finrag_llama_doc_hashes",
    "finrag_ref_docs",
    "finrag_chunks",
    "finrag_documents",
)


def _database_url() -> str:
    database_url = os.getenv("FINRAG_TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("需要设置 FINRAG_TEST_DATABASE_URL 指向 PostgreSQL 测试库")
    return database_url


def reset_postgres_tables(database_url: str) -> None:
    import psycopg

    try:
        with psycopg.connect(database_url, autocommit=True) as conn:
            for table in TABLES:
                conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    except psycopg.OperationalError as exc:
        pytest.skip(f"无法连接 PostgreSQL 测试库: {exc}")


@pytest.fixture
def postgres_url():
    database_url = _database_url()
    reset_postgres_tables(database_url)
    yield database_url
    reset_postgres_tables(database_url)
