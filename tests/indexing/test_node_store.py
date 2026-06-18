"""Tests for PostgreSQLLlamaIndexDocumentStore."""

from llama_index.core.schema import TextNode

from finrag.storage import PostgreSQLLlamaIndexDocumentStore
import finrag.storage.llama_docstore as llama_docstore_module
import finrag.storage as stores_module


def _store(postgres_url):
    return PostgreSQLLlamaIndexDocumentStore(postgres_url)


def test_store_persists_and_loads_nodes(postgres_url):
    store = _store(postgres_url)
    node = TextNode(
        text="客户风险等级匹配",
        id_="chunk-test",
        metadata={
            "document_id": "doc-1",
            "knowledge_base_id": "kb-finance",
            "chunk_id": "chunk-test",
            "chunk_level": 3,
            "chunk_idx": 0,
        },
    )
    store.add_documents([node])

    loaded = store.load_all_nodes("kb-finance")
    assert any(n.node_id == "chunk-test" for n in loaded)

    retrieved = store.get_node("chunk-test")
    assert retrieved is not None
    assert retrieved.text == "客户风险等级匹配"


def test_store_deletes_nodes_by_document(postgres_url):
    store = _store(postgres_url)
    node = TextNode(
        text="删除测试",
        id_="chunk-del",
        metadata={
            "document_id": "doc-del",
            "knowledge_base_id": "kb-finance",
            "chunk_id": "chunk-del",
            "chunk_level": 3,
            "chunk_idx": 0,
        },
    )
    store.add_documents([node])

    store.delete_nodes_by_document("doc-del", "kb-finance")
    assert store.get_node("chunk-del") is None


def test_store_deletes_document_ref_doc_and_hashes(postgres_url):
    store = _store(postgres_url)
    target_node = TextNode(
        text="删除关联状态",
        id_="chunk-del-associated",
        metadata={
            "document_id": "doc-del",
            "knowledge_base_id": "kb-finance",
            "chunk_id": "chunk-del-associated",
            "chunk_level": 3,
            "chunk_idx": 0,
        },
    )
    other_node = TextNode(
        text="其他文档保留",
        id_="chunk-keep",
        metadata={
            "document_id": "doc-keep",
            "knowledge_base_id": "kb-finance",
            "chunk_id": "chunk-keep",
            "chunk_level": 3,
            "chunk_idx": 0,
        },
    )
    store.add_documents([target_node, other_node])
    store.set_document_hash("chunk-del-associated", "hash-del")
    store.set_document_hash("chunk-keep", "hash-keep")

    store.delete_nodes_by_document("doc-del", "kb-finance")

    assert store.get_node("chunk-del-associated") is None
    assert store.get_document_hash("chunk-del-associated") is None
    assert store.get_ref_doc_info("doc-del") is None
    assert store.get_node("chunk-keep") is not None
    assert store.get_document_hash("chunk-keep") == "hash-keep"
    assert store.get_ref_doc_info("doc-keep") is not None


def test_delete_nodes_by_document_cleans_associated_docstore_tables(monkeypatch):
    statements = []

    class FakeConnection:
        def execute(self, sql, params=()):
            statements.append((" ".join(sql.split()), params))
            return self

    class FakeConnectionContext:
        def __enter__(self):
            return FakeConnection()

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeDatabase:
        def __init__(self, database_url):
            self.database_url = database_url

        def connect(self):
            return FakeConnectionContext()

    monkeypatch.setattr(llama_docstore_module, "Database", FakeDatabase)
    store = stores_module.PostgreSQLLlamaIndexDocumentStore("postgresql://test")
    statements.clear()

    store.delete_nodes_by_document("doc-del", "kb-finance")

    assert statements == [
        (
            "DELETE FROM finrag_llama_doc_hashes "
            "WHERE chunk_id IN ( SELECT chunk_id FROM finrag_chunks "
            "WHERE knowledge_base_id = %s AND document_id = %s )",
            ("kb-finance", "doc-del"),
        ),
        (
            "DELETE FROM finrag_chunks WHERE knowledge_base_id = %s AND document_id = %s",
            ("kb-finance", "doc-del"),
        ),
        ("DELETE FROM finrag_ref_docs WHERE ref_doc_id = %s", ("doc-del",)),
    ]


def test_store_returns_none_for_missing_node(postgres_url):
    store = _store(postgres_url)
    assert store.get_node("nonexistent") is None


def test_store_exports_expected_methods():
    assert hasattr(PostgreSQLLlamaIndexDocumentStore, "add_documents")
    assert hasattr(PostgreSQLLlamaIndexDocumentStore, "get_node")
    assert hasattr(PostgreSQLLlamaIndexDocumentStore, "load_all_nodes")
    assert hasattr(PostgreSQLLlamaIndexDocumentStore, "delete_nodes_by_document")
    assert hasattr(PostgreSQLLlamaIndexDocumentStore, "delete_ref_doc")
    assert hasattr(PostgreSQLLlamaIndexDocumentStore, "docs")


def test_hash_table_schema_uses_chunk_id_column(monkeypatch):
    statements = []

    class FakeConnection:
        def execute(self, sql, params=()):
            statements.append(sql)
            return self

    class FakeConnectionContext:
        def __enter__(self):
            return FakeConnection()

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeDatabase:
        def __init__(self, database_url):
            self.database_url = database_url

        def connect(self):
            return FakeConnectionContext()

    monkeypatch.setattr(llama_docstore_module, "Database", FakeDatabase)

    stores_module.PostgreSQLLlamaIndexDocumentStore("postgresql://test")

    hash_table_statements = [
        statement
        for statement in statements
        if "CREATE TABLE IF NOT EXISTS finrag_llama_doc_hashes" in statement
    ]
    assert len(hash_table_statements) == 1
    assert "chunk_id TEXT PRIMARY KEY" in hash_table_statements[0]
    assert "doc_id TEXT PRIMARY KEY" not in hash_table_statements[0]
