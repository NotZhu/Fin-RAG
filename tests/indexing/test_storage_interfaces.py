from importlib import import_module
from inspect import Parameter, signature
import importlib.util

from llama_index.core.schema import TextNode

from tests.support.fakes import (
    MemoryIndexManifestStore,
    MemoryNodeStore,
)


def test_indexing_does_not_export_removed_public_contracts():
    import finrag.indexing as indexing

    assert not hasattr(indexing, "VectorStoreGateway")
    assert not hasattr(indexing, "DocumentRegistryStore")
    assert not hasattr(indexing, "IndexManifestStore")
    assert not hasattr(indexing, "NodeStore")


def test_scoped_store_contracts_require_knowledge_base_id():
    from finrag.storage import (
        PostgreSQLIndexManifestStore,
        PostgreSQLLlamaIndexDocumentStore,
    )

    required_parameters = [
        signature(PostgreSQLIndexManifestStore.save_manifest).parameters["knowledge_base_id"],
        signature(PostgreSQLIndexManifestStore.load_manifest).parameters["knowledge_base_id"],
        signature(PostgreSQLLlamaIndexDocumentStore.load_all_nodes).parameters["knowledge_base_id"],
        signature(PostgreSQLLlamaIndexDocumentStore.delete_nodes_by_document).parameters["knowledge_base_id"],
    ]

    assert all(parameter.default is Parameter.empty for parameter in required_parameters)


def test_manifest_store_keeps_payloads_per_knowledge_base():
    store = MemoryIndexManifestStore()

    store.save_manifest({"knowledge_base_id": "finance", "version": 1}, "finance")
    store.save_manifest({"knowledge_base_id": "risk", "version": 2}, "risk")

    assert store.load_manifest("finance") == {"knowledge_base_id": "finance", "version": 1}
    assert store.load_manifest("risk") == {"knowledge_base_id": "risk", "version": 2}


def test_node_store_loads_and_deletes_nodes_by_knowledge_base():
    store = MemoryNodeStore()
    finance_node = TextNode(
        text="finance",
        id_="finance-leaf",
        metadata={"document_id": "doc-finance", "knowledge_base_id": "finance", "chunk_level": 3, "chunk_idx": 0},
    )
    risk_node = TextNode(
        text="risk",
        id_="risk-leaf",
        metadata={"document_id": "doc-risk", "knowledge_base_id": "risk", "chunk_level": 3, "chunk_idx": 0},
    )
    store.replace_document_nodes("doc-finance", [finance_node], "finance")
    store.replace_document_nodes("doc-risk", [risk_node], "risk")

    assert [node.node_id for node in store.load_all_nodes("finance")] == ["finance-leaf"]
    assert [node.node_id for node in store.load_all_nodes("risk")] == ["risk-leaf"]

    store.delete_document("doc-risk", "risk")

    assert [node.node_id for node in store.load_all_nodes("finance")] == ["finance-leaf"]
    assert store.load_all_nodes("risk") == []


def test_memory_node_store_preserves_llamaindex_textnode_contract():
    store = MemoryNodeStore("postgresql://test")
    node = TextNode(
        text="客户风险等级",
        id_="doc-1-leaf",
        metadata={
            "chunk_id": "doc-1-leaf",
            "document_id": "doc-1",
            "knowledge_base_id": "kb-finance",
            "chunk_level": 3,
            "chunk_idx": 0,
        },
    )

    store.replace_document_nodes("doc-1", [node], "kb-finance")

    restored = store.load_leaf_nodes("kb-finance")
    assert restored[0].node_id == "doc-1-leaf"
    assert isinstance(restored[0], TextNode)


def test_postgresql_store_implementations_are_split_and_reexported():
    expected_exports = {
        "finrag.storage.document_registry": "PostgreSQLDocumentRegistry",
        "finrag.storage.llama_docstore": "PostgreSQLLlamaIndexDocumentStore",
        "finrag.storage.manifest_store": "PostgreSQLIndexManifestStore",
    }

    storage_module = import_module("finrag.storage")
    for module_name, class_name in expected_exports.items():
        implementation_class = getattr(import_module(module_name), class_name)
        assert getattr(storage_module, class_name) is implementation_class

    assert importlib.util.find_spec("finrag.storage.stores") is None
    removed_sparse_store = "finrag.storage." + "bm25" + "_store"
    assert importlib.util.find_spec(removed_sparse_store) is None


def test_postgresql_scoped_stores_do_not_emit_legacy_data_migrations(monkeypatch):
    import finrag.storage.llama_docstore as llama_docstore_module
    import finrag.storage.manifest_store as manifest_store_module
    from finrag.storage import (
        PostgreSQLIndexManifestStore,
        PostgreSQLLlamaIndexDocumentStore,
    )

    statements = []

    class FakeConnection:
        def execute(self, sql, params=()):
            statements.append(" ".join(sql.split()))
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
    monkeypatch.setattr(manifest_store_module, "Database", FakeDatabase)

    PostgreSQLIndexManifestStore("postgresql://test")
    PostgreSQLLlamaIndexDocumentStore("postgresql://test")

    schema_sql = "\n".join(statements)
    legacy_fragments = [
        "ADD COLUMN IF NOT EXISTS",
        "SET knowledge_base_id = 'finance'",
        "ALTER COLUMN knowledge_base_id SET NOT NULL",
        "DROP CONSTRAINT",
        "DROP COLUMN IF EXISTS id",
        "DEFAULT 'finance'",
    ]

    for fragment in legacy_fragments:
        assert fragment not in schema_sql
