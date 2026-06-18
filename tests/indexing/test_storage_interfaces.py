from importlib import import_module
from inspect import Parameter, signature
import importlib.util

from llama_index.core.schema import TextNode

from finrag.indexing import (
    BM25StateStore,
    DocumentRegistryStore,
    NodeStore,
    SparseVector,
)
from tests.support.fakes import (
    MemoryBM25StateStore,
    MemoryDocumentRegistry,
    MemoryIndexManifestStore,
    MemoryNodeStore,
)


def test_store_interfaces_do_not_export_vector_backend_gateway():
    node_store = MemoryNodeStore("postgresql://test")
    registry = MemoryDocumentRegistry("postgresql://test")

    assert isinstance(node_store, NodeStore)
    assert isinstance(registry, DocumentRegistryStore)

    import finrag.indexing as indexing

    assert not hasattr(indexing, "VectorStoreGateway")


def test_bm25_state_store_contract_uses_sparse_vector_payloads():
    class MemoryBM25StateStore:
        def replace_document_chunks(self, knowledge_base_id, document_id, chunk_token_counts):
            self.knowledge_base_id = knowledge_base_id
            self.document_id = document_id
            self.chunk_token_counts = {chunk_id: dict(counts) for chunk_id, counts in chunk_token_counts.items()}

        def delete_document(self, knowledge_base_id, document_id):
            self.deleted_knowledge_base_id = knowledge_base_id
            self.deleted_document_id = document_id

        def clear(self, knowledge_base_id):
            self.cleared = knowledge_base_id

        def build_query_sparse_vector(self, knowledge_base_id, tokens):
            return SparseVector(indices=[1, 3], values=[0.5, 0.25], token_count=len(list(tokens)))

        def build_document_sparse_vector(self, knowledge_base_id, tokens):
            return SparseVector(indices=[2], values=[0.75], token_count=len(list(tokens)))

    store = MemoryBM25StateStore()
    store.replace_document_chunks("kb-finance", "doc-1", {"chunk-1": {"风险": 2}})
    sparse = store.build_query_sparse_vector("kb-finance", ["风险", "等级"])
    document_sparse = store.build_document_sparse_vector("kb-finance", ["风险"])

    assert isinstance(store, BM25StateStore)
    assert not hasattr(store, "replace_document_terms")
    assert not hasattr(store, "build_sparse_vector")
    assert sparse.indices == [1, 3]
    assert sparse.values == [0.5, 0.25]
    assert sparse.token_count == 2
    assert document_sparse.indices == [2]


def test_scoped_store_contracts_require_knowledge_base_id():
    from finrag.storage import (
        PostgreSQLBM25StateStore,
        PostgreSQLIndexManifestStore,
        PostgreSQLLlamaIndexDocumentStore,
    )

    required_parameters = [
        signature(PostgreSQLIndexManifestStore.save_manifest).parameters["knowledge_base_id"],
        signature(PostgreSQLIndexManifestStore.load_manifest).parameters["knowledge_base_id"],
        signature(PostgreSQLBM25StateStore.replace_document_chunks).parameters["knowledge_base_id"],
        signature(PostgreSQLBM25StateStore.delete_document).parameters["knowledge_base_id"],
        signature(PostgreSQLBM25StateStore.clear).parameters["knowledge_base_id"],
        signature(PostgreSQLBM25StateStore.build_query_sparse_vector).parameters["knowledge_base_id"],
        signature(PostgreSQLBM25StateStore.build_document_sparse_vector).parameters["knowledge_base_id"],
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


def test_bm25_store_keeps_statistics_per_knowledge_base():
    store = MemoryBM25StateStore()
    store.replace_document_chunks("finance", "doc-finance", {"finance-leaf": {"风险": 2, "等级": 1}})
    finance_vector_before = store.build_document_sparse_vector("finance", ["风险", "风险", "等级"])

    store.replace_document_chunks(
        "risk",
        "doc-risk",
        {
            "risk-leaf-a": {"风险": 1},
            "risk-leaf-b": {"风险": 1},
        },
    )

    finance_vector_after = store.build_document_sparse_vector("finance", ["风险", "风险", "等级"])
    risk_vector = store.build_document_sparse_vector("risk", ["风险"])

    assert finance_vector_after == finance_vector_before
    assert finance_vector_after.values != risk_vector.values


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


def test_node_store_protocol_preserves_llamaindex_textnode_contract():
    store: NodeStore = MemoryNodeStore("postgresql://test")
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
        "finrag.storage.bm25_store": "PostgreSQLBM25StateStore",
        "finrag.storage.manifest_store": "PostgreSQLIndexManifestStore",
    }

    storage_module = import_module("finrag.storage")
    for module_name, class_name in expected_exports.items():
        implementation_class = getattr(import_module(module_name), class_name)
        assert getattr(storage_module, class_name) is implementation_class

    assert importlib.util.find_spec("finrag.storage.stores") is None


def test_postgresql_scoped_stores_do_not_emit_legacy_data_migrations(monkeypatch):
    import finrag.storage.bm25_store as bm25_store_module
    import finrag.storage.llama_docstore as llama_docstore_module
    import finrag.storage.manifest_store as manifest_store_module
    from finrag.storage import (
        PostgreSQLBM25StateStore,
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

    monkeypatch.setattr(bm25_store_module, "Database", FakeDatabase)
    monkeypatch.setattr(llama_docstore_module, "Database", FakeDatabase)
    monkeypatch.setattr(manifest_store_module, "Database", FakeDatabase)

    PostgreSQLBM25StateStore("postgresql://test")
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
        "term TEXT UNIQUE NOT NULL",
    ]

    for fragment in legacy_fragments:
        assert fragment not in schema_sql

    assert "UNIQUE (knowledge_base_id, term)" in schema_sql
    assert "PRIMARY KEY (knowledge_base_id, chunk_id, term_id)" in schema_sql
