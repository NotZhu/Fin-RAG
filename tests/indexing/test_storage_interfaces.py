from importlib import import_module
import importlib.util

from llama_index.core.schema import TextNode

from finrag.indexing import (
    BM25StateStore,
    DocumentRegistryStore,
    NodeStore,
    SparseVector,
)
from tests.support.fakes import MemoryDocumentRegistry, MemoryNodeStore


def test_store_interfaces_do_not_export_vector_backend_gateway():
    node_store = MemoryNodeStore("postgresql://test")
    registry = MemoryDocumentRegistry("postgresql://test")

    assert isinstance(node_store, NodeStore)
    assert isinstance(registry, DocumentRegistryStore)

    import finrag.indexing as indexing

    assert not hasattr(indexing, "VectorStoreGateway")


def test_bm25_state_store_contract_uses_sparse_vector_payloads():
    class MemoryBM25StateStore:
        def replace_document_chunks(self, document_id, chunk_token_counts):
            self.document_id = document_id
            self.chunk_token_counts = {chunk_id: dict(counts) for chunk_id, counts in chunk_token_counts.items()}

        def delete_document(self, document_id):
            self.deleted_document_id = document_id

        def clear(self):
            self.cleared = True

        def build_query_sparse_vector(self, tokens):
            return SparseVector(indices=[1, 3], values=[0.5, 0.25], token_count=len(list(tokens)))

        def build_document_sparse_vector(self, tokens):
            return SparseVector(indices=[2], values=[0.75], token_count=len(list(tokens)))

    store = MemoryBM25StateStore()
    store.replace_document_chunks("doc-1", {"chunk-1": {"风险": 2}})
    sparse = store.build_query_sparse_vector(["风险", "等级"])
    document_sparse = store.build_document_sparse_vector(["风险"])

    assert isinstance(store, BM25StateStore)
    assert not hasattr(store, "replace_document_terms")
    assert not hasattr(store, "build_sparse_vector")
    assert sparse.indices == [1, 3]
    assert sparse.values == [0.5, 0.25]
    assert sparse.token_count == 2
    assert document_sparse.indices == [2]


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

    store.replace_document_nodes("doc-1", [node])

    restored = store.load_leaf_nodes()
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
