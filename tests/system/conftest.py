import pytest

import finrag.application.system as system_module
import finrag.storage as stores_module
from tests.support.fakes import (
    MemoryBM25StateStore,
    MemoryDocumentRegistry,
    MemoryIndexManifestStore,
    MemoryKnowledgeBaseRegistry,
    MemoryNodeStore,
)


@pytest.fixture(autouse=True)
def fake_postgres_redis_stack(monkeypatch):
    class TestDocumentRegistry(MemoryDocumentRegistry):
        def __init__(self, database_url):
            super().__init__(database_url)

    class TestLlamaIndexDocumentStore(MemoryNodeStore):
        def __init__(self, database_url):
            super().__init__(database_url)

        @property
        def docs(self):
            return dict(self.nodes)

        def add_documents(self, docs, allow_update=True, batch_size=2048, store_text=True):
            for node in docs:
                self.nodes[node.node_id] = node

        async def async_add_documents(self, docs, allow_update=True, batch_size=2048, store_text=True):
            self.add_documents(docs, allow_update=allow_update, batch_size=batch_size, store_text=store_text)

        def get_document(self, doc_id, raise_error=True):
            node = self.nodes.get(doc_id)
            if node is None and raise_error:
                raise ValueError(f"Document {doc_id} not found")
            return node

        async def aget_document(self, doc_id, raise_error=True):
            return self.get_document(doc_id, raise_error=raise_error)

        def document_exists(self, doc_id):
            return doc_id in self.nodes

        async def adocument_exists(self, doc_id):
            return self.document_exists(doc_id)

        def load_all_nodes(self):
            return list(self.nodes.values())

        def get_node(self, node_id):
            return self.nodes.get(node_id)

        def delete_nodes_by_document(self, document_id):
            to_delete = [nid for nid, n in self.nodes.items() if (n.metadata or {}).get("document_id") == document_id]
            for nid in to_delete:
                del self.nodes[nid]

    class TestBM25StateStore(MemoryBM25StateStore):
        def __init__(self, database_url):
            super().__init__(database_url)

    class TestIndexManifestStore(MemoryIndexManifestStore):
        def __init__(self, database_url):
            super().__init__(database_url)

    class TestKnowledgeBaseRegistry(MemoryKnowledgeBaseRegistry):
        def __init__(self, database_url):
            super().__init__(database_url)

    for target in (stores_module, system_module):
        monkeypatch.setattr(target, "PostgreSQLDocumentRegistry", TestDocumentRegistry, raising=False)
        monkeypatch.setattr(target, "PostgreSQLLlamaIndexDocumentStore", TestLlamaIndexDocumentStore, raising=False)
        monkeypatch.setattr(target, "PostgreSQLBM25StateStore", TestBM25StateStore, raising=False)
        monkeypatch.setattr(target, "PostgreSQLIndexManifestStore", TestIndexManifestStore, raising=False)
        monkeypatch.setattr(target, "PostgreSQLKnowledgeBaseRegistry", TestKnowledgeBaseRegistry, raising=False)
