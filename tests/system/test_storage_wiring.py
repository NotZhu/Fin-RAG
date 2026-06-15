from finrag.core.config import RAGConfig
import finrag.application.system as system_module
import finrag.storage as stores_module


def test_system_uses_postgres_stores_and_redis_cache_when_enabled(monkeypatch, tmp_path):
    constructed = {}

    class FakeRegistry:
        def __init__(self, database_url):
            constructed["registry_url"] = database_url
            self.records = {}

    class FakeDocStore:
        def __init__(self, database_url):
            constructed["docstore_url"] = database_url

    class FakeBM25Store:
        def __init__(self, database_url):
            constructed["bm25_url"] = database_url

    class FakeManifestStore:
        def __init__(self, database_url):
            constructed["manifest_url"] = database_url

    for target in (stores_module, system_module):
        monkeypatch.setattr(target, "PostgreSQLDocumentRegistry", FakeRegistry, raising=False)
        monkeypatch.setattr(target, "PostgreSQLLlamaIndexDocumentStore", FakeDocStore, raising=False)
        monkeypatch.setattr(target, "PostgreSQLBM25StateStore", FakeBM25Store, raising=False)
        monkeypatch.setattr(target, "PostgreSQLIndexManifestStore", FakeManifestStore, raising=False)

    config = RAGConfig(
        data_path=str(tmp_path),
        database_url="postgresql://finrag:test@localhost:5432/finrag",
        redis_url="redis://localhost:6379/0",
    )

    rag = system_module.FinRAGSystem(config)

    assert isinstance(rag.document_registry, FakeRegistry)
    assert isinstance(rag.bm25_store, FakeBM25Store)
    assert isinstance(rag.manifest_store, FakeManifestStore)
