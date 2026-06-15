import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from llama_index.embeddings import dashscope as dashscope_module
from llama_index.core.schema import TextNode

import finrag.indexing.milvus as milvus_module
from finrag.core.config import PROJECT_ROOT
from finrag.indexing.milvus import (
    DENSE_EMBEDDING_FIELD,
    DOC_ID_FIELD,
    INDEX_TYPE,
    MANIFEST_SCHEMA_VERSION,
    MILVUS_COLLECTION_NAME,
    SPARSE_EMBEDDING_FIELD,
    TEXT_FIELD,
    BM25SparseEmbeddingFunction,
    IndexConstructionModule,
)
from finrag.storage import PostgreSQLBM25StateStore


class EmbeddingDimensionProbe:
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim


class EmbeddingWithoutDimensionProbe:
    pass


def _node(node_id: str, text: str, document_id: str = "doc-1") -> TextNode:
    return TextNode(
        text=text,
        id_=node_id,
        metadata={
            "chunk_id": node_id,
            "document_id": document_id,
            "knowledge_base_id": "kb-finance",
            "filename": "policy.txt",
            "chunk_level": 3,
        },
    )


def test_milvus_dense_index_uses_llamaindex_milvus_vector_store_by_default(monkeypatch):
    captured = {}

    class RecordingMilvusVectorStore:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(milvus_module, "_load_milvus_vector_store_class", lambda: RecordingMilvusVectorStore)
    module = IndexConstructionModule(
        model_name="text-embedding-v4",
        embed_model=EmbeddingDimensionProbe(1024),
    )

    store = module.init_collection()

    assert isinstance(store, RecordingMilvusVectorStore)
    assert captured["collection_name"] == MILVUS_COLLECTION_NAME
    assert captured["embedding_field"] == DENSE_EMBEDDING_FIELD
    assert captured["text_key"] == TEXT_FIELD
    assert captured["doc_id_field"] == DOC_ID_FIELD
    assert captured["dim"] == 1024


def test_dashscope_text_embedding_v4_uses_known_default_dimension(monkeypatch):
    captured = {}

    class RecordingMilvusVectorStore:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(milvus_module, "_load_milvus_vector_store_class", lambda: RecordingMilvusVectorStore)
    module = IndexConstructionModule(
        model_name="text-embedding-v4",
        embed_model=EmbeddingWithoutDimensionProbe(),
    )

    module.init_collection()

    assert module.embedding_dimensions == 1024
    assert captured["dim"] == 1024


def test_embedding_requires_dashscope_key(monkeypatch, tmp_path):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        IndexConstructionModule(model_name="text-embedding-v4")


def test_dashscope_embedding_uses_api_safe_batch_size(monkeypatch):
    captured = {}

    class RecordingDashScopeEmbedding:
        def __init__(self, model_name, api_key, embed_batch_size=25):
            captured["model_name"] = model_name
            captured["api_key"] = api_key
            captured["embed_batch_size"] = embed_batch_size
            self.embed_dim = 1024

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(dashscope_module, "DashScopeEmbedding", RecordingDashScopeEmbedding)

    IndexConstructionModule(model_name="text-embedding-v4")

    assert captured == {
        "model_name": "text-embedding-v4",
        "api_key": "test-key",
        "embed_batch_size": 10,
    }


def test_index_module_rejects_removed_local_vector_paths(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    removed_model = "mo" + "ck"

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        IndexConstructionModule(model_name=removed_model)

    removed_embedding_option = "use_" + "mo" + "ck_embedding"
    with pytest.raises(TypeError, match=removed_embedding_option):
        IndexConstructionModule(
            model_name="text-embedding-v4",
            embed_model=EmbeddingDimensionProbe(1024),
            **{removed_embedding_option: True},
        )

    removed_milvus_option = "use_" + "fa" + "ke_milvus"
    with pytest.raises(TypeError, match=removed_milvus_option):
        IndexConstructionModule(
            model_name="text-embedding-v4",
            embed_model=EmbeddingDimensionProbe(1024),
            **{removed_milvus_option: True},
        )


def test_manifest_changes_when_hierarchical_chunking_config_changes(tmp_path):
    module = IndexConstructionModule(
        model_name="text-embedding-v4",
        embed_model=EmbeddingDimensionProbe(1024),
    )

    baseline = module.build_manifest(chunk_size=300, chunk_overlap=60)
    changed_size = module.build_manifest(chunk_size=600, chunk_overlap=60)
    changed_overlap = module.build_manifest(chunk_size=300, chunk_overlap=30)

    assert baseline["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert baseline["index_type"] == INDEX_TYPE
    assert baseline["chunking"]["chunk_sizes"] == [1200, 600, 300]
    assert baseline["chunking"]["chunk_overlap"] == 60
    assert baseline["chunking"] != changed_size["chunking"]
    assert baseline["chunking"] != changed_overlap["chunking"]


def test_manifest_records_actual_embedding_dimension_for_injected_model(tmp_path):
    module = IndexConstructionModule(
        model_name="custom-1024",
        embed_model=EmbeddingDimensionProbe(1024),
    )

    manifest = module.build_manifest()

    assert manifest["embedding"] == {"model": "custom-1024", "dimensions": 1024}


def test_delete_vectors_by_document_id_uses_document_metadata_filter():
    class RecordingVectorStore:
        def __init__(self):
            self.filters = None

        def delete_nodes(self, filters=None):
            self.filters = filters

    vector_store = RecordingVectorStore()
    module = IndexConstructionModule(
        model_name="text-embedding-v4",
        embed_model=EmbeddingDimensionProbe(1024),
    )
    module.vector_store = vector_store

    module.delete_vectors_by_document_id("doc-a")

    assert vector_store.filters is not None
    assert vector_store.filters.filters[0].key == DOC_ID_FIELD
    assert vector_store.filters.filters[0].value == "doc-a"


def test_manifest_uses_minimal_registry_managed_shape(tmp_path):
    (tmp_path / "policy.md").write_text("旧制度", encoding="utf-8")
    module = IndexConstructionModule(
        model_name="text-embedding-v4",
        embed_model=EmbeddingDimensionProbe(1024),
    )

    manifest = module.build_manifest()

    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["index_type"] == "LlamaIndexRouter"
    assert manifest["embedding"] == {"model": "text-embedding-v4", "dimensions": 1024}
    assert manifest["milvus"]["collection"] == MILVUS_COLLECTION_NAME
    assert manifest["milvus"]["sparse_enabled"] is False
    assert manifest["milvus"]["rrf_k"] == 60
    assert "document_count" in manifest
    assert "node_count" in manifest
    assert "summary_document_count" in manifest
    assert "created_at" not in manifest
    assert "source_fingerprint" not in manifest


def test_manifest_matches_ignores_source_directory_changes(tmp_path):
    class MemoryManifestStore:
        def __init__(self):
            self.manifest = None

        def save_manifest(self, manifest):
            self.manifest = dict(manifest)

        def load_manifest(self):
            return dict(self.manifest) if self.manifest is not None else None

    manifest_store = MemoryManifestStore()
    module = IndexConstructionModule(
        model_name="text-embedding-v4",
        embed_model=EmbeddingDimensionProbe(1024),
        manifest_store=manifest_store,
    )
    source = tmp_path / "policy.md"
    source.write_text("旧制度", encoding="utf-8")
    module.save_manifest(module.build_manifest())

    source.write_text("新制度", encoding="utf-8")
    expected_manifest = module.build_manifest()

    assert module.manifest_matches(expected_manifest) is True


def test_index_module_can_delegate_manifest_persistence_to_store(tmp_path):
    class MemoryManifestStore:
        def __init__(self):
            self.manifest = None

        def save_manifest(self, manifest):
            self.manifest = dict(manifest)

        def load_manifest(self):
            return dict(self.manifest) if self.manifest is not None else None

    manifest_store = MemoryManifestStore()
    module = IndexConstructionModule(
        model_name="text-embedding-v4",
        embed_model=EmbeddingDimensionProbe(1024),
        manifest_store=manifest_store,
    )
    manifest = module.build_manifest()

    module.save_manifest(manifest)

    assert module.load_manifest() == manifest
    assert not (tmp_path / "milvus_manifest" / "index_manifest.json").exists()


def test_index_module_rejects_removed_index_save_path():
    with pytest.raises(TypeError, match="index_save_path"):
        IndexConstructionModule(model_name="text-embedding-v4", index_save_path="storage/index_manifest")


def test_manifest_persistence_requires_postgres_store(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    module = IndexConstructionModule(
        model_name="text-embedding-v4",
        embed_model=EmbeddingDimensionProbe(1024),
    )
    manifest = module.build_manifest()

    with pytest.raises(RuntimeError, match="PostgreSQL index manifest store"):
        module.save_manifest(manifest)

    assert not (tmp_path / "storage" / "index_manifest" / "index_manifest.json").exists()


def test_milvus_vector_store_receives_llamaindex_dense_schema(monkeypatch, tmp_path):
    captured = {}

    class CapturingFactoryStore:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(milvus_module, "_load_milvus_vector_store_class", lambda: CapturingFactoryStore)
    module = IndexConstructionModule(
        model_name="custom-1024",
        embed_model=EmbeddingDimensionProbe(1024),
    )

    store = module.init_collection()

    assert store is module.vector_store
    assert captured["collection_name"] == MILVUS_COLLECTION_NAME
    assert captured["dim"] == 1024
    assert captured["embedding_field"] == DENSE_EMBEDDING_FIELD
    assert captured["doc_id_field"] == DOC_ID_FIELD
    assert DOC_ID_FIELD not in captured["scalar_field_names"]
    assert "page_number" not in captured["scalar_field_names"]
    assert DOC_ID_FIELD in captured["output_fields"]
    assert captured["text_key"] == TEXT_FIELD
    assert captured["enable_sparse"] is False
    assert captured["index_config"]["index_type"] == "HNSW"
    assert captured["index_config"]["metric_type"] == "IP"
    assert captured["search_config"] == {"ef": 64}
    assert "chunk_id" in captured["scalar_field_names"]
    assert "root_chunk_id" in captured["scalar_field_names"]


def test_milvus_vector_store_can_initialize_in_worker_thread_without_existing_event_loop(monkeypatch, tmp_path):
    captured = {}

    class EventLoopCheckingStore:
        def __init__(self, **kwargs):
            captured["loop"] = asyncio.get_event_loop()
            captured.update(kwargs)

    monkeypatch.setattr(milvus_module, "_load_milvus_vector_store_class", lambda: EventLoopCheckingStore)
    module = IndexConstructionModule(
        model_name="custom-1024",
        embed_model=EmbeddingDimensionProbe(1024),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        store = executor.submit(module.init_collection).result()

    assert store is module.vector_store
    assert captured["loop"] is not None
    assert captured["collection_name"] == MILVUS_COLLECTION_NAME


def test_milvus_vector_store_receives_sparse_schema_and_rrf_ranker(monkeypatch, tmp_path):
    captured = {}

    class CapturingFactoryStore:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class StaticSparseEmbeddingFunction:
        def encode_queries(self, queries):
            return [{1: 1.0} for _ in queries]

        def encode_documents(self, documents):
            return [{1: 1.0} for _ in documents]

    sparse_embedding = StaticSparseEmbeddingFunction()
    monkeypatch.setattr(milvus_module, "_load_milvus_vector_store_class", lambda: CapturingFactoryStore)
    module = IndexConstructionModule(
        model_name="custom-1024",
        embed_model=EmbeddingDimensionProbe(1024),
        sparse_embedding_function=sparse_embedding,
        rrf_k=83,
    )

    module.init_collection()
    manifest = module.build_manifest()

    assert captured["enable_sparse"] is True
    assert captured["sparse_embedding_field"] == SPARSE_EMBEDDING_FIELD
    assert captured["sparse_embedding_function"] is sparse_embedding
    assert captured["sparse_index_config"]["index_type"] == "SPARSE_INVERTED_INDEX"
    assert captured["sparse_index_config"]["metric_type"] == "IP"
    assert captured["hybrid_ranker"] == "RRFRanker"
    assert captured["hybrid_ranker_params"] == {"k": 83}
    assert manifest["index_type"] == INDEX_TYPE
    assert manifest["milvus"]["sparse_enabled"] is True
    assert manifest["milvus"]["rrf_k"] == 83


def test_bm25_sparse_embedding_function_uses_persistent_state_for_queries_and_documents(postgres_url):
    store = PostgreSQLBM25StateStore(postgres_url)
    store.replace_document_chunks("doc-a", {"leaf-a": {"风险": 2, "等级": 1}})
    sparse_embedding = BM25SparseEmbeddingFunction(store)

    query_vectors = sparse_embedding.encode_queries(["风险等级"])
    document_vectors = sparse_embedding.encode_documents(["风险 风险 等级"])

    assert query_vectors[0]
    assert document_vectors[0]
    assert set(query_vectors[0]).issubset(set(document_vectors[0]))
    assert set(query_vectors[0].values()) == {1.0}
    assert max(document_vectors[0].values()) != 1.0


def test_project_no_longer_imports_chroma_or_exports_removed_vector_gateway():
    forbidden = [
        "import " + "chromadb",
        "vector_stores." + "chroma",
        "Chroma" + "VectorStore",
        "_In" + "Memory" + "Milvus",
        "memory:" + "//" + "finrag",
        "use_" + "fa" + "ke_milvus",
        "use_" + "mo" + "ck_embedding",
        "Mo" + "ck" + "Embedding",
        "Mo" + "ck" + "LLM",
        "model_name=" + '"mo' + 'ck"',
        "embedding_model=" + '"mo' + 'ck"',
    ]
    for root_name in ["src", "tests"]:
        for path in (PROJECT_ROOT / root_name).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for needle in forbidden:
                assert needle not in text, f"{needle!r} remains in {path}"

    import finrag.indexing as indexing

    assert not hasattr(indexing, "SimpleVectorStore")
    assert not hasattr(indexing, "HashEmbedding")
    assert not hasattr(indexing, "VectorStoreGateway")
