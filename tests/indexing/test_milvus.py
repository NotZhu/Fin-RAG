import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

import pytest
from llama_index.core.schema import TextNode
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction

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
    IndexConstructionModule,
)


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
        model_name="BAAI/bge-m3",
        embed_model=EmbeddingDimensionProbe(1024),
    )

    store = module.init_collection()

    assert isinstance(store, RecordingMilvusVectorStore)
    assert captured["collection_name"] == MILVUS_COLLECTION_NAME
    assert captured["embedding_field"] == DENSE_EMBEDDING_FIELD
    assert captured["text_key"] == TEXT_FIELD
    assert captured["doc_id_field"] == DOC_ID_FIELD
    assert captured["dim"] == 1024


def test_embedding_requires_endpoint_and_key(monkeypatch, tmp_path):
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="EMBEDDING_BASE_URL"):
        IndexConstructionModule(model_name="BAAI/bge-m3")


def test_openai_compatible_embedding_uses_configured_endpoint(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps({"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}).encode("utf-8")

    def fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(milvus_module.urlrequest, "urlopen", fake_urlopen)

    module = IndexConstructionModule(
        model_name="BAAI/bge-m3",
        embedding_base_url="https://api.siliconflow.cn/v1",
        embedding_api_key="sf-key",
    )

    assert module.embedding_dimensions == 1024
    assert module.embed_model.get_text_embedding("现金流覆盖率") == [0.1, 0.2, 0.3]
    assert captured["url"] == "https://api.siliconflow.cn/v1/embeddings"
    assert captured["headers"]["Authorization"] == "Bearer sf-key"
    assert captured["payload"] == {"model": "BAAI/bge-m3", "input": ["现金流覆盖率"]}
    assert captured["timeout"] == 60.0


def test_openai_compatible_embedding_requires_endpoint_and_key(monkeypatch):
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="EMBEDDING_BASE_URL"):
        IndexConstructionModule(model_name="BAAI/bge-m3")


def test_index_module_rejects_removed_local_vector_paths(monkeypatch):
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    removed_model = "mo" + "ck"

    with pytest.raises(RuntimeError, match="EMBEDDING_BASE_URL"):
        IndexConstructionModule(model_name=removed_model)

    removed_embedding_option = "use_" + "mo" + "ck_embedding"
    with pytest.raises(TypeError, match=removed_embedding_option):
        IndexConstructionModule(
            model_name="BAAI/bge-m3",
            embed_model=EmbeddingDimensionProbe(1024),
            **{removed_embedding_option: True},
        )

    removed_milvus_option = "use_" + "fa" + "ke_milvus"
    with pytest.raises(TypeError, match=removed_milvus_option):
        IndexConstructionModule(
            model_name="BAAI/bge-m3",
            embed_model=EmbeddingDimensionProbe(1024),
            **{removed_milvus_option: True},
        )


def test_manifest_records_docling_node_structure_and_rejects_legacy_chunking_config(tmp_path):
    module = IndexConstructionModule(
        model_name="BAAI/bge-m3",
        embed_model=EmbeddingDimensionProbe(1024),
    )

    manifest = module.build_manifest()

    assert MANIFEST_SCHEMA_VERSION == 3
    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["index_type"] == INDEX_TYPE
    assert manifest["node_structure"] == {
        "parser": "docling_node_parser",
        "hierarchy": ["document", "section", "leaf"],
        "indexed_levels": ["leaf"],
    }
    assert "chunking" not in manifest
    with pytest.raises(TypeError, match="chunk_size"):
        module.build_manifest(chunk_size=300)


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
        model_name="BAAI/bge-m3",
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
    assert manifest["milvus"]["sparse_enabled"] is True
    assert manifest["milvus"]["sparse_provider"] == "milvus_builtin_bm25"
    assert manifest["milvus"]["sparse_index"]["metric_type"] == "BM25"
    assert manifest["milvus"]["rrf_k"] == 60
    assert "document_count" in manifest
    assert "node_count" in manifest
    assert "created_at" not in manifest
    assert "source_fingerprint" not in manifest
    assert "llamaindex_index_store_dir" not in manifest


def test_manifest_matches_ignores_source_directory_changes(tmp_path):
    class MemoryManifestStore:
        def __init__(self):
            self.manifests = {}

        def save_manifest(self, manifest, knowledge_base_id):
            self.manifests[knowledge_base_id] = dict(manifest)

        def load_manifest(self, knowledge_base_id):
            manifest = self.manifests.get(knowledge_base_id)
            return dict(manifest) if manifest is not None else None

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
            self.manifests = {}

        def save_manifest(self, manifest, knowledge_base_id):
            self.manifests[knowledge_base_id] = dict(manifest)

        def load_manifest(self, knowledge_base_id):
            manifest = self.manifests.get(knowledge_base_id)
            return dict(manifest) if manifest is not None else None

    manifest_store = MemoryManifestStore()
    module = IndexConstructionModule(
        model_name="text-embedding-v4",
        embed_model=EmbeddingDimensionProbe(1024),
        manifest_store=manifest_store,
    )
    manifest = module.build_manifest(knowledge_base_id="kb-finance")

    module.save_manifest(manifest)

    assert module.load_manifest("kb-finance") == manifest
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
        enable_sparse=False,
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


def test_milvus_vector_store_returns_llamaindex_node_payload_fields(monkeypatch):
    captured = {}

    class CapturingFactoryStore:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(milvus_module, "_load_milvus_vector_store_class", lambda: CapturingFactoryStore)
    module = IndexConstructionModule(
        model_name="custom-1024",
        embed_model=EmbeddingDimensionProbe(1024),
    )

    module.init_collection()

    assert "_node_content" in captured["output_fields"]
    assert "_node_type" in captured["output_fields"]
    assert TEXT_FIELD in captured["output_fields"]


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


def test_milvus_vector_store_receives_builtin_bm25_sparse_schema_and_rrf_ranker(monkeypatch, tmp_path):
    captured = {}

    class CapturingFactoryStore:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(milvus_module, "_load_milvus_vector_store_class", lambda: CapturingFactoryStore)
    module = IndexConstructionModule(
        model_name="custom-1024",
        embed_model=EmbeddingDimensionProbe(1024),
        rrf_k=83,
    )

    module.init_collection()
    manifest = module.build_manifest()

    assert captured["enable_sparse"] is True
    assert captured["sparse_embedding_field"] == SPARSE_EMBEDDING_FIELD
    assert isinstance(captured["sparse_embedding_function"], BM25BuiltInFunction)
    assert captured["sparse_embedding_function"].input_field_names == [TEXT_FIELD]
    assert captured["sparse_embedding_function"].output_field_names == [SPARSE_EMBEDDING_FIELD]
    assert captured["sparse_index_config"]["index_type"] == "SPARSE_INVERTED_INDEX"
    assert captured["sparse_index_config"]["metric_type"] == "BM25"
    assert captured["hybrid_ranker"] == "RRFRanker"
    assert captured["hybrid_ranker_params"] == {"k": 83}
    assert manifest["index_type"] == INDEX_TYPE
    assert manifest["milvus"]["sparse_enabled"] is True
    assert manifest["milvus"]["sparse_provider"] == "milvus_builtin_bm25"
    assert manifest["milvus"]["rrf_k"] == 83


def test_milvus_vector_store_can_disable_builtin_bm25_sparse_schema(monkeypatch):
    captured = {}

    class CapturingFactoryStore:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(milvus_module, "_load_milvus_vector_store_class", lambda: CapturingFactoryStore)
    module = IndexConstructionModule(
        model_name="custom-1024",
        embed_model=EmbeddingDimensionProbe(1024),
        enable_sparse=False,
    )

    module.init_collection()
    manifest = module.build_manifest()

    assert captured["enable_sparse"] is False
    assert captured["sparse_embedding_function"] is None
    assert manifest["milvus"]["sparse_enabled"] is False
    assert manifest["milvus"]["sparse_provider"] is None
    assert manifest["milvus"]["sparse_index"] is None


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