from pathlib import Path
from inspect import Parameter, signature
from types import SimpleNamespace

import pytest

import finrag.application.knowledge_base as knowledge_base_module
from finrag.application.document_lifecycle import DocumentLifecycleService
from finrag.application.knowledge_base import KnowledgeBaseService
from finrag.application.knowledge_base_scope import KnowledgeBaseScope
from finrag.application.qa_pipeline import QAPipelineService
from finrag.application.source_files import ManagedSourceFileService
from finrag.application.system import FinRAGSystem
from finrag.core.config import RAGConfig


def test_document_lifecycle_entrypoints_require_knowledge_base_id():
    required_parameters = [
        signature(FinRAGSystem.list_documents).parameters["knowledge_base_id"],
        signature(FinRAGSystem.delete_document).parameters["knowledge_base_id"],
        signature(FinRAGSystem.reindex_document).parameters["knowledge_base_id"],
        signature(FinRAGSystem.knowledge_base_scope).parameters["knowledge_base_id"],
        signature(FinRAGSystem.rebuild_from_sources).parameters["knowledge_base_id"],
        signature(DocumentLifecycleService.delete_document).parameters["knowledge_base_id"],
        signature(DocumentLifecycleService.reindex_document).parameters["knowledge_base_id"],
        signature(KnowledgeBaseService.rebuild_from_sources).parameters["knowledge_base_id"],
        signature(KnowledgeBaseScope.from_config).parameters["knowledge_base_id"],
    ]

    assert all(parameter.default is Parameter.empty for parameter in required_parameters)


def test_managed_source_file_service_owns_safe_paths(tmp_path):
    config = RAGConfig(data_path=str(tmp_path / "data"))
    registry = SimpleNamespace(records={})
    service = ManagedSourceFileService(config, registry)
    record = SimpleNamespace(document_id="doc-1", filename="../unsafe/policy.md", knowledge_base_id="kb-finance")

    assert service.safe_source_filename("../unsafe/policy", ".md") == "policy.md"
    assert service.pending_source_path(record) == Path(config.data_path) / ".pending" / "kb-finance" / "doc-1" / "policy.md"
    assert service.final_source_path(record) == Path(config.data_path) / "kb-finance" / "policy.md"


def test_finrag_system_delegates_ask_question_to_pipeline(tmp_path):
    system = FinRAGSystem(RAGConfig(data_path=str(tmp_path)))

    class FakePipeline:
        def __init__(self):
            self.calls = []

        def ask_question(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "response"

    fake_pipeline = FakePipeline()
    system.qa_pipeline = fake_pipeline
    ensure_calls = []
    system.ensure_knowledge_base_ready = lambda knowledge_base_id: ensure_calls.append(knowledge_base_id)

    result = system.ask_question("客户风险等级如何匹配？", knowledge_base_id="finance", return_trace=True)

    assert result == "response"
    assert ensure_calls == ["finance"]
    assert fake_pipeline.calls == [
        (
            ("客户风险等级如何匹配？",),
            {
                "return_sources": False,
                "return_trace": True,
                "knowledge_base_id": "finance",
                "event_sink": None,
                "cancel_event": None,
            },
        )
    ]
    assert isinstance(system._default_qa_pipeline(), QAPipelineService)


def test_finrag_system_does_not_expose_removed_private_forwarders():
    removed_forwarders = {
        "_retrieve_and_grade",
        "_ensure_generation_module",
        "_relevance_grader",
        "_build_sources",
        "_dedupe_evidence_nodes",
        "_node_trace",
        "_build_snippet",
        "_delete_managed_source_file",
        "_pending_source_path",
        "_final_source_path",
        "_promote_document_source_file_locked",
        "_safe_source_filename",
        "_safe_int",
        "_safe_score",
        "_elapsed_ms",
    }

    assert all(not hasattr(FinRAGSystem, name) for name in removed_forwarders)


def test_finrag_system_delegates_document_lifecycle_entrypoints(tmp_path):
    system = FinRAGSystem(RAGConfig(data_path=str(tmp_path)))

    class FakeDocumentLifecycle:
        def __init__(self):
            self.calls = []

        def prepare_uploaded_file(self, *args, **kwargs):
            self.calls.append(("prepare_uploaded_file", args, kwargs))
            return {"status": "prepared"}

        def ingest_uploaded_file(self, *args, **kwargs):
            self.calls.append(("ingest_uploaded_file", args, kwargs))
            return {"status": "ingested"}

        def index_registered_document(self, *args, **kwargs):
            self.calls.append(("index_registered_document", args, kwargs))
            return {"status": "indexed"}

        def delete_document(self, *args, **kwargs):
            self.calls.append(("delete_document", args, kwargs))
            return {"status": "deleted"}

        def reindex_document(self, *args, **kwargs):
            self.calls.append(("reindex_document", args, kwargs))
            return {"status": "reindexed"}

    fake_documents = FakeDocumentLifecycle()
    system.document_lifecycle = fake_documents
    upload_path = tmp_path / "upload.md"

    assert system.prepare_uploaded_file(upload_path, "policy.md", "kb-finance") == {"status": "prepared"}
    assert system.ingest_uploaded_file(upload_path, "policy.md", "kb-finance") == {"status": "ingested"}
    assert system.index_registered_document("doc-1") == {"status": "indexed"}
    assert system.delete_document("doc-1", "kb-finance") == {"status": "deleted"}
    assert system.reindex_document("doc-1", "kb-finance") == {"status": "reindexed"}

    assert fake_documents.calls == [
        ("prepare_uploaded_file", (upload_path, "policy.md", "kb-finance"), {}),
        ("ingest_uploaded_file", (upload_path, "policy.md", "kb-finance"), {}),
        ("index_registered_document", ("doc-1",), {}),
        ("delete_document", ("doc-1", "kb-finance"), {}),
        ("reindex_document", ("doc-1", "kb-finance"), {}),
    ]
    assert isinstance(system._default_document_lifecycle(), DocumentLifecycleService)


def test_finrag_system_delegates_knowledge_base_entrypoints(tmp_path):
    system = FinRAGSystem(RAGConfig(data_path=str(tmp_path)))

    class FakeKnowledgeBase:
        def __init__(self):
            self.calls = []

        def initialize_system(self):
            self.calls.append(("initialize_system", (), {}))

        def build_knowledge_base(self):
            self.calls.append(("build_knowledge_base", (), {}))

        def ensure_knowledge_base_ready(self):
            self.calls.append(("ensure_knowledge_base_ready", (), {}))

        def rebuild_from_sources(self, knowledge_base_id):
            self.calls.append(("rebuild_from_sources", (knowledge_base_id,), {}))
            return {"document_count": 1}

    fake_knowledge_base = FakeKnowledgeBase()
    system.knowledge_base = fake_knowledge_base

    system.initialize_system()
    system.build_knowledge_base()
    system.ensure_knowledge_base_ready()
    result = system.rebuild_from_sources("kb-finance")

    assert result == {"document_count": 1}
    assert fake_knowledge_base.calls == [
        ("initialize_system", (), {}),
        ("build_knowledge_base", (), {}),
        ("ensure_knowledge_base_ready", (), {}),
        ("rebuild_from_sources", ("kb-finance",), {}),
    ]
    assert isinstance(system._default_knowledge_base(), KnowledgeBaseService)


def test_knowledge_base_initialization_always_passes_document_registry(monkeypatch, tmp_path):
    captured = {}
    registry = SimpleNamespace(records={})

    class FakeDataPreparationModule:
        def __init__(self, data_path, *, document_registry=None, **kwargs):
            captured["data_path"] = data_path
            captured["document_registry"] = document_registry

    class FakeIndexConstructionModule:
        def __init__(self, **kwargs):
            pass

    class FakeGenerationIntegrationModule:
        llm = object()

        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(knowledge_base_module, "DataPreparationModule", FakeDataPreparationModule)
    monkeypatch.setattr(knowledge_base_module, "IndexConstructionModule", FakeIndexConstructionModule)
    monkeypatch.setattr(knowledge_base_module, "GenerationIntegrationModule", FakeGenerationIntegrationModule)

    system = SimpleNamespace(
        config=SimpleNamespace(
            data_path=str(tmp_path),
            knowledge_base_id="kb-finance",
            embedding_model="text-embedding-v4",
            milvus_collection="collection",
            milvus_host="localhost",
            milvus_port=19530,
            rrf_k=60,
            llm_model="qwen",
            temperature=0.1,
            max_tokens=512,
        ),
        document_registry=registry,
        node_store=object(),
        bm25_store=None,
        manifest_store=object(),
    )

    KnowledgeBaseService(system).initialize_system()

    assert captured["data_path"] == str(tmp_path)
    assert captured["document_registry"] is registry


def test_knowledge_base_initialization_creates_scoped_runtime(monkeypatch, tmp_path):
    captured = {}
    registry = SimpleNamespace(records={})

    class FakeDataPreparationModule:
        def __init__(self, data_path, *, knowledge_base_id=None, document_registry=None, docstore=None, **kwargs):
            self.data_path = data_path
            self.knowledge_base_id = knowledge_base_id
            self.document_registry = document_registry
            self.docstore = docstore

    class FakeIndexConstructionModule:
        def __init__(self, **kwargs):
            captured["collection_name"] = kwargs["collection_name"]
            captured["sparse_knowledge_base_id"] = kwargs["sparse_embedding_function"].knowledge_base_id
            self.collection_name = kwargs["collection_name"]
            self.sparse_embedding_function = kwargs["sparse_embedding_function"]

    class FakeGenerationIntegrationModule:
        llm = object()

        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(knowledge_base_module, "DataPreparationModule", FakeDataPreparationModule)
    monkeypatch.setattr(knowledge_base_module, "IndexConstructionModule", FakeIndexConstructionModule)
    monkeypatch.setattr(knowledge_base_module, "GenerationIntegrationModule", FakeGenerationIntegrationModule)

    config = SimpleNamespace(
        data_path=str(tmp_path),
        knowledge_base_id="finance",
        embedding_model="text-embedding-v4",
        milvus_collection="finrag_leaf_nodes",
        milvus_host="localhost",
        milvus_port=19530,
        rrf_k=60,
        llm_model="qwen",
        temperature=0.1,
        max_tokens=512,
    )
    system = SimpleNamespace(
        config=config,
        document_registry=registry,
        llama_docstore=object(),
        bm25_store=object(),
        manifest_store=object(),
        kb_runtimes={},
        knowledge_base_scope=lambda knowledge_base_id: KnowledgeBaseScope.from_config(config, knowledge_base_id),
    )

    KnowledgeBaseService(system).initialize_system("risk")

    runtime = system.kb_runtimes["risk"]
    assert runtime.scope.collection_name == "finrag_leaf_nodes__kb_risk"
    assert runtime.data_module.knowledge_base_id == "risk"
    assert runtime.index_module.collection_name == "finrag_leaf_nodes__kb_risk"
    assert runtime.generation_module.llm is not None
    assert captured == {
        "collection_name": "finrag_leaf_nodes__kb_risk",
        "sparse_knowledge_base_id": "risk",
    }
    assert system.index_module is runtime.index_module


def test_knowledge_base_initialization_does_not_pass_ocr_config_to_data_module(monkeypatch, tmp_path):
    captured = {}
    registry = SimpleNamespace(records={})

    class FakeDataPreparationModule:
        def __init__(self, data_path, **kwargs):
            captured.update(kwargs)

    class FakeIndexConstructionModule:
        def __init__(self, **kwargs):
            self.sparse_embedding_function = kwargs["sparse_embedding_function"]

    class FakeGenerationIntegrationModule:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(knowledge_base_module, "DataPreparationModule", FakeDataPreparationModule)
    monkeypatch.setattr(knowledge_base_module, "IndexConstructionModule", FakeIndexConstructionModule)
    monkeypatch.setattr(knowledge_base_module, "GenerationIntegrationModule", FakeGenerationIntegrationModule)

    system = SimpleNamespace(
        config=RAGConfig(
            data_path=str(tmp_path),
        ),
        document_registry=registry,
        llama_docstore=object(),
        bm25_store=None,
        manifest_store=object(),
        kb_runtimes={},
        knowledge_base_scope=lambda knowledge_base_id: KnowledgeBaseScope.from_config(system.config, knowledge_base_id),
    )

    KnowledgeBaseService(system).initialize_system("risk")

    assert "ocr_enabled" not in captured
    assert "ocr_lang" not in captured
    assert "tesseract_cmd" not in captured


def test_build_knowledge_base_assumes_registry_management(tmp_path):
    calls = []

    class FakeWriteLock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    system = SimpleNamespace(
        _write_lock=FakeWriteLock(),
        config=SimpleNamespace(data_path=str(tmp_path), knowledge_base_id="kb-finance"),
        data_module=SimpleNamespace(storage_context=None),
        index_module=SimpleNamespace(load_index=lambda *args, **kwargs: None, manifest_matches=lambda m: False),
        _ensure_modules=lambda knowledge_base_id=None: None,
        _configure_knowledge_base_scope_locked=lambda knowledge_base_id: knowledge_base_id,
        _build_expected_manifest=lambda knowledge_base_id: {"schema_version": 1},
        _load_leaf_nodes_from_docstore=lambda knowledge_base_id: [],
        _refresh_retrieval=lambda vector_index, leaf_nodes, knowledge_base_id: calls.append(
            ("refresh", vector_index, leaf_nodes, knowledge_base_id)
        ),
        _full_rebuild_locked=lambda knowledge_base_id: calls.append(("rebuild", knowledge_base_id)),
    )

    KnowledgeBaseService(system).build_knowledge_base()

    assert calls == [("rebuild", "kb-finance")]


def test_load_leaf_nodes_from_docstore_returns_only_leaf_nodes():
    all_nodes = [object(), object()]
    leaf_nodes = [all_nodes[1]]
    captured = {}
    system = FinRAGSystem.__new__(FinRAGSystem)
    system.llama_docstore = SimpleNamespace(load_all_nodes=lambda knowledge_base_id: all_nodes)
    system.index_module = SimpleNamespace(sparse_embedding_function=None)
    system.knowledge_base_scope = lambda knowledge_base_id: SimpleNamespace(
        knowledge_base_id=knowledge_base_id,
        runtime_cache_key=knowledge_base_id,
        collection_name="collection",
    )

    def load_prepared_nodes(nodes):
        captured["nodes"] = nodes
        return leaf_nodes

    system.data_module = SimpleNamespace(load_prepared_nodes=load_prepared_nodes)

    assert system._load_leaf_nodes_from_docstore("kb-finance") is leaf_nodes
    assert captured["nodes"] is all_nodes


def test_rebuild_via_pipeline_propagates_pipeline_errors(monkeypatch):
    import finrag.indexing.nodes as nodes_module

    def fail_build_ingestion_pipeline(*args, **kwargs):
        raise RuntimeError("pipeline failed")

    def fail_classic_rebuild():
        raise AssertionError("classic rebuild should not run")

    monkeypatch.setattr(nodes_module, "build_ingestion_pipeline", fail_build_ingestion_pipeline)

    system = FinRAGSystem.__new__(FinRAGSystem)
    system.config = SimpleNamespace()
    system.llama_docstore = object()
    system.data_module = SimpleNamespace(
        documents=[object()],
        all_nodes=[],
        chunks=[],
        chunk_documents=fail_classic_rebuild,
    )
    system.index_module = SimpleNamespace(
        embed_model=object(),
        init_collection=lambda reset: object(),
    )

    with pytest.raises(RuntimeError, match="pipeline failed"):
        system._rebuild_via_pipeline("kb-finance")


def test_rebuild_via_pipeline_stores_hierarchy_nodes_and_indexes_only_leaf_nodes(monkeypatch):
    import finrag.indexing.nodes as nodes_module
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

    parent = TextNode(
        text="parent node",
        id_="parent-1",
        metadata={
            "document_id": "doc-1",
            "knowledge_base_id": "kb-finance",
            "chunk_id": "parent-1",
            "chunk_level": 2,
            "chunk_idx": 0,
        },
        relationships={NodeRelationship.CHILD: [RelatedNodeInfo(node_id="leaf-1")]},
    )
    leaf = TextNode(
        text="leaf node",
        id_="leaf-1",
        metadata={
            "document_id": "doc-1",
            "knowledge_base_id": "kb-finance",
            "chunk_id": "leaf-1",
            "chunk_level": 3,
            "chunk_idx": 0,
        },
        relationships={NodeRelationship.PARENT: RelatedNodeInfo(node_id="parent-1")},
    )
    captured = {}

    class FakePipeline:
        def run(self, *, documents, show_progress=False):
            captured["pipeline_documents"] = documents
            return [parent, leaf]

    class FakeDocstore:
        def delete_knowledge_base(self, knowledge_base_id):
            captured["deleted_kb"] = knowledge_base_id

        def add_documents(self, nodes):
            captured["stored_node_ids"] = [node.node_id for node in nodes]

    class FakeIndexModule:
        embed_model = object()

        def init_collection(self, *, reset=False):
            captured["init_reset"] = reset
            return object()

        def build_vector_index(self, nodes, *, storage_context=None, reset=True):
            captured["indexed_node_ids"] = [node.node_id for node in nodes]
            captured["index_reset"] = reset
            return "vector-index"

    monkeypatch.setattr(nodes_module, "build_ingestion_pipeline", lambda *args, **kwargs: FakePipeline())

    system = FinRAGSystem.__new__(FinRAGSystem)
    system.config = SimpleNamespace()
    system.llama_docstore = FakeDocstore()
    system.data_module = SimpleNamespace(
        documents=[object()],
        all_nodes=[],
        chunks=[],
        storage_context=object(),
    )
    system.index_module = FakeIndexModule()
    system.bm25_store = None

    leaf_nodes = system._rebuild_via_pipeline("kb-finance")

    assert leaf_nodes == [leaf]
    assert captured["deleted_kb"] == "kb-finance"
    assert captured["stored_node_ids"] == ["parent-1", "leaf-1"]
    assert captured["indexed_node_ids"] == ["leaf-1"]
    assert captured["index_reset"] is False
