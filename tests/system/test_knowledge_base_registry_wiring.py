from types import SimpleNamespace

import pytest

from finrag.application.knowledge_base_scope import KnowledgeBaseScope
from finrag.application.system import FinRAGSystem
from finrag.core.config import RAGConfig
from finrag.core.node_schema import TextNode
from finrag.ingestion import DocumentRecord
from finrag.storage.knowledge_base_registry import (
    KnowledgeBaseArchivedError,
    KnowledgeBaseNotFoundError,
    ProtectedKnowledgeBaseError,
)


def test_finrag_system_exposes_default_knowledge_base(tmp_path):
    system = FinRAGSystem(RAGConfig(data_path=str(tmp_path / "documents"), knowledge_base_id="finance"))

    knowledge_bases = system.list_knowledge_bases()

    assert knowledge_bases[0]["knowledge_base_id"] == "finance"
    assert knowledge_bases[0]["document_count"] == 0
    assert knowledge_bases[0]["status"] == "active"


def test_finrag_system_can_create_knowledge_base_by_id(tmp_path):
    system = FinRAGSystem(RAGConfig(data_path=str(tmp_path / "documents"), knowledge_base_id="finance"))

    created = system.create_knowledge_base("risk")

    assert created["knowledge_base_id"] == "risk"
    assert created["status"] == "active"
    assert system.list_knowledge_bases()[-1]["knowledge_base_id"] == "risk"


def test_finrag_system_can_archive_and_restore_knowledge_base(tmp_path):
    system = FinRAGSystem(RAGConfig(data_path=str(tmp_path / "documents"), knowledge_base_id="finance"))
    system.create_knowledge_base("risk")

    archived = system.archive_knowledge_base("risk")
    listed_archived = {
        item["knowledge_base_id"]: item
        for item in system.list_knowledge_bases()
    }
    restored = system.restore_knowledge_base("risk")

    assert archived["status"] == "archived"
    assert archived["archived_at"] is not None
    assert listed_archived["risk"]["status"] == "archived"
    assert restored["status"] == "active"
    assert restored["archived_at"] is None


def test_finrag_system_rejects_archive_delete_and_rebuild_for_protected_or_archived_kb(tmp_path):
    system = FinRAGSystem(RAGConfig(data_path=str(tmp_path / "documents"), knowledge_base_id="finance"))
    system.create_knowledge_base("risk")
    system.archive_knowledge_base("risk")

    with pytest.raises(ProtectedKnowledgeBaseError):
        system.archive_knowledge_base("finance")
    with pytest.raises(ProtectedKnowledgeBaseError):
        system.delete_knowledge_base("finance")
    with pytest.raises(KnowledgeBaseArchivedError):
        system.rebuild_from_sources("risk")


def test_finrag_system_delete_knowledge_base_clears_records_sources_and_runtime(tmp_path):
    data_path = tmp_path / "documents"
    source_dir = data_path / "risk"
    source_dir.mkdir(parents=True)
    source = source_dir / "policy.md"
    source.write_text("# policy", encoding="utf-8")
    system = FinRAGSystem(RAGConfig(data_path=str(data_path), knowledge_base_id="finance"))
    system.create_knowledge_base("risk")
    system.document_registry.records["doc-risk"] = DocumentRecord(
        document_id="doc-risk",
        source_path=str(source),
        filename="policy.md",
        file_type="md",
        content_hash="sha256:risk",
        knowledge_base_id="risk",
        status="indexed",
        chunk_count=1,
    )
    system.llama_docstore.add_documents(
        [
            TextNode(
                text="risk policy",
                id_="risk-leaf",
                metadata={
                    "knowledge_base_id": "risk",
                    "document_id": "doc-risk",
                    "chunk_level": 3,
                },
            )
        ]
    )
    system.bm25_store.replace_document_chunks("risk", "doc-risk", {"risk-leaf": {"risk": 1}})
    system.manifest_store.save_manifest({"knowledge_base_id": "risk"}, "risk")
    system.kb_runtimes["risk"] = SimpleNamespace(
        scope=system.knowledge_base_scope("risk"),
        data_module=object(),
        index_module=object(),
        generation_module=object(),
    )

    deleted = system.delete_knowledge_base("risk")

    assert deleted["status"] == "deleted"
    assert system.document_registry.records["doc-risk"].status == "deleted"
    assert not source_dir.exists()
    assert system.llama_docstore.load_all_nodes("risk") == []
    assert system.bm25_store.documents == {}
    assert system.manifest_store.load_manifest("risk") is None
    assert "risk" not in system.kb_runtimes
    assert [item["knowledge_base_id"] for item in system.list_knowledge_bases()] == ["finance"]
    with pytest.raises(KnowledgeBaseNotFoundError):
        system.restore_knowledge_base("risk")


def test_finrag_system_builds_scope_for_explicit_knowledge_base(tmp_path):
    system = FinRAGSystem(
        RAGConfig(
            data_path=str(tmp_path / "documents"),
            knowledge_base_id="finance",
            milvus_collection="finrag_leaf_nodes",
        )
    )

    scope = system.knowledge_base_scope("finance")

    assert isinstance(scope, KnowledgeBaseScope)
    assert scope.knowledge_base_id == "finance"
    assert scope.collection_name == "finrag_leaf_nodes__kb_finance"
