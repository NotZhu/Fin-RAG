from types import SimpleNamespace

from finrag.application.system import FinRAGSystem
from finrag.core.config import RAGConfig
from finrag.ingestion import DocumentRecord


def test_ready_counts_registry_documents_when_nodes_are_restored_from_store(tmp_path):
    system = FinRAGSystem(RAGConfig(data_path=str(tmp_path)))
    system.knowledge_query_engine = object()
    system.data_module = SimpleNamespace(
        get_statistics=lambda: {
            "total_documents": 0,
            "total_chunks": 26,
            "file_types": {},
            "avg_chunk_size": 545.0,
        }
    )
    system.document_registry.records = {
        "doc-a": DocumentRecord(
            document_id="doc-a",
            source_path=str(tmp_path / "a.md"),
            filename="a.md",
            file_type="md",
            content_hash="sha256:a",
            knowledge_base_id="kb-finance",
            status="indexed",
            chunk_count=3,
        ),
        "doc-b": DocumentRecord(
            document_id="doc-b",
            source_path=str(tmp_path / "b.txt"),
            filename="b.txt",
            file_type="txt",
            content_hash="sha256:b",
            knowledge_base_id="kb-finance",
            status="indexed",
            chunk_count=2,
        ),
    }

    stats = system.get_statistics()
    ready = system.ready()

    assert stats == {
        "total_documents": 2,
        "total_chunks": 26,
        "file_types": {"md": 1, "txt": 1},
        "avg_chunk_size": 545.0,
    }
    assert ready == {
        "ready": True,
        "status": "ready",
        "total_documents": 2,
        "total_chunks": 26,
        "last_error": None,
    }
