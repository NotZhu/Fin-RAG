from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from finrag.api import create_app
from finrag.core.response_schema import FinRAGResponse, RetrievedSource
from finrag.storage.knowledge_base_registry import KnowledgeBaseArchivedError


def parse_sse_events(body: str):
    events = []
    for block in body.strip().split("\n\n"):
        if not block:
            continue
        event_name = "message"
        data = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        events.append((event_name, data))
    return events


class FakeFinRAGSystem:
    def __init__(self):
        self.documents = []
        self.config = SimpleNamespace(
            knowledge_base_id="finance",
            upload_dir="storage/uploads",
            max_upload_bytes=20 * 1024 * 1024,
        )
        self.data_module = SimpleNamespace(get_statistics=lambda: {"total_documents": len(self.documents), "total_chunks": 0})

    def initialize_system(self):
        return None

    def build_knowledge_base(self):
        return None

    def ensure_knowledge_base_ready(self, knowledge_base_id=None):
        return None

    def ready(self):
        return {"ready": True, "status": "ready", "total_documents": len(self.documents), "total_chunks": 0, "last_error": None}

    def list_documents(self, knowledge_base_id: str):
        return [
            document
            for document in self.documents
            if document["knowledge_base_id"] == knowledge_base_id
        ]

    def ingest_uploaded_file(self, file_path: Path, filename: str, knowledge_base_id: str):
        record = {
            "document_id": "doc-1",
            "filename": filename,
            "file_type": file_path.suffix.lstrip("."),
            "knowledge_base_id": knowledge_base_id,
            "status": "indexed",
            "chunk_count": 1,
            "last_error": None,
        }
        self.documents.append(record)
        return record

    def prepare_uploaded_file(self, file_path: Path, filename: str, knowledge_base_id: str):
        record = {
            "document_id": "doc-async",
            "filename": filename,
            "file_type": file_path.suffix.lstrip("."),
            "knowledge_base_id": knowledge_base_id,
            "status": "uploaded",
            "chunk_count": 0,
            "last_error": None,
        }
        self.documents.append(record)
        return record

    def index_registered_document(self, document_id: str):
        for document in self.documents:
            if document["document_id"] == document_id:
                document["status"] = "indexed"
                document["chunk_count"] = 1
                return document
        return {"document_id": document_id, "status": "failed"}

    def delete_document(self, document_id: str, knowledge_base_id: str):
        for document in self.documents:
            if document["document_id"] != document_id:
                continue
            if document["knowledge_base_id"] != knowledge_base_id:
                raise KeyError(document_id)
            self.documents = [doc for doc in self.documents if doc["document_id"] != document_id]
            return {"document_id": document_id, "status": "deleted"}
        raise KeyError(document_id)

    def reindex_document(self, document_id: str, knowledge_base_id: str):
        for document in self.documents:
            if document["document_id"] != document_id:
                continue
            if document["knowledge_base_id"] != knowledge_base_id:
                raise KeyError(document_id)
            return {"document_id": document_id, "status": "indexed", "knowledge_base_id": document["knowledge_base_id"]}
        raise KeyError(document_id)

    def ask_question(self, question: str, **kwargs):
        assert "stream" not in kwargs
        return FinRAGResponse(
            question=question,
            answer="资料显示客户风险等级应与产品风险等级匹配[1]",
            sources=[
                RetrievedSource(
                    source_id=1,
                    filename="适当性管理办法.md",
                    file_type="md",
                    page_number=None,
                    chunk_id="node-1",
                    parent_chunk_id="parent-1",
                    root_chunk_id="root-1",
                    chunk_level=3,
                    chunk_idx=0,
                    score=0.9,
                    snippet="客户风险等级应与产品风险等级匹配",
                )
            ],
            trace={"retrieval_strategy": "llamaindex_router", "route_type": "knowledge", "events": []},
            retrieval_strategy="llamaindex_router",
            route_type="knowledge",
        )


class FakeMissingDocumentSystem(FakeFinRAGSystem):
    def delete_document(self, document_id: str, knowledge_base_id: str):
        raise KeyError(document_id)

    def reindex_document(self, document_id: str, knowledge_base_id: str):
        raise KeyError(document_id)


class FakeFailingIngestSystem(FakeFinRAGSystem):
    def ingest_uploaded_file(self, file_path: Path, filename: str, knowledge_base_id: str):
        raise RuntimeError("索引失败")


class FakeArchivedKnowledgeBaseSystem(FakeFinRAGSystem):
    def ingest_uploaded_file(self, file_path: Path, filename: str, knowledge_base_id: str):
        raise KnowledgeBaseArchivedError(knowledge_base_id)

    def prepare_uploaded_file(self, file_path: Path, filename: str, knowledge_base_id: str):
        raise KnowledgeBaseArchivedError(knowledge_base_id)

    def delete_document(self, document_id: str, knowledge_base_id: str):
        raise KnowledgeBaseArchivedError(knowledge_base_id)

    def reindex_document(self, document_id: str, knowledge_base_id: str):
        raise KnowledgeBaseArchivedError(knowledge_base_id)


def test_removed_document_lifecycle_routes_are_not_registered(tmp_path):
    app = create_app(system_factory=FakeFinRAGSystem, upload_dir=tmp_path)
    client = TestClient(app)

    responses = [
        client.get("/documents"),
        client.post(
            "/documents/upload",
            files={"file": ("policy.md", b"# policy\ncontent", "text/markdown")},
        ),
        client.delete("/documents/doc-1"),
        client.post("/documents/doc-1/reindex"),
    ]

    assert [response.status_code for response in responses] == [404, 404, 404, 404]
    assert [response.json() for response in responses] == [
        {"detail": "Not Found"},
        {"detail": "Not Found"},
        {"detail": "Not Found"},
        {"detail": "Not Found"},
    ]


def test_document_upload_list_delete_and_ask_api(tmp_path):
    app = create_app(system_factory=FakeFinRAGSystem, upload_dir=tmp_path)
    client = TestClient(app)

    upload = client.post(
        "/knowledge-bases/finance/documents/upload",
        files={"file": ("policy.md", b"# policy\ncontent", "text/markdown")},
    )
    assert upload.status_code == 200
    assert upload.json()["status"] == "indexed"

    listed = client.get("/knowledge-bases/finance/documents")
    assert listed.status_code == 200
    assert listed.json()["documents"][0]["filename"] == "policy.md"

    with client.stream(
        "POST",
        "/knowledge-bases/kb-finance/ask",
        json={
            "question": "客户风险等级如何匹配？",
            "return_sources": True,
            "return_trace": True,
        },
    ) as answer:
        body = "".join(answer.iter_text())
    assert answer.status_code == 200
    assert answer.headers["content-type"].startswith("text/event-stream")
    event_names = [name for name, _ in parse_sse_events(body)]
    assert event_names == ["done"]
    assert "资料显示客户风险等级应与产品风险等级匹配" in body
    assert "适当性管理办法.md" in body

    deleted = client.delete("/knowledge-bases/finance/documents/doc-1")
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"


def test_document_routes_filter_by_knowledge_base_id(tmp_path):
    app = create_app(system_factory=FakeFinRAGSystem, upload_dir=tmp_path)
    client = TestClient(app)

    finance_upload = client.post(
        "/knowledge-bases/finance/documents/upload",
        files={"file": ("finance.md", b"# finance\ncontent", "text/markdown")},
    )
    risk_upload = client.post(
        "/knowledge-bases/risk/documents/upload",
        files={"file": ("risk.md", b"# risk\ncontent", "text/markdown")},
    )

    assert finance_upload.status_code == 200
    assert risk_upload.status_code == 200

    scoped_finance = client.get("/knowledge-bases/finance/documents")
    scoped_risk = client.get("/knowledge-bases/risk/documents")

    assert [doc["filename"] for doc in scoped_finance.json()["documents"]] == ["finance.md"]
    assert [doc["filename"] for doc in scoped_risk.json()["documents"]] == ["risk.md"]


def test_scoped_document_delete_and_reindex_return_404_for_other_knowledge_base(tmp_path):
    app = create_app(system_factory=FakeFinRAGSystem, upload_dir=tmp_path)
    client = TestClient(app)

    upload = client.post(
        "/knowledge-bases/finance/documents/upload",
        files={"file": ("policy.md", b"# policy\ncontent", "text/markdown")},
    )
    document_id = upload.json()["document_id"]

    bad_delete = client.delete(f"/knowledge-bases/risk/documents/{document_id}")
    bad_reindex = client.post(f"/knowledge-bases/risk/documents/{document_id}/reindex")
    good_reindex = client.post(f"/knowledge-bases/finance/documents/{document_id}/reindex")

    assert bad_delete.status_code == 404
    assert bad_delete.json()["error"]["code"] == "document_not_found"
    assert bad_reindex.status_code == 404
    assert bad_reindex.json()["error"]["code"] == "document_not_found"
    assert good_reindex.status_code == 200
    assert good_reindex.json()["knowledge_base_id"] == "finance"


def test_upload_removes_temporary_file_after_sync_index(tmp_path):
    app = create_app(system_factory=FakeFinRAGSystem, upload_dir=tmp_path)
    client = TestClient(app)

    upload = client.post(
        "/knowledge-bases/kb-finance/documents/upload",
        files={"file": ("policy.md", b"# policy\ncontent", "text/markdown")},
    )

    assert upload.status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_scoped_upload_uses_path_knowledge_base_id(tmp_path):
    app = create_app(system_factory=FakeFinRAGSystem, upload_dir=tmp_path)
    client = TestClient(app)

    upload = client.post(
        "/knowledge-bases/finance/documents/upload",
        files={"file": ("policy.md", b"# policy\ncontent", "text/markdown")},
    )

    assert upload.status_code == 200
    assert upload.json()["knowledge_base_id"] == "finance"


def test_upload_max_size_uses_system_config(tmp_path):
    class TinyUploadLimitSystem(FakeFinRAGSystem):
        def __init__(self):
            super().__init__()
            self.config.max_upload_bytes = 5

    app = create_app(system_factory=TinyUploadLimitSystem, upload_dir=tmp_path)
    client = TestClient(app)

    upload = client.post(
        "/knowledge-bases/kb-finance/documents/upload",
        files={"file": ("policy.md", b"123456", "text/markdown")},
    )

    assert upload.status_code == 400
    assert upload.json()["error"]["code"] == "file_too_large"
    assert list(tmp_path.iterdir()) == []


def test_async_upload_removes_temporary_file_after_prepare(tmp_path):
    app = create_app(system_factory=FakeFinRAGSystem, upload_dir=tmp_path)
    client = TestClient(app)

    upload = client.post(
        "/knowledge-bases/kb-finance/documents/upload",
        data={"async_index": "true"},
        files={"file": ("policy.md", b"# policy\ncontent", "text/markdown")},
    )

    assert upload.status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_upload_removes_temporary_file_when_indexing_fails(tmp_path):
    app = create_app(system_factory=FakeFailingIngestSystem, upload_dir=tmp_path)
    client = TestClient(app)

    upload = client.post(
        "/knowledge-bases/kb-finance/documents/upload",
        files={"file": ("policy.md", b"# policy\ncontent", "text/markdown")},
    )

    assert upload.status_code == 500
    assert list(tmp_path.iterdir()) == []


def test_upload_uses_basename_for_client_supplied_filename_path(tmp_path):
    app = create_app(system_factory=FakeFinRAGSystem, upload_dir=tmp_path)
    client = TestClient(app)

    upload = client.post(
        "/knowledge-bases/kb-finance/documents/upload",
        files={"file": ("../secret/policy.md", b"# policy\ncontent", "text/markdown")},
    )

    assert upload.status_code == 200
    assert upload.json()["filename"] == "policy.md"


def test_delete_and_reindex_missing_document_return_404(tmp_path):
    app = create_app(system_factory=FakeMissingDocumentSystem, upload_dir=tmp_path)
    client = TestClient(app)

    deleted = client.delete("/knowledge-bases/finance/documents/missing-doc")
    reindexed = client.post("/knowledge-bases/finance/documents/missing-doc/reindex")

    assert deleted.status_code == 404
    assert deleted.json()["error"]["code"] == "document_not_found"
    assert reindexed.status_code == 404
    assert reindexed.json()["error"]["code"] == "document_not_found"


def test_ask_ignores_removed_retrieval_strategy_field(tmp_path):
    app = create_app(system_factory=FakeFinRAGSystem, upload_dir=tmp_path)
    client = TestClient(app)

    with client.stream(
        "POST",
        "/knowledge-bases/finance/ask",
        json={
            "question": "客户风险等级如何匹配？",
            "retrieval_strategy": "unknown",
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "llamaindex_router" in body


def test_upload_rejects_unsupported_file_type(tmp_path):
    app = create_app(system_factory=FakeFinRAGSystem, upload_dir=tmp_path)
    client = TestClient(app)

    upload = client.post(
        "/knowledge-bases/kb-finance/documents/upload",
        files={"file": ("malware.exe", b"not a document", "application/octet-stream")},
    )

    assert upload.status_code == 400
    assert upload.json()["error"]["code"] == "unsupported_file_type"


def test_upload_rejects_invalid_knowledge_base_id_before_storing_file(tmp_path):
    app = create_app(system_factory=FakeFinRAGSystem, upload_dir=tmp_path)
    client = TestClient(app)

    upload = client.post(
        "/knowledge-bases/bad!/documents/upload",
        files={"file": ("policy.md", b"# policy\ncontent", "text/markdown")},
    )

    assert upload.status_code == 400
    assert upload.json()["error"]["code"] == "invalid_knowledge_base_id"
    assert list(tmp_path.iterdir()) == []


def test_archived_knowledge_base_document_mutations_return_conflict(tmp_path):
    app = create_app(system_factory=FakeArchivedKnowledgeBaseSystem, upload_dir=tmp_path)
    client = TestClient(app)

    upload = client.post(
        "/knowledge-bases/risk/documents/upload",
        files={"file": ("policy.md", b"# policy\ncontent", "text/markdown")},
    )
    deleted = client.delete("/knowledge-bases/risk/documents/doc-1")
    reindexed = client.post("/knowledge-bases/risk/documents/doc-1/reindex")

    assert upload.status_code == 409
    assert upload.json()["error"]["code"] == "knowledge_base_archived"
    assert deleted.status_code == 409
    assert deleted.json()["error"]["code"] == "knowledge_base_archived"
    assert reindexed.status_code == 409
    assert reindexed.json()["error"]["code"] == "knowledge_base_archived"


def test_upload_accepts_supported_file_and_rejects_unsupported_suffix(tmp_path):
    app = create_app(system_factory=FakeFinRAGSystem, upload_dir=tmp_path)
    client = TestClient(app)

    upload = client.post(
        "/knowledge-bases/kb-finance/documents/upload",
        files={"file": ("policy.md", b"# policy\ncontent", "text/markdown")},
    )
    bad = client.post(
        "/knowledge-bases/kb-finance/documents/upload",
        files={"file": ("malware.exe", b"nope", "application/octet-stream")},
    )

    assert upload.status_code == 200
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "unsupported_file_type"


def test_async_upload_returns_real_document_id(tmp_path):
    app = create_app(system_factory=FakeFinRAGSystem, upload_dir=tmp_path)
    client = TestClient(app)

    upload = client.post(
        "/knowledge-bases/kb-finance/documents/upload",
        data={"async_index": "true"},
        files={"file": ("policy.txt", b"content", "text/plain")},
    )

    assert upload.status_code == 200
    assert upload.json()["document_id"] == "doc-async"
    assert upload.json()["status"] in {"uploaded", "parsing", "indexed"}
