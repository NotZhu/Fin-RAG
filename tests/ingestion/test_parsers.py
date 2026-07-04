import inspect

from llama_index.core import Document

from finrag.ingestion.parsers import (
    SUPPORTED_SUFFIXES,
    compute_content_hash,
    load_documents,
)
from tests.support.fakes import MemoryDocumentRegistry


def _registry():
    return MemoryDocumentRegistry("postgresql://test")


class FakeDoclingReader:
    def __init__(self, documents):
        self.documents = documents
        self.loaded_paths = []

    def load_data(self, path):
        self.loaded_paths.append(str(path))
        return self.documents


class FailingDoclingReader:
    def load_data(self, path):
        raise RuntimeError(f"cannot parse {path}")


def test_parser_registry_is_removed_from_public_ingestion_api():
    import finrag.ingestion as ingestion
    import finrag.ingestion.parsers as parsers
    from finrag.ingestion.docling_loader import load_docling_documents

    assert not hasattr(ingestion, "ParserRegistry")
    assert not hasattr(parsers, "ParserRegistry")
    assert "parser_registry" not in inspect.signature(load_documents).parameters
    assert "reader" not in inspect.signature(load_docling_documents).parameters


def test_supported_suffixes_match_enterprise_rag_document_standard():
    enterprise_standard = {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".csv",
        ".md",
        ".txt",
        ".html",
        ".htm",
        ".json",
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
    }
    long_tail_formats = {
        ".dotx",
        ".docm",
        ".dotm",
        ".potx",
        ".ppsx",
        ".pptm",
        ".potm",
        ".ppsm",
        ".xlsm",
        ".text",
        ".qmd",
        ".rmd",
        ".xhtml",
        ".odt",
        ".ott",
        ".ods",
        ".ots",
        ".odp",
        ".otp",
        ".xml",
        ".nxml",
        ".xbrl",
        ".dclg",
        ".adoc",
        ".asciidoc",
        ".asc",
        ".bmp",
        ".webp",
        ".vtt",
        ".tex",
        ".latex",
        ".eml",
        ".epub",
        ".mp3",
        ".mp4",
        ".wav",
        ".avi",
        ".mov",
        ".zip",
        ".gz",
    }

    assert SUPPORTED_SUFFIXES == enterprise_standard
    assert not long_tail_formats & SUPPORTED_SUFFIXES


def test_load_docling_documents_converts_docling_json_and_minimal_metadata(tmp_path, monkeypatch):
    import finrag.ingestion.docling_loader as docling_loader

    source = tmp_path / "annual.pdf"
    source.write_bytes(b"%PDF mocked")
    reader = FakeDoclingReader(
        [
            Document(
                text='{"schema_name":"DoclingDocument","body":"营业收入 100"}',
                metadata={"headings": ["旧字段"], "doc_items": [{"label": "text"}]},
            )
        ]
    )
    monkeypatch.setattr(docling_loader, "_make_docling_reader", lambda: reader)

    documents = docling_loader.load_docling_documents(
        source,
        knowledge_base_id="kb-finance",
        data_root=tmp_path,
    )

    assert reader.loaded_paths == [str(source)]
    assert len(documents) == 1
    document = documents[0]
    assert isinstance(document, Document)
    assert "DoclingDocument" in document.text
    assert document.metadata["knowledge_base_id"] == "kb-finance"
    assert document.metadata["filename"] == "annual.pdf"
    assert document.metadata["file_type"] == "pdf"
    assert document.metadata["parser_name"] == "docling"
    for key in ("headings", "doc_items", "bbox", "origin", "element_type", "page_number", "section_title"):
        assert key not in document.metadata


def test_load_docling_documents_returns_empty_on_docling_failure_without_fallback(
    tmp_path,
    caplog,
    monkeypatch,
):
    import finrag.ingestion.docling_loader as docling_loader

    source = tmp_path / "broken.pdf"
    source.write_bytes(b"not really a pdf")
    monkeypatch.setattr(docling_loader, "_make_docling_reader", FailingDoclingReader)

    documents = docling_loader.load_docling_documents(
        source,
        knowledge_base_id="kb-risk",
        data_root=tmp_path,
    )

    assert documents == []
    assert "Docling 解析失败" in caplog.text


def test_load_documents_uses_docling_loader_for_supported_files(tmp_path, monkeypatch):
    import finrag.ingestion.parsers as parsers

    (tmp_path / "policy.md").write_text("# 制度\n客户尽调", encoding="utf-8")
    (tmp_path / "annual.pdf").write_bytes(b"%PDF mocked")
    (tmp_path / "board_deck.pptx").write_bytes(b"pptx mocked")
    (tmp_path / "ledger.xlsx").write_bytes(b"xlsx mocked")
    (tmp_path / "scan.png").write_bytes(b"ocr image")
    (tmp_path / "legacy.odt").write_bytes(b"unsupported long tail")
    (tmp_path / "voice.mp3").write_bytes(b"unsupported audio")
    seen_paths = []

    def fake_loader(path, *, knowledge_base_id, data_root=None):
        seen_paths.append(path.name)
        return [
            Document(
                text=f"# {path.name}",
                metadata=parsers.build_base_metadata(
                    path,
                    path.name,
                    path.suffix.lower().lstrip("."),
                    knowledge_base_id=knowledge_base_id,
                    data_root=data_root,
                )
                | {"parser_name": "docling"},
            )
        ]

    monkeypatch.setattr(parsers, "load_docling_documents", fake_loader)

    documents = load_documents(tmp_path, knowledge_base_id="kb-finance")

    assert seen_paths == ["annual.pdf", "board_deck.pptx", "ledger.xlsx", "policy.md", "scan.png"]
    assert [doc.metadata["filename"] for doc in documents] == [
        "annual.pdf",
        "board_deck.pptx",
        "ledger.xlsx",
        "policy.md",
        "scan.png",
    ]
    assert {doc.metadata["knowledge_base_id"] for doc in documents} == {"kb-finance"}
    assert {doc.metadata["parser_name"] for doc in documents} == {"docling"}


def test_load_documents_uses_registry_and_filters_current_knowledge_base_id(tmp_path, monkeypatch):
    import finrag.ingestion.parsers as parsers

    kb_a = tmp_path / "kb-a-policy.txt"
    kb_b = tmp_path / "kb-b-policy.txt"
    deleted = tmp_path / "deleted.txt"
    kb_a.write_text("A 资料库 客户尽调", encoding="utf-8")
    kb_b.write_text("B 资料库 理财适当性", encoding="utf-8")
    deleted.write_text("已删除 旧制度", encoding="utf-8")
    registry = _registry()
    rec_a = registry.upsert_uploaded(
        source_path=kb_a,
        filename="a.txt",
        file_type="txt",
        content_hash=compute_content_hash(kb_a),
        knowledge_base_id="kb-a",
    )
    rec_b = registry.upsert_uploaded(
        source_path=kb_b,
        filename="b.txt",
        file_type="txt",
        content_hash=compute_content_hash(kb_b),
        knowledge_base_id="kb-b",
    )
    rec_deleted = registry.upsert_uploaded(
        source_path=deleted,
        filename="deleted.txt",
        file_type="txt",
        content_hash=compute_content_hash(deleted),
        knowledge_base_id="kb-a",
    )
    registry.mark_indexed(rec_a.document_id, chunk_count=1)
    registry.mark_indexed(rec_b.document_id, chunk_count=1)
    registry.mark_deleted(rec_deleted.document_id)

    def fake_loader(path, *, knowledge_base_id, data_root=None):
        return [
            Document(
                text=path.read_text(encoding="utf-8"),
                metadata=parsers.build_base_metadata(
                    path,
                    path.name,
                    path.suffix.lower().lstrip("."),
                    knowledge_base_id=knowledge_base_id,
                    data_root=data_root,
                ),
            )
        ]

    monkeypatch.setattr(parsers, "load_docling_documents", fake_loader)

    kb_a_documents = load_documents(tmp_path, knowledge_base_id="kb-a", document_registry=registry)
    kb_b_documents = load_documents(tmp_path, knowledge_base_id="kb-b", document_registry=registry)

    assert {doc.metadata["knowledge_base_id"] for doc in kb_a_documents} == {"kb-a"}
    assert {doc.metadata["filename"] for doc in kb_a_documents} == {"a.txt"}
    assert {doc.metadata["knowledge_base_id"] for doc in kb_b_documents} == {"kb-b"}
    assert {doc.metadata["filename"] for doc in kb_b_documents} == {"b.txt"}


def test_load_documents_skips_registry_source_paths_outside_data_root(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    external = tmp_path / "external.md"
    external.write_text("# 外部文件\n不应被索引", encoding="utf-8")
    registry = _registry()
    registry.upsert_uploaded(
        source_path=external,
        filename="external.md",
        file_type="md",
        content_hash=compute_content_hash(external),
        knowledge_base_id="kb-finance",
    )

    documents = load_documents(data_root, knowledge_base_id="kb-finance", document_registry=registry)

    assert documents == []


def test_load_documents_accepts_registered_paths_inside_data_root(tmp_path, monkeypatch):
    import finrag.ingestion.parsers as parsers

    data_root = tmp_path / "data"
    data_root.mkdir()
    source = data_root / "policy.md"
    source.write_text("# 制度\n客户尽调", encoding="utf-8")
    registry = _registry()
    record = registry.upsert_uploaded(
        source_path=source,
        filename="policy.md",
        file_type="md",
        content_hash=compute_content_hash(source),
        knowledge_base_id="kb-finance",
    )
    registry.mark_indexed(record.document_id, chunk_count=1)

    def fake_loader(path, *, knowledge_base_id, data_root=None):
        return [
            Document(
                text="# 制度\n客户尽调",
                metadata=parsers.build_base_metadata(
                    path,
                    "# 制度\n客户尽调",
                    "md",
                    knowledge_base_id=knowledge_base_id,
                    data_root=data_root,
                ),
            )
        ]

    monkeypatch.setattr(parsers, "load_docling_documents", fake_loader)

    documents = load_documents(data_root, knowledge_base_id="kb-finance", document_registry=registry)

    assert len(documents) == 1
    assert documents[0].metadata["document_id"] == record.document_id
    assert documents[0].metadata["filename"] == "policy.md"


def test_document_registry_public_list_hides_internal_paths_and_hashes(tmp_path):
    source = tmp_path / "policy.txt"
    source.write_text("制度内容", encoding="utf-8")
    registry = _registry()
    record = registry.upsert_uploaded(
        source_path=source,
        filename="policy.txt",
        file_type="txt",
        content_hash=compute_content_hash(source),
        knowledge_base_id="kb-finance",
    )
    registry.mark_indexed(record.document_id, chunk_count=1)

    public_record = registry.list_public()[0]

    assert "source_path" not in public_record
    assert "content_hash" not in public_record
    assert public_record["filename"] == "policy.txt"
    assert public_record["status"] == "indexed"
