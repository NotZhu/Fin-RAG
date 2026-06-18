from llama_index.core import Document

from finrag.ingestion.parsers import ParserRegistry, compute_content_hash, load_documents
from tests.support.fakes import MemoryDocumentRegistry


def _registry():
    return MemoryDocumentRegistry("postgresql://test")


def test_parser_registry_uses_stable_metadata_without_forced_inference(tmp_path):
    source = tmp_path / "messy_001.md"
    source.write_text("第一条\n客户风险等级应与产品风险等级匹配", encoding="utf-8")

    parser = ParserRegistry.default()
    documents = parser.load(source, knowledge_base_id="kb-finance")

    assert len(documents) == 1
    doc = documents[0]
    assert isinstance(doc, Document)
    metadata = doc.metadata
    assert doc.text.startswith("第一条")
    assert metadata["knowledge_base_id"] == "kb-finance"
    assert metadata["filename"] == "messy_001.md"
    assert metadata["file_type"] == "md"
    assert metadata["page_number"] is None
    assert "title_hint" not in metadata
    assert "doc_title" not in metadata
    assert "regulatory_topic" not in metadata
    assert "department" not in metadata
    assert "source_name" not in metadata
    assert "page" not in metadata


def test_parser_registry_does_not_infer_markdown_heading_metadata(tmp_path):
    source = tmp_path / "risk_policy.md"
    source.write_text("# 适当性管理办法\n\n客户风险等级应与产品风险等级匹配", encoding="utf-8")

    parser = ParserRegistry.default()
    documents = parser.load(source, knowledge_base_id="kb-finance")

    metadata = documents[0].metadata
    assert metadata["knowledge_base_id"] == "kb-finance"
    assert "title_hint" not in metadata


def test_document_registry_tracks_lifecycle_and_deduplicates_by_hash(tmp_path):
    registry = _registry()
    first_hash = "sha256:first"

    first = registry.upsert_uploaded(
        source_path=tmp_path / "policy.txt",
        filename="policy.txt",
        file_type="txt",
        content_hash=first_hash,
        knowledge_base_id="kb-finance",
    )
    duplicate = registry.find_by_hash(first_hash, "kb-finance")
    registry.mark_indexed(first.document_id, chunk_count=3)

    assert duplicate is not None
    assert duplicate.document_id == first.document_id
    assert registry.get(first.document_id).status == "indexed"
    assert registry.get(first.document_id).chunk_count == 3
    assert registry.get(first.document_id).filename == "policy.txt"


def test_document_registry_keeps_same_filename_until_system_retires_old_document(tmp_path):
    registry = _registry()
    first_path = tmp_path / "policy-v1.txt"
    second_path = tmp_path / "policy-v2.txt"
    first_path.write_text("旧制度", encoding="utf-8")
    second_path.write_text("新制度", encoding="utf-8")

    first = registry.upsert_uploaded(
        source_path=first_path,
        filename="policy.txt",
        file_type="txt",
        content_hash="sha256:v1",
        knowledge_base_id="kb-finance",
    )
    second = registry.upsert_uploaded(
        source_path=second_path,
        filename="policy.txt",
        file_type="txt",
        content_hash="sha256:v2",
        knowledge_base_id="kb-finance",
    )

    assert second.document_id != first.document_id
    assert registry.get(first.document_id).status == "uploaded"
    assert registry.get(second.document_id).status == "uploaded"
    assert first_path.exists()


def test_load_documents_uses_registry_and_filters_current_knowledge_base_id(tmp_path):
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


def test_load_documents_accepts_registered_paths_inside_data_root(tmp_path):
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
