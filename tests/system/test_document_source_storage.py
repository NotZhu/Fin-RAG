from pathlib import Path

from finrag.application.system import FinRAGSystem
from finrag.core.config import RAGConfig
from tests.support.fakes import MemoryDocumentRegistry


def _registry():
    return MemoryDocumentRegistry("postgresql://test")


def _system(tmp_path):
    system = FinRAGSystem(
        RAGConfig(
            data_path=str(tmp_path / "data"),
            reranker_provider="none",
        )
    )
    system.document_registry = _registry()
    return system


def test_prepare_uploaded_file_stores_pending_source_with_original_filename(tmp_path):
    source = tmp_path / "incoming.md"
    source.write_text("# 制度\n客户尽调", encoding="utf-8")
    system = _system(tmp_path)

    prepared = system.prepare_uploaded_file(source, "../unsafe/policy.md", "finance")
    record = system.document_registry.get(prepared["document_id"])

    assert record.filename == "policy.md"
    assert Path(record.source_path) == Path(system.config.data_path) / ".pending" / "finance" / record.document_id / "policy.md"
    assert Path(record.source_path).read_text(encoding="utf-8") == "# 制度\n客户尽调"
    assert not (Path(system.config.data_path) / f"{record.document_id}.md").exists()


def test_prepare_uploaded_file_updates_source_path_without_full_registry_save(tmp_path):
    class RegistryWithoutFullSave(MemoryDocumentRegistry):
        def __init__(self):
            super().__init__("postgresql://test")
            self.updated_source_paths = []

        def save(self) -> None:
            raise AssertionError("prepare_uploaded_file should not full-save the document registry")

        def update_source_path(self, document_id: str, source_path: str) -> None:
            self.updated_source_paths.append((document_id, source_path))
            self.records[document_id].source_path = source_path

    source = tmp_path / "incoming.md"
    source.write_text("# 制度\n客户尽调", encoding="utf-8")
    system = _system(tmp_path)
    registry = RegistryWithoutFullSave()
    system.document_registry = registry

    prepared = system.prepare_uploaded_file(source, "policy.md", "finance")

    record = registry.get(prepared["document_id"])
    assert registry.updated_source_paths == [(record.document_id, record.source_path)]
    assert Path(record.source_path) == Path(system.config.data_path) / ".pending" / "finance" / record.document_id / "policy.md"


def test_promote_document_source_file_updates_source_path_without_full_registry_save(tmp_path):
    class RegistryWithoutFullSave(MemoryDocumentRegistry):
        def __init__(self):
            super().__init__("postgresql://test")
            self.updated_source_paths = []

        def save(self) -> None:
            raise AssertionError("promote_document_source_file should not full-save the document registry")

        def update_source_path(self, document_id: str, source_path: str) -> None:
            self.updated_source_paths.append((document_id, source_path))
            self.records[document_id].source_path = source_path

    system = _system(tmp_path)
    registry = RegistryWithoutFullSave()
    system.document_registry = registry
    pending_path = tmp_path / "pending" / "policy.md"
    pending_path.parent.mkdir(parents=True)
    pending_path.write_text("# 制度\n客户尽调", encoding="utf-8")
    record = registry.upsert_uploaded(
        source_path=pending_path,
        filename="policy.md",
        file_type="md",
        content_hash="sha256:policy",
        knowledge_base_id="finance",
    )

    system._managed_source_files().promote_document_source_file(record)

    assert registry.updated_source_paths == [(record.document_id, str(Path(system.config.data_path) / "finance" / "policy.md"))]
    assert Path(record.source_path).read_text(encoding="utf-8") == "# 制度\n客户尽调"


def test_index_registered_document_promotes_pending_source_to_original_filename(monkeypatch, tmp_path):
    source = tmp_path / "incoming.md"
    source.write_text("# 制度\n客户尽调", encoding="utf-8")
    system = _system(tmp_path)
    prepared = system.prepare_uploaded_file(source, "policy.md", "finance")
    document_id = prepared["document_id"]
    pending_path = Path(system.document_registry.get(document_id).source_path)

    def fake_index_document(document_id, *, retire_replacements):
        system.document_registry.mark_indexed(document_id, chunk_count=1)
        return system._public_document(document_id)

    monkeypatch.setattr(system, "_index_document_locked", fake_index_document)

    indexed = system.index_registered_document(document_id)
    record = system.document_registry.get(document_id)

    assert indexed["status"] == "indexed"
    assert Path(record.source_path) == Path(system.config.data_path) / "finance" / "policy.md"
    assert Path(record.source_path).read_text(encoding="utf-8") == "# 制度\n客户尽调"
    assert not pending_path.exists()