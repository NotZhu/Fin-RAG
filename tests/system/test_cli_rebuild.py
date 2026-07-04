import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from llama_index.core import Document
from llama_index.core.schema import TextNode

from finrag.core.config import PROJECT_ROOT, RAGConfig
from finrag.ingestion import DocumentRecord
from finrag.application.system import FinRAGSystem
from finrag import cli

requires_live_vector_stack = pytest.mark.skipif(
    os.getenv("FINRAG_RUN_INTEGRATION") != "1",
    reason="需要运行中的 Milvus 和 DashScope embedding",
)


def test_cli_rebuild_invokes_source_rebuild(monkeypatch, capsys):
    calls = []

    class FakeSystem:
        def rebuild_from_sources(self, knowledge_base_id):
            calls.append(knowledge_base_id)
            return {"document_count": 2, "chunk_count": 5, "manifest_schema_version": 9}

    monkeypatch.setattr(cli, "FinRAGSystem", FakeSystem)

    exit_code = cli.main(["rebuild", "--knowledge-base-id", "finance"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert calls == ["finance"]
    assert "document_count=2" in output
    assert "manifest_schema_version=9" in output


def test_cli_rebuild_requires_knowledge_base_id():
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["rebuild"])

    assert exc_info.value.code == 2


def test_cli_rebuild_accepts_kb_alias(monkeypatch):
    calls = []

    class FakeSystem:
        def rebuild_from_sources(self, knowledge_base_id):
            calls.append(knowledge_base_id)
            return {"document_count": 0, "chunk_count": 0, "manifest_schema_version": 1}

    monkeypatch.setattr(cli, "FinRAGSystem", FakeSystem)

    exit_code = cli.main(["rebuild", "--kb", "risk"])

    assert exit_code == 0
    assert calls == ["risk"]


def test_cli_rebuild_rejects_invalid_knowledge_base_id():
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["rebuild", "--knowledge-base-id", "bad!"])

    assert exc_info.value.code == 2


def test_cli_import_does_not_require_project_root_on_pythonpath(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")

    result = subprocess.run(
        [sys.executable, "-c", "import finrag.cli; print('ok')"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_cli_eval_invokes_ragas_evaluation(monkeypatch, tmp_path, capsys):
    eval_path = tmp_path / "eval.jsonl"
    report_path = tmp_path / "report.json"
    calls = {}

    def fake_run(args):
        calls["args"] = args
        return SimpleNamespace(provider="ragas", rows=[{"question": "q"}])

    monkeypatch.setattr(cli, "run_evaluation_report", fake_run)

    exit_code = cli.main(["eval", "--dataset", str(eval_path), "--output", str(report_path)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert calls["args"].dataset == eval_path
    assert calls["args"].output == report_path
    assert "provider=ragas" in output


def test_rebuild_from_sources_ignores_existing_registry_records(monkeypatch, tmp_path):
    import finrag.indexing.nodes as nodes_module

    source = tmp_path / "finance" / "source.md"
    source.parent.mkdir(parents=True)
    source.write_text("# 新源文件\n客户风险等级应与产品风险等级匹配", encoding="utf-8")
    system = FinRAGSystem(RAGConfig(data_path=str(tmp_path)))
    system.document_registry.records = {
        "old-doc": DocumentRecord(
            document_id="old-doc",
            source_path=str(tmp_path / "missing.md"),
            filename="missing.md",
            file_type="md",
            content_hash="sha256:old",
            knowledge_base_id="finance",
            status="indexed",
            chunk_count=1,
            last_error="旧错误",
        )
    }

    class RecordingDataModule:
        def __init__(self):
            self.document_registry = system.document_registry
            self.documents = []
            self.chunks = []
            self.all_nodes = []
            self.storage_context = None
            self.registry_seen = "not-called"
            self.existing_record_seen = None

        def load_documents(self):
            self.registry_seen = self.document_registry
            self.existing_record_seen = system.document_registry.records["old-doc"].to_dict()
            if self.document_registry is None:
                self.documents = [
                    Document(
                        text=source.read_text(encoding="utf-8"),
                        metadata={
                            "document_id": "source-doc",
                            "source_path": str(source),
                            "filename": "source.md",
                            "file_type": "md",
                            "knowledge_base_id": "finance",
                        },
                    )
                ]
            else:
                self.documents = []
            return self.documents

    class FakeIndexModule:
        def __init__(self):
            self.manifest = {"schema_version": 9}
            self.embed_model = object()

        def init_collection(self, *, reset=False):
            return object()

        def clear_index(self, *, storage_context=None):
            return object()

        def build_vector_index(self, nodes, *, storage_context=None, reset=True):
            return object()

        def load_index(self, expected_manifest=None, *, storage_context=None):
            return object()

        def save_manifest(self, manifest):
            self.manifest = dict(manifest)

        def load_manifest(self, knowledge_base_id):
            return self.manifest

        def build_manifest(self, **kwargs):
            return {"schema_version": 9, "node_structure": "docling", **kwargs}

    class FakePipeline:
        def run(self, *, documents, show_progress=False):
            assert len(documents) == 1
            return [
                TextNode(
                    text="客户风险等级应与产品风险等级匹配",
                    id_="source-doc-leaf",
                    metadata={
                        "document_id": "source-doc",
                        "source_path": str(source),
                        "filename": "source.md",
                        "file_type": "md",
                        "knowledge_base_id": "finance",
                        "chunk_id": "source-doc-leaf",
                        "chunk_level": 3,
                        "chunk_idx": 0,
                    },
                )
            ]

    def build_fake_pipeline(*args, **kwargs):
        return FakePipeline()

    monkeypatch.setattr(nodes_module, "build_ingestion_pipeline", build_fake_pipeline)

    data_module = RecordingDataModule()
    system.data_module = data_module
    system.index_module = FakeIndexModule()
    system.llama_docstore = SimpleNamespace(delete_knowledge_base=lambda knowledge_base_id: None, add_documents=lambda nodes: None)
    system.generation_module = SimpleNamespace()
    system._refresh_retrieval = lambda vector_index, leaf_nodes, knowledge_base_id: None

    result = system.rebuild_from_sources("finance")

    assert data_module.registry_seen is None
    assert data_module.existing_record_seen["status"] == "parsing"
    assert data_module.existing_record_seen["chunk_count"] == 0
    assert data_module.existing_record_seen["last_error"] is None
    assert result["document_count"] == 1
    assert system.document_registry.list_public()[0]["filename"] == "source.md"
    assert "old-doc" not in system.document_registry.records


def test_rebuild_marks_source_documents_parsing_and_touches_knowledge_base(monkeypatch, tmp_path):
    import finrag.indexing.nodes as nodes_module
    import tests.support.fakes as fakes

    counter = {"value": 0}

    def next_time():
        counter["value"] += 1
        return f"2026-06-22T16:{counter['value']:02d}:00+00:00"

    monkeypatch.setattr(fakes, "_utc_now_iso", next_time)

    source = tmp_path / "finance" / "source.md"
    source.parent.mkdir(parents=True)
    source.write_text("# 新源文件\n客户风险等级应与产品风险等级匹配", encoding="utf-8")
    system = FinRAGSystem(RAGConfig(data_path=str(tmp_path)))
    initial_updated_at = system.knowledge_base_registry.get("finance").updated_at
    observed = {}

    class FakeIndexModule:
        def __init__(self):
            self.manifest = {"schema_version": 9}
            self.embed_model = object()

        def init_collection(self, *, reset=False):
            return object()

        def build_vector_index(self, nodes, *, storage_context=None, reset=True):
            return object()

        def load_index(self, expected_manifest=None, *, storage_context=None):
            return object()

        def save_manifest(self, manifest):
            self.manifest = dict(manifest)

        def load_manifest(self, knowledge_base_id):
            return self.manifest

        def build_manifest(self, **kwargs):
            return {"schema_version": 9, "node_structure": "docling", **kwargs}

    class RecordingDataModule:
        def __init__(self):
            self.document_registry = system.document_registry
            self.documents = []
            self.chunks = []
            self.all_nodes = []
            self.storage_context = None
            self.data_path = str(tmp_path)

        def load_documents(self):
            if self.document_registry is None:
                self.documents = [
                    Document(
                        text=source.read_text(encoding="utf-8"),
                        metadata={
                            "document_id": "source-doc",
                            "source_path": str(source),
                            "filename": "source.md",
                            "file_type": "md",
                            "knowledge_base_id": "finance",
                        },
                    )
                ]
            else:
                self.documents = []
            return self.documents

    class FakePipeline:
        def run(self, *, documents, show_progress=False):
            observed["during_rebuild_docs"] = system.document_registry.list_public("finance")
            observed["during_rebuild_updated_at"] = system.knowledge_base_registry.get("finance").updated_at
            return [
                TextNode(
                    text="客户风险等级应与产品风险等级匹配",
                    id_="source-doc-leaf",
                    metadata={
                        "document_id": "source-doc",
                        "source_path": str(source),
                        "filename": "source.md",
                        "file_type": "md",
                        "knowledge_base_id": "finance",
                        "chunk_id": "source-doc-leaf",
                        "chunk_level": 3,
                        "chunk_idx": 0,
                    },
                )
            ]

    monkeypatch.setattr(nodes_module, "build_ingestion_pipeline", lambda *args, **kwargs: FakePipeline())

    system.data_module = RecordingDataModule()
    system.index_module = FakeIndexModule()
    system.llama_docstore = SimpleNamespace(delete_knowledge_base=lambda knowledge_base_id: None, add_documents=lambda nodes: None)
    system.generation_module = SimpleNamespace()
    system._refresh_retrieval = lambda vector_index, leaf_nodes, knowledge_base_id: None

    system.rebuild_from_sources("finance")

    assert observed["during_rebuild_docs"][0]["status"] == "parsing"
    assert observed["during_rebuild_updated_at"] != initial_updated_at
    assert system.knowledge_base_registry.get("finance").updated_at != observed["during_rebuild_updated_at"]


@requires_live_vector_stack
def test_rebuild_from_sources_replaces_document_registry_from_source_files(tmp_path):
    source = tmp_path / "finance" / "policy.md"
    source.parent.mkdir(parents=True)
    source.write_text("# 制度\n客户风险等级应与产品风险等级匹配", encoding="utf-8")
    system = FinRAGSystem(RAGConfig(data_path=str(tmp_path)))

    result = system.rebuild_from_sources("finance")

    documents = system.document_registry.list()
    assert result["document_count"] == 1
    assert result["chunk_count"] > 0
    assert documents[0]["filename"] == "policy.md"
    assert documents[0]["status"] == "indexed"
    assert documents[0]["chunk_count"] == result["chunk_count"]
