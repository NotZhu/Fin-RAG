from pathlib import Path
import tomllib


def test_cleanup_contract_removes_local_storage_modules_from_source_tree():
    project_root = Path(__file__).resolve().parents[2]

    assert not (project_root / "src" / "finrag" / "indexing" / "node_store.py").exists()
    assert not (project_root / "src" / "finrag" / "ingestion" / "document_ingestion.py").exists()
    assert "class DocumentRegistry" not in (project_root / "src" / "finrag" / "ingestion" / "parsers.py").read_text(encoding="utf-8")


def test_project_text_no_longer_mentions_chroma_or_vector_gateway():
    project_root = Path(__file__).resolve().parents[2]
    searchable = [
        *project_root.joinpath("src").rglob("*.py"),
        project_root / "README.md",
        project_root / ".env.example",
    ]

    combined = "\n".join(path.read_text(encoding="utf-8") for path in searchable if path.exists())

    assert "Chroma" not in combined
    assert "chromadb" not in combined
    assert "VectorStoreGateway" not in combined
    assert "RAG_VECTOR_BACKEND" not in combined
    assert "RAG_STORAGE_BACKEND" not in combined


def test_runtime_dependencies_include_postgres_client_only():
    project_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = "\n".join(pyproject["project"]["dependencies"])

    assert "psycopg" in dependencies
    assert "redis" not in dependencies


def test_integration_test_entrypoint_exists():
    project_root = Path(__file__).resolve().parents[2]
    integration_dir = project_root / "tests" / "integration"

    assert integration_dir.is_dir()
    assert any(integration_dir.glob("test_*.py"))
