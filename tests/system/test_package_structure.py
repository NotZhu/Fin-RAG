from pathlib import Path
import importlib.util


def test_public_package_imports_are_available():
    from finrag.api import create_app
    from finrag.application import FinRAGSystem
    from finrag.core.config import PROJECT_ROOT, RAGConfig
    from finrag.core import FinRAGResponse

    assert create_app is not None
    assert FinRAGSystem is not None
    assert FinRAGResponse is not None
    assert Path(RAGConfig().data_path) == PROJECT_ROOT / "data" / "documents"
    assert not hasattr(RAGConfig(), "index_save_path")


def test_removed_local_storage_implementations_are_not_public():
    import finrag.indexing as indexing
    import finrag.ingestion as ingestion

    assert not hasattr(indexing, "SQLiteNodeStore")
    assert not hasattr(ingestion, "DocumentRegistry")


def test_project_directories_are_professionally_grouped():
    from finrag.core.config import PROJECT_ROOT

    assert (PROJECT_ROOT / "src" / "finrag").is_dir()
    assert (PROJECT_ROOT / "src" / "finrag" / "application").is_dir()
    assert (PROJECT_ROOT / "src" / "finrag" / "storage").is_dir()
    assert (PROJECT_ROOT / "src" / "finrag" / "api" / "routes").is_dir()
    assert (PROJECT_ROOT / "apps" / "web" / "src" / "app").is_dir()
    assert (PROJECT_ROOT / "apps" / "web" / "src" / "api").is_dir()
    assert (PROJECT_ROOT / "apps" / "web" / "src" / "components").is_dir()
    assert (PROJECT_ROOT / "apps" / "web" / "src" / "styles").is_dir()
    assert (PROJECT_ROOT / "apps" / "web" / "src" / "types").is_dir()
    assert (PROJECT_ROOT / "data" / "documents").is_dir()
    assert (PROJECT_ROOT / "datasets" / "eval").is_dir()
    assert (PROJECT_ROOT / "scripts").is_dir()
    assert (PROJECT_ROOT / "tests" / "api").is_dir()
    assert (PROJECT_ROOT / "tests" / "core").is_dir()
    assert (PROJECT_ROOT / "tests" / "evaluation").is_dir()
    assert (PROJECT_ROOT / "tests" / "generation").is_dir()
    assert (PROJECT_ROOT / "tests" / "ingestion").is_dir()
    assert (PROJECT_ROOT / "tests" / "indexing").is_dir()
    assert (PROJECT_ROOT / "tests" / "retrieval").is_dir()
    assert (PROJECT_ROOT / "tests" / "system").is_dir()


def test_internal_removed_module_paths_are_not_importable():
    from finrag.core.config import PROJECT_ROOT

    removed_files = [
        "src/finrag/core/system.py",
        "src/finrag/ingestion/document_ingestion.py",
        "src/finrag/ingestion/postgres_registry.py",
        "src/finrag/indexing/data_preparation.py",
        "src/finrag/indexing/index_construction.py",
        "src/finrag/indexing/postgres_store.py",
        "src/finrag/retrieval/retrieval_optimization.py",
        "src/finrag/generation/generation_integration.py",
        "scripts/run_retrieval_eval.py",
        "scripts/run_ragas_eval.py",
    ]
    for relative_path in removed_files:
        assert not (PROJECT_ROOT / relative_path).exists(), relative_path

    removed_modules = [
        "finrag.core.system",
        "finrag.ingestion.document_ingestion",
        "finrag.ingestion.postgres_registry",
        "finrag.indexing.data_preparation",
        "finrag.indexing.index_construction",
        "finrag.indexing.postgres_store",
        "finrag.retrieval.retrieval_optimization",
        "finrag.generation.generation_integration",
        "finrag.storage.stores",
    ]
    for module_name in removed_modules:
        assert importlib.util.find_spec(module_name) is None, module_name


def test_core_schema_reexports_llamaindex_native_retrieval_types():
    from llama_index.core.schema import NodeWithScore, TextNode

    from finrag.core.node_schema import NodeWithScore as ExportedNodeWithScore
    from finrag.core.node_schema import TextNode as ExportedTextNode

    import finrag.core as core
    import finrag.core.node_schema as node_schema

    assert not hasattr(core, "KnowledgeDocument")
    assert not hasattr(core, "RAGResponse")
    assert not hasattr(node_schema, "KnowledgeDocument")
    assert ExportedTextNode is TextNode
    assert ExportedNodeWithScore is NodeWithScore



