from finrag.core.config import PROJECT_ROOT


def test_readme_is_github_style_and_current_state_only():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    for badge in [
        "Python 3.11",
        "FastAPI",
        "LlamaIndex",
        "Milvus",
        "PostgreSQL",
        "Redis",
        "React 18",
        "Vite 5",
        "pytest",
        "vitest",
    ]:
        assert badge in readme

    for historical_term in ["改造", "迁移", "Phase", "legacy", "fallback", "旧实现"]:
        assert historical_term not in readme

    assert "docker compose up -d postgres redis etcd minio milvus" in readme
    assert "finrag rebuild" in readme
    assert "uvicorn finrag.api:app --host 127.0.0.1 --port 8000" in readme
    assert "python -m scripts.evaluate_retrieval --json" in readme
    assert "python -m scripts.evaluate_ragas datasets/eval/finance_ragas_eval_set.jsonl --json" in readme
    assert "chunk_sizes=[1200, 600, 300]" in readme
    assert "RAG_CHUNK_SIZE=300" in readme
    assert "RAG_CHUNK_OVERLAP=60" in readme
    assert "none / jina" in readme
    assert "LlamaIndex `AutoMergingRetriever`" not in readme
    ask_request = readme.split("### Ask Request", 1)[1].split("### Ask Response", 1)[0]
    assert '"retrieval_strategy": "hybrid_rerank"' not in ask_request
    assert "general -> 普通 LLM" in readme

    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "sentence-transformers" not in pyproject
