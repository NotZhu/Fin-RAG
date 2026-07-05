from finrag.core.config import PROJECT_ROOT


def test_readme_is_github_style_and_current_state_only():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    for badge in [
        "Python 3.11",
        "FastAPI",
        "LlamaIndex",
        "Milvus",
        "PostgreSQL",
        "React 18",
        "Vite 5",
        "pytest",
        "vitest",
    ]:
        assert badge in readme

    for historical_term in ["改造", "迁移", "Phase", "legacy", "fallback", "旧实现"]:
        assert historical_term not in readme

    assert "docker compose up -d postgres etcd minio milvus" in readme
    assert "finrag rebuild --knowledge-base-id finance" in readme
    assert "uvicorn finrag.api:app --host 127.0.0.1 --port 8000" in readme
    assert "python -m scripts.generate_demo_documents --clean" in readme
    assert "python -m scripts.evaluate_demo_documents --json" in readme
    assert "python -m scripts.evaluate_demo_documents_ragas --dry-run" in readme
    assert "datasets/eval/demo_documents_suite.json" in readme
    assert "python -m scripts.evaluate_retrieval" not in readme
    assert "python -m scripts.evaluate_ragas" not in readme
    assert "`DoclingNodeParser`" in readme
    assert "`HierarchyBuilder`" in readme
    assert "none / jina" in readme
    assert "LlamaIndex `AutoMergingRetriever`" not in readme
    ask_request = readme.split("### Ask Request", 1)[1].split("### Ask Response", 1)[0]
    assert '"retrieval_strategy": "hybrid_rerank"' not in ask_request
    for removed_document_endpoint in [
        "`/documents`",
        "`/documents/upload`",
        "`/documents/{document_id}`",
        "`/documents/{document_id}/reindex`",
    ]:
        assert removed_document_endpoint not in readme
    assert "general -> 普通 LLM" in readme

    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "sentence-transformers" not in pyproject
    assert "ragas" in pyproject
    assert "langchain-openai" in pyproject

    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "RAGAS_EVAL_ENABLED=false" in env_example
    assert "RAGAS_LLM_MODEL=qwen3.7-max" in env_example
    assert "RAGAS_EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1" in env_example
    assert "RAGAS_EMBEDDING_MODEL=BAAI/bge-m3" in env_example
    assert "RAGAS_ANSWER_RELEVANCY_STRICTNESS=1" in env_example
