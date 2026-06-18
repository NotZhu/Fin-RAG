from pathlib import Path

import pytest

from finrag.core.config import PACKAGE_DIR, PROJECT_ROOT, RAGConfig


def test_default_paths_are_absolute_and_finance_oriented():
    config = RAGConfig()

    assert Path(config.data_path).is_absolute()
    assert Path(config.data_path) == PROJECT_ROOT / "data" / "documents"
    assert config.knowledge_base_id == "finance"
    assert config.chunk_size == 300
    assert config.chunk_overlap == 60
    assert config.reranker_provider == "none"
    assert config.reranker_model == "jina-reranker-v2-base-multilingual"
    assert config.reranker_endpoint == ""
    assert config.reranker_api_key == ""
    assert config.retrieval_strategy == "llamaindex_router"
    assert Path(config.llamaindex_index_store_dir) == PROJECT_ROOT / "storage" / "llamaindex"
    assert config.auto_merge_ratio_threshold == 0.5
    assert config.context_token_budget == 2400
    assert Path(config.upload_dir) == PROJECT_ROOT / "storage" / "uploads"
    assert config.max_upload_bytes == 20 * 1024 * 1024
    assert PACKAGE_DIR == PROJECT_ROOT / "src" / "finrag"
    assert not hasattr(config, "hybrid_enabled")
    assert not hasattr(config, "ask_streaming")
    assert not hasattr(config, "storage_backend")
    assert not hasattr(config, "registry_path")
    assert not hasattr(config, "node_store_path")
    assert not hasattr(config, "document_dir")
    assert not hasattr(config, "query_rewrite_enabled")
    assert not hasattr(config, "query_rewrite_max_rounds")
    assert not hasattr(config, "low_confidence_threshold")
    assert not hasattr(config, "min_evidence_count")
    assert not hasattr(config, "auto_merge_l1_threshold")
    assert not hasattr(config, "auto_merge_l2_threshold")
    assert not hasattr(config, "evidence_token_budget")


def test_config_rejects_removed_local_storage_kwargs():
    with pytest.raises(TypeError, match="未知 RAGConfig 字段"):
        RAGConfig(index_save_path="storage/index_manifest")


def test_config_can_be_created_from_environment(monkeypatch):
    monkeypatch.setenv("RAG_DATA_PATH", "env_data")
    monkeypatch.setenv("RAG_KNOWLEDGE_BASE_ID", "kb-test")
    monkeypatch.setenv("RAG_TOP_K", "5")
    monkeypatch.setenv("RAG_RETRIEVAL_CANDIDATE_K", "12")
    monkeypatch.setenv("RAG_RRF_K", "42")
    monkeypatch.setenv("RAG_CHUNK_SIZE", "300")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "60")
    monkeypatch.setenv("RAG_RERANKER_PROVIDER", "jina")
    monkeypatch.setenv("RAG_RERANKER_MODEL", "jina-reranker-v2-base-multilingual")
    monkeypatch.setenv("RAG_RERANKER_ENDPOINT", "https://api.jina.ai/v1/rerank")
    monkeypatch.setenv("RAG_RERANKER_API_KEY", "secret")
    monkeypatch.setenv("RAG_RERANKER_TOP_N", "4")
    monkeypatch.setenv("RAG_RETRIEVAL_STRATEGY", "llamaindex_router")
    monkeypatch.setenv("RAG_LLAMAINDEX_INDEX_STORE_DIR", "custom_llama")
    monkeypatch.setenv("RAG_AUTO_MERGE_RATIO_THRESHOLD", "0.75")
    monkeypatch.setenv("RAG_CONTEXT_TOKEN_BUDGET", "1800")
    monkeypatch.setenv("RAG_UPLOAD_DIR", "tmp_uploads")
    monkeypatch.setenv("RAG_MAX_UPLOAD_BYTES", "4096")

    config = RAGConfig.from_env()

    assert Path(config.data_path) == PROJECT_ROOT / "env_data"
    assert config.knowledge_base_id == "kb-test"
    assert config.top_k == 5
    assert config.retrieval_candidate_k == 12
    assert config.rrf_k == 42
    assert config.chunk_size == 300
    assert config.chunk_overlap == 60
    assert config.reranker_provider == "jina"
    assert config.reranker_model == "jina-reranker-v2-base-multilingual"
    assert config.reranker_endpoint == "https://api.jina.ai/v1/rerank"
    assert config.reranker_api_key == "secret"
    assert config.reranker_top_n == 4
    assert config.retrieval_strategy == "llamaindex_router"
    assert Path(config.llamaindex_index_store_dir) == PROJECT_ROOT / "custom_llama"
    assert config.auto_merge_ratio_threshold == 0.75
    assert config.context_token_budget == 1800
    assert Path(config.upload_dir) == PROJECT_ROOT / "tmp_uploads"
    assert config.max_upload_bytes == 4096
    assert not hasattr(config, "context_window_size")
    assert not hasattr(config, "retry_candidate_k")
    assert not hasattr(config, "step_back_enabled")
    assert not hasattr(config, "hyde_enabled")


def test_removed_llamaindex_storage_dir_alias_is_ignored(monkeypatch):
    monkeypatch.setenv("RAG_LLAMAINDEX_STORAGE_DIR", "ignored_llama")

    config = RAGConfig.from_env()

    assert Path(config.llamaindex_index_store_dir) == PROJECT_ROOT / "storage" / "llamaindex"


def test_llamaindex_index_store_dir_uses_current_environment_key(monkeypatch):
    monkeypatch.setenv("RAG_LLAMAINDEX_STORAGE_DIR", "ignored_llama")
    monkeypatch.setenv("RAG_LLAMAINDEX_INDEX_STORE_DIR", "new_llama")

    config = RAGConfig.from_env()

    assert Path(config.llamaindex_index_store_dir) == PROJECT_ROOT / "new_llama"


def test_service_stack_config_is_postgres_redis_milvus_only(monkeypatch):
    monkeypatch.setenv("RAG_STORAGE_BACKEND", "local")
    monkeypatch.setenv("RAG_DATABASE_URL", "postgresql://finrag:test@db:5432/finrag")
    monkeypatch.setenv("RAG_REDIS_URL", "redis://redis:6379/1")
    monkeypatch.setenv("RAG_MILVUS_HOST", "milvus")
    monkeypatch.setenv("RAG_MILVUS_PORT", "19531")
    monkeypatch.setenv("RAG_MILVUS_COLLECTION", "finrag_test_nodes")
    monkeypatch.setenv("RAG_HYBRID_ENABLED", "false")
    monkeypatch.setenv("RAG_ASK_STREAMING", "true")

    config = RAGConfig.from_env()

    assert not hasattr(config, "storage_backend")
    assert not hasattr(config, "hybrid_enabled")
    assert not hasattr(config, "ask_streaming")
    assert config.database_url == "postgresql://finrag:test@db:5432/finrag"
    assert config.redis_url == "redis://redis:6379/1"
    assert config.milvus_host == "milvus"
    assert config.milvus_port == 19531
    assert config.milvus_collection == "finrag_test_nodes"
    assert "storage_backend" not in config.to_dict()
    assert "vector_backend" not in config.to_dict()
    assert "hybrid_enabled" not in config.to_dict()
    assert "ask_streaming" not in config.to_dict()


def test_env_example_documents_postgres_redis_milvus_stack():
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    assert "RAG_VECTOR_BACKEND=" not in env_example
    assert "RAG_STORAGE_BACKEND=" not in env_example
    assert "RAG_INDEX_SAVE_PATH=" not in env_example
    assert "RAG_DOCUMENT_REGISTRY_PATH=" not in env_example
    assert "RAG_NODE_STORE_PATH=" not in env_example
    assert "RAG_DOCUMENT_DIR=" not in env_example
    assert "RAG_EMBEDDING_MODEL=" + "mo" + "ck" not in env_example
    assert "chroma" not in env_example.lower()
    assert "sqlite" not in env_example.lower()
    assert "documents.json" not in env_example.lower()

    for key in [
        "RAG_DATABASE_URL",
        "RAG_REDIS_URL",
        "RAG_MILVUS_HOST",
        "RAG_MILVUS_PORT",
        "RAG_MILVUS_COLLECTION",
        "RAG_CHUNK_SIZE",
        "RAG_CHUNK_OVERLAP",
        "RAG_RERANKER_ENDPOINT",
        "RAG_RERANKER_API_KEY",
        "RAG_RETRIEVAL_STRATEGY",
        "RAG_LLAMAINDEX_INDEX_STORE_DIR",
        "RAG_AUTO_MERGE_RATIO_THRESHOLD",
        "RAG_CONTEXT_TOKEN_BUDGET",
        "RAG_UPLOAD_DIR",
        "RAG_MAX_UPLOAD_BYTES",
    ]:
        assert f"{key}=" in env_example
    assert "RAG_HYBRID_ENABLED=" not in env_example
    assert "RAG_ASK_STREAMING=" not in env_example
    assert "RAG_QUERY_REWRITE_ENABLED=" not in env_example
    assert "RAG_QUERY_REWRITE_MAX_ROUNDS=" not in env_example
    assert "RAG_LOW_CONFIDENCE_THRESHOLD=" not in env_example
    assert "RAG_MIN_EVIDENCE_COUNT=" not in env_example
    assert "RAG_AUTO_MERGE_L1_THRESHOLD=" not in env_example
    assert "RAG_AUTO_MERGE_L2_THRESHOLD=" not in env_example
    assert "RAG_EVIDENCE_TOKEN_BUDGET=" not in env_example
