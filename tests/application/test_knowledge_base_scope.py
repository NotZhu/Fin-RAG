from pathlib import Path

from finrag.application.knowledge_base_scope import KnowledgeBaseScope
from finrag.core.config import RAGConfig


def test_scope_derives_all_runtime_names_from_knowledge_base_id(tmp_path):
    config = RAGConfig(data_path=str(tmp_path / "documents"), milvus_collection="finrag_leaf_nodes")

    scope = KnowledgeBaseScope.from_config(config, "risk-desk")

    assert scope.knowledge_base_id == "risk-desk"
    assert scope.collection_name == "finrag_leaf_nodes__kb_risk_desk"
    assert scope.source_root == Path(config.data_path) / "risk-desk"
    assert scope.manifest_key == "risk-desk"
    assert scope.runtime_cache_key == "risk-desk"


def test_scope_requires_explicit_knowledge_base_id(tmp_path):
    config = RAGConfig(data_path=str(tmp_path / "documents"), knowledge_base_id="finance")

    scope = KnowledgeBaseScope.from_config(config, "finance")

    assert scope.knowledge_base_id == "finance"
    assert scope.source_root == Path(config.data_path) / "finance"
