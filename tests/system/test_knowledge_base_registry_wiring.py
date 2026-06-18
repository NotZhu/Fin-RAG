from finrag.application.knowledge_base_scope import KnowledgeBaseScope
from finrag.application.system import FinRAGSystem
from finrag.core.config import RAGConfig


def test_finrag_system_exposes_default_knowledge_base(tmp_path):
    system = FinRAGSystem(RAGConfig(data_path=str(tmp_path / "documents"), knowledge_base_id="finance"))

    knowledge_bases = system.list_knowledge_bases()

    assert knowledge_bases[0]["knowledge_base_id"] == "finance"
    assert knowledge_bases[0]["document_count"] == 0


def test_finrag_system_can_create_knowledge_base_by_id(tmp_path):
    system = FinRAGSystem(RAGConfig(data_path=str(tmp_path / "documents"), knowledge_base_id="finance"))

    created = system.create_knowledge_base("risk")

    assert created["knowledge_base_id"] == "risk"
    assert system.list_knowledge_bases()[-1]["knowledge_base_id"] == "risk"


def test_finrag_system_builds_scope_for_explicit_knowledge_base(tmp_path):
    system = FinRAGSystem(
        RAGConfig(
            data_path=str(tmp_path / "documents"),
            knowledge_base_id="finance",
            milvus_collection="finrag_leaf_nodes",
        )
    )

    scope = system.knowledge_base_scope("finance")

    assert isinstance(scope, KnowledgeBaseScope)
    assert scope.knowledge_base_id == "finance"
    assert scope.collection_name == "finrag_leaf_nodes__kb_finance"
