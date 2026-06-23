from types import SimpleNamespace

from finrag.application.knowledge_base_scope import KnowledgeBaseScope
from finrag.application.qa_pipeline import QAPipelineService
from finrag.core.config import RAGConfig


def _system_without_router(default_knowledge_base_id: str = "kb-config-default"):
    config = RAGConfig(knowledge_base_id=default_knowledge_base_id, milvus_collection="finrag_leaf_nodes")
    return SimpleNamespace(
        config=config,
        router_engine=None,
        knowledge_base_scope=lambda knowledge_base_id: KnowledgeBaseScope.from_config(config, knowledge_base_id),
    )


def test_knowledge_unavailable_trace_records_effective_knowledge_base_id(
):
    emitted_events = []
    service = QAPipelineService(_system_without_router())

    response = service.ask_question(
        "客户风险等级如何匹配？",
        knowledge_base_id="kb-risk",
        return_trace=True,
        event_sink=emitted_events.append,
    )

    assert response.route_type == "knowledge"
    assert response.trace is not None
    assert response.trace.final_decision == "knowledge_unavailable"
    assert response.trace.filters == {
        "knowledge_base_id": "kb-risk",
        "collection": "finrag_leaf_nodes__kb_kb_risk",
    }
    assert response.trace.pipeline_steps[-1]["id"] == "query_router"
    assert response.trace.pipeline_steps[-1]["status"] == "error"
    assert any(event["type"] == "error" for event in emitted_events)
    assert emitted_events[-1]["type"] == "done"
