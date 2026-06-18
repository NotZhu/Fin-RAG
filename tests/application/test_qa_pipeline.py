from types import SimpleNamespace

from finrag.application.qa_pipeline import QAPipelineService


def _system_without_router(default_knowledge_base_id: str = "kb-config-default"):
    return SimpleNamespace(
        config=SimpleNamespace(
            knowledge_base_id=default_knowledge_base_id,
            retrieval_strategy="llamaindex_router",
            auto_merge_ratio_threshold=0.5,
        ),
        router_engine=None,
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
    assert response.trace.filters == {"knowledge_base_id": "kb-risk"}
    assert response.trace.pipeline_steps[-1]["id"] == "router"
    assert response.trace.pipeline_steps[-1]["status"] == "error"
    assert any(event["type"] == "error" for event in emitted_events)
    assert emitted_events[-1]["type"] == "done"
