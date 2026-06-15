from finrag.core import FinRAGResponse, RAGTrace, RetrievedSource


def test_finrag_response_serializes_sources_and_trace():
    source = RetrievedSource(
        source_id=1,
        filename="policy.md",
        file_type="md",
        page_number=None,
        chunk_id="node-1",
        parent_chunk_id="parent-1",
        root_chunk_id="root-1",
        chunk_level=3,
        chunk_idx=0,
        score=0.9,
        snippet="客户风险等级应与产品风险等级匹配",
    )
    trace = RAGTrace(
        retrieval_strategy="llamaindex_router",
        route_type="knowledge",
        filters={"knowledge_base_id": "kb-finance"},
        timings_ms={"retrieval": 3.0},
        retrieved_nodes=[{"node_id": "node-1"}],
        evidence_nodes=[{"node_id": "node-1"}],
        events=[{"stage": "query_analysis"}],
        source_count=1,
    )

    payload = FinRAGResponse(
        question="客户风险等级如何匹配？",
        answer="客户风险等级应与产品风险等级匹配[1]",
        sources=[source],
        trace=trace,
        retrieval_strategy="llamaindex_router",
        route_type="knowledge",
    ).to_dict()

    assert payload["route_type"] == "knowledge"
    assert payload["retrieval_strategy"] == "llamaindex_router"
    assert "rewritten_query" not in payload
    assert payload["sources"][0]["filename"] == "policy.md"
    assert payload["sources"][0]["page_number"] is None
    assert payload["sources"][0]["score"] == 0.9
    assert "file_type" not in payload["sources"][0]
    assert "chunk_id" not in payload["sources"][0]
    assert "parent_chunk_id" not in payload["sources"][0]
    assert "root_chunk_id" not in payload["sources"][0]
    assert "chunk_level" not in payload["sources"][0]
    assert "chunk_idx" not in payload["sources"][0]
    assert "title_hint" not in payload["sources"][0]
    assert "department" not in payload["sources"][0]
    assert "regulatory_topic" not in payload["sources"][0]
    assert payload["trace"]["retrieval_strategy"] == "llamaindex_router"
    assert payload["trace"]["route_type"] == "knowledge"
    assert payload["trace"]["retrieved_nodes"] == [{"node_id": "node-1"}]
    assert payload["trace"]["evidence_nodes"] == [{"node_id": "node-1"}]
    legacy_trace_key = "_".join(("final", "evidence"))
    assert legacy_trace_key not in payload["trace"]
    assert "retrieved_chunks" not in payload["trace"]
    assert "context_documents" not in payload["trace"]
