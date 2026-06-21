"""Tests for LlamaIndex-first knowledge query engine path in QAPipelineService."""

from types import SimpleNamespace

from llama_index.core.schema import NodeWithScore, TextNode

from finrag.core.config import RAGConfig
from finrag.application.system import FinRAGSystem
import finrag.application.system as system_module


class FakeGeneration:
    def __init__(self):
        self.llm = None  # simulate no LLM so pipeline uses retrieve-only path


class FakeRouterEngine:
    """Mock top-level LlamaIndex router engine returning pre-set result nodes."""

    def __init__(self, results, *, answer="根据证据回答[1]", response_gen=None, error=None):
        self.results = results
        self.answer = answer
        self.response_gen = response_gen
        self.error = error
        self.calls = []

    def query(self, query_str):
        self.calls.append(str(query_str))
        if self.error is not None:
            raise self.error
        return FakeResponse(self.results, answer=self.answer, response_gen=self.response_gen)


class FakeResponse:
    def __init__(self, source_nodes, *, answer, response_gen=None):
        self.source_nodes = source_nodes
        self.response_gen = response_gen
        self.response = answer

    def __str__(self):
        return self.response


def _knowledge_node(text, chunk_id, **extra_meta):
    return TextNode(
        text=text,
        id_=chunk_id,
        metadata={
            "filename": "suitability.md",
            "file_type": "md",
            "chunk_id": chunk_id,
            "parent_chunk_id": f"parent-{chunk_id}",
            "root_chunk_id": "root-1",
            "chunk_level": 3,
            "chunk_idx": 0,
            "knowledge_base_id": "kb-finance",
            **extra_meta,
        },
    )


def _setup_knowledge_system(tmp_path, engine, generation=None):
    """Create a FinRAGSystem wired for the router query engine path."""
    system = FinRAGSystem(RAGConfig(data_path=str(tmp_path)))
    system.router_engine = engine
    system.generation_module = generation or FakeGeneration()
    system.data_module = SimpleNamespace()
    system.index_module = SimpleNamespace()

    def noop_ensure(knowledge_base_id=None):
        pass

    system.ensure_knowledge_base_ready = noop_ensure
    return system


def test_knowledge_engine_retrieves_and_generates(tmp_path):
    node = _knowledge_node("客户风险等级应与产品风险等级匹配", "leaf-1")
    engine = FakeRouterEngine([NodeWithScore(node=node, score=0.9)])
    system = _setup_knowledge_system(tmp_path, engine)

    response = system.ask_question("客户风险等级如何匹配？", knowledge_base_id="kb-finance", return_sources=True, return_trace=True)
    payload = response.to_dict()

    assert payload["answer"] == "根据证据回答[1]"
    assert len(system.router_engine.calls) == 1
    assert payload["route_type"] == "knowledge"
    assert payload["retrieval_strategy"] == "llamaindex_router"
    assert payload["trace"]["final_decision"] == "generate"
    assert payload["trace"]["evidence_nodes"][0]["chunk_id"] == "leaf-1"


def test_generated_answer_body_is_returned_with_structured_sources(tmp_path):
    node = _knowledge_node("开户资料包括营业执照和授权委托书", "leaf-1", filename="policy.md")
    answer = (
        "对公客户首次建立关系时，需要提供营业执照、授权委托书等资料。[1]\n\n"
        "[1] 来源：policy.md\n"
        "[2] 来源：policy.md"
    )
    engine = FakeRouterEngine([NodeWithScore(node=node, score=0.9)], answer=answer)
    system = _setup_knowledge_system(tmp_path, engine)

    response = system.ask_question("对公客户首次建立关系时需要哪些资料？", knowledge_base_id="kb-finance", return_sources=True)

    assert response.answer == answer
    assert response.sources[0].filename == "policy.md"


def test_knowledge_engine_emits_stream_tokens(tmp_path):
    node = _knowledge_node("客户风险等级应与产品风险等级匹配", "leaf-1")

    engine = FakeRouterEngine(
        [NodeWithScore(node=node, score=0.9)],
        response_gen=iter(["根据", "证据", "回答[1]"]),
    )
    system = _setup_knowledge_system(tmp_path, engine)
    events = []

    response = system.ask_question("客户风险等级如何匹配？", knowledge_base_id="kb-finance", return_sources=True, event_sink=events.append)
    assert response.answer == "根据证据回答[1]"
    assert [event["text"] for event in events if event["type"] == "token"] == ["根据", "证据", "回答[1]"]


def test_knowledge_engine_treats_missing_scores_as_zero(tmp_path):
    node = _knowledge_node("客户风险等级应与产品风险等级匹配", "leaf-1")
    engine = FakeRouterEngine([NodeWithScore(node=node, score=None)])
    system = _setup_knowledge_system(tmp_path, engine)

    response = system.ask_question("客户风险等级如何匹配？", knowledge_base_id="kb-finance", return_sources=True, return_trace=True)
    payload = response.to_dict()

    assert payload["answer"] == "根据证据回答[1]"
    assert payload["sources"][0]["score"] == 0.0
    assert payload["trace"]["retrieved_nodes"][0]["score"] is None


def test_ask_pipeline_uses_llamaindex_router_strategy(tmp_path):
    node = _knowledge_node("受益所有人识别用于确认实际控制关系", "leaf-1")
    engine = FakeRouterEngine([NodeWithScore(node=node, score=0.9)])
    system = _setup_knowledge_system(tmp_path, engine)

    response = system.ask_question("为什么要做受益所有人识别？", knowledge_base_id="kb-finance", return_trace=True)
    payload = response.to_dict()

    assert payload["retrieval_strategy"] == "llamaindex_router"
    assert payload["trace"]["retrieval_strategy"] == "llamaindex_router"


def test_router_path_does_not_initialize_generation_module(tmp_path):
    node = _knowledge_node("受益所有人识别用于确认实际控制关系", "leaf-1")
    engine = FakeRouterEngine([NodeWithScore(node=node, score=0.9)])
    system = _setup_knowledge_system(tmp_path, engine)
    system.generation_module = None

    response = system.ask_question("为什么要做受益所有人识别？", knowledge_base_id="kb-finance")

    assert response.answer == "根据证据回答[1]"
    assert system.generation_module is None


def test_router_general_route_without_sources(tmp_path):
    engine = FakeRouterEngine([], answer="普通回答")
    system = _setup_knowledge_system(tmp_path, engine)

    response = system.ask_question("帮我写一首春天的诗", knowledge_base_id="kb-finance", return_sources=True, return_trace=True)
    payload = response.to_dict()

    assert len(system.router_engine.calls) == 1
    assert payload["answer"] == "普通回答"
    assert payload["route_type"] == "general"
    assert payload["sources"] == []
    assert payload["trace"]["final_decision"] == "generate"


def test_general_route_bypasses_knowledge_engine(tmp_path):
    engine = FakeRouterEngine([], answer="普通回答")
    system = _setup_knowledge_system(tmp_path, engine)

    response = system.ask_question("帮我写一首春天的诗", knowledge_base_id="kb-finance", return_trace=True)
    payload = response.to_dict()

    assert payload["route_type"] == "general"
    assert payload["answer"] == "普通回答"
    assert system.router_engine.calls == ["帮我写一首春天的诗"]
    assert payload["sources"] == []
    assert payload["trace"]["retrieved_nodes"] == []
    assert payload["trace"]["final_decision"] == "generate"


def test_sources_map_leaf_scores_to_parent_evidence(tmp_path):
    leaf = TextNode(
        text="叶子命中", id_="leaf-1",
        metadata={"chunk_id": "leaf-1", "parent_chunk_id": "parent-1", "root_chunk_id": "root-1",
                   "chunk_level": 3, "chunk_idx": 0, "filename": "policy.md", "file_type": "md"},
    )
    parent = TextNode(
        text="父级证据", id_="parent-1",
        metadata={"chunk_id": "parent-1", "parent_chunk_id": "root-1", "root_chunk_id": "root-1",
                   "chunk_level": 2, "chunk_idx": 0, "filename": "policy.md", "file_type": "md"},
    )
    system = FinRAGSystem(RAGConfig(data_path=str(tmp_path)))
    sources = system.qa_pipeline.build_sources([parent], [NodeWithScore(node=leaf, score=0.73)])
    assert sources[0].score == 0.73


def test_evidence_is_deduped_and_sources_are_renumbered(tmp_path):
    first = _knowledge_node("客户风险等级应与产品风险等级匹配", "leaf-1")
    gen = FakeGeneration()
    engine = FakeRouterEngine([NodeWithScore(node=first, score=0.9)])
    system = _setup_knowledge_system(tmp_path, engine, generation=gen)
    system.data_module = SimpleNamespace()

    response = system.ask_question("客户风险等级如何匹配？", knowledge_base_id="kb-finance", return_sources=True)
    assert response.answer == "根据证据回答[1]"


def test_analysis_event_sent_before_query(tmp_path):
    node = _knowledge_node("客户风险等级应与产品风险等级匹配", "leaf-1")
    engine = FakeRouterEngine([NodeWithScore(node=node, score=0.9)])
    system = _setup_knowledge_system(tmp_path, engine)
    events = []

    system.ask_question("客户风险等级如何匹配？", knowledge_base_id="kb-finance", return_sources=True, event_sink=events.append)

    analysis_events = [e for e in events if e["type"] == "analysis"]
    assert len(analysis_events) == 1
    route_events = [e for e in events if e["type"] == "route"]
    assert len(route_events) == 1
    assert route_events[0]["route_type"] == "knowledge"


def test_pipeline_step_events_and_trace_are_emitted_for_router_path(tmp_path):
    node = _knowledge_node("客户风险等级应与产品风险等级匹配", "leaf-1")
    engine = FakeRouterEngine(
        [NodeWithScore(node=node, score=0.9)],
        response_gen=iter(["根据", "证据", "回答[1]"]),
    )
    system = _setup_knowledge_system(tmp_path, engine)
    events = []

    response = system.ask_question(
        "客户风险等级如何匹配？",
        knowledge_base_id="kb-finance",
        return_sources=True,
        return_trace=True,
        event_sink=events.append,
    )
    payload = response.to_dict()

    step_events = [event for event in events if event["type"] == "pipeline_step"]
    step_ids = [event["id"] for event in step_events]
    assert "query_analysis" in step_ids
    assert "router" in step_ids
    assert "hybrid_search" in step_ids
    assert "evidence_window" in step_ids
    assert "streaming_answer" in step_ids
    assert any(event["id"] == "streaming_answer" and event["status"] == "running" for event in step_events)
    assert any(event["id"] == "streaming_answer" and event["status"] == "complete" for event in step_events)
    assert payload["trace"]["pipeline_steps"][-1]["id"] == "streaming_answer"


def test_knowledge_unavailable_returns_structured_error(tmp_path):
    engine = FakeRouterEngine([], error=RuntimeError("Milvus 不可用"))
    system = _setup_knowledge_system(tmp_path, engine)
    events = []

    response = system.ask_question(
        "客户风险等级如何匹配？",
        knowledge_base_id="kb-finance",
        return_sources=True,
        return_trace=True,
        event_sink=events.append,
    )
    payload = response.to_dict()

    assert "知识库当前不可用" in payload["answer"]
    assert payload["trace"]["final_decision"] == "knowledge_unavailable"
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["code"] == "milvus_unavailable"
    assert error_events[0]["retryable"] is True
    done_events = [e for e in events if e["type"] == "done"]
    assert len(done_events) == 1
    assert any(
        event["type"] == "pipeline_step" and event["id"] == "router" and event["status"] == "error"
        for event in events
    )


def test_missing_router_engine_does_not_fall_back_to_query_analysis(tmp_path):
    class NoFallbackGeneration(FakeGeneration):
        def analyze_query(self, question):
            raise AssertionError("router-only pipeline should not call analyze_query")

    system = _setup_knowledge_system(tmp_path, engine=None, generation=NoFallbackGeneration())
    events = []

    response = system.ask_question("客户风险等级如何匹配？", knowledge_base_id="kb-finance", return_trace=True, event_sink=events.append)
    payload = response.to_dict()

    assert "知识库当前不可用" in payload["answer"]
    assert payload["trace"]["final_decision"] == "knowledge_unavailable"
    assert payload["trace"]["events"][0]["stage"] == "error"
    assert "router_engine" in payload["trace"]["events"][0]["message"]
    assert [event["type"] for event in events if event["type"] != "pipeline_step"] == ["analysis", "error", "done"]
    assert any(
        event["type"] == "pipeline_step" and event["id"] == "router" and event["status"] == "error"
        for event in events
    )


def test_ask_question_does_not_fallback_to_another_knowledge_base_runtime(tmp_path):
    finance_engine = FakeRouterEngine([], answer="finance answer")
    system = FinRAGSystem.__new__(FinRAGSystem)
    system.config = RAGConfig(data_path=str(tmp_path), milvus_collection="finrag_leaf_nodes")
    system.qa_pipeline = system_module.QAPipelineService(system)
    system.reranker = None
    system.kb_runtimes = {
        "finance": system_module.KnowledgeBaseRuntime(
            scope=system.knowledge_base_scope("finance"),
            data_module=SimpleNamespace(),
            index_module=SimpleNamespace(),
            generation_module=FakeGeneration(),
            router_engine=finance_engine,
        ),
        "risk": system_module.KnowledgeBaseRuntime(
            scope=system.knowledge_base_scope("risk"),
            data_module=SimpleNamespace(),
            index_module=SimpleNamespace(),
            generation_module=FakeGeneration(),
            router_engine=None,
        ),
    }
    system.ensure_knowledge_base_ready = lambda knowledge_base_id=None: system._activate_runtime(
        system.kb_runtimes[knowledge_base_id or system.config.knowledge_base_id]
    )

    response = system.ask_question("客户风险等级如何匹配？", knowledge_base_id="risk", return_trace=True)

    assert finance_engine.calls == []
    assert "知识库当前不可用" in response.answer
    assert response.trace is not None
    assert response.trace.filters == {
        "knowledge_base_id": "risk",
        "collection": "finrag_leaf_nodes__kb_risk",
    }
