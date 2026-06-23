from types import SimpleNamespace

from llama_index.core import Settings
from llama_index.core.base.response.schema import Response
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.postprocessor import PrevNextNodePostprocessor, SimilarityPostprocessor
from llama_index.core.query_engine import CustomQueryEngine, RetrieverQueryEngine, RouterQueryEngine, TransformQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.storage.docstore import SimpleDocumentStore

from finrag.application.llamaindex_engines import (
    LazyKnowledgeQueryEngine,
    _build_hyde_engine,
    _build_knowledge_router,
    _build_step_back_engine,
    build_node_postprocessors,
    build_top_router,
)


class EmptyRetriever(BaseRetriever):
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return []


class StaticLLM(CustomLLM):
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="static-test-llm")

    def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
        return CompletionResponse(text="测试回答")

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs):
        yield CompletionResponse(text="测试回答", delta="测试回答")


class SelectingLLM(StaticLLM):
    def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
        assert "客户风险等级如何匹配？" in prompt
        assert "{query}" not in prompt
        assert "{tool_names}" not in prompt
        return CompletionResponse(text='[{"choice": 1, "reason": "知识库问题"}]')


class FinancialAnalysisSelectingLLM(StaticLLM):
    def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
        assert "如果我想判断公司收入质量好不好，应该重点看哪些财务信号？" in prompt
        assert "经营复盘" in prompt
        assert "财务指标" in prompt
        return CompletionResponse(text='[{"choice": 1, "reason": "财务资料库问题"}]')


class EchoQueryEngine(CustomQueryEngine):
    def custom_query(self, query_str: str):
        return Response(response=f"echo:{query_str}")


def test_hyde_engine_reuses_auto_merge_engine_with_transform_query_engine():
    old_llm = Settings._llm
    Settings._llm = None
    try:
        auto_merge_engine = RetrieverQueryEngine.from_args(
            retriever=EmptyRetriever(),
            llm=StaticLLM(),
            response_mode="compact",
        )
        engine = _build_hyde_engine(auto_merge_engine, StaticLLM())
    finally:
        Settings._llm = old_llm

    assert isinstance(engine, TransformQueryEngine)
    assert engine._query_engine is auto_merge_engine
    assert engine._query_transform.__class__.__name__ == "HyDEQueryTransform"


def test_step_back_engine_reuses_auto_merge_engine_with_transform_query_engine():
    old_llm = Settings._llm
    Settings._llm = None
    try:
        auto_merge_engine = RetrieverQueryEngine.from_args(
            retriever=EmptyRetriever(),
            llm=StaticLLM(),
            response_mode="compact",
        )
        engine = _build_step_back_engine(auto_merge_engine, StaticLLM())
    finally:
        Settings._llm = old_llm

    assert isinstance(engine, TransformQueryEngine)
    assert engine._query_engine is auto_merge_engine
    assert engine._query_transform.__class__.__name__ == "StepDecomposeQueryTransform"


def test_top_router_uses_injected_llm_without_default_openai():
    class FakeSystem:
        knowledge_query_engine = EchoQueryEngine()

        def ensure_knowledge_base_ready(self):
            return None

    old_llm = Settings._llm
    Settings._llm = None
    try:
        router = build_top_router(system=FakeSystem(), llm=SelectingLLM())
    finally:
        Settings._llm = old_llm

    assert isinstance(router, RouterQueryEngine)
    response = router.query("客户风险等级如何匹配？")

    assert str(response) == "echo:客户风险等级如何匹配？"
    assert response.metadata["selected_query_engine"] == "knowledge_router"
    assert response.metadata["route_type"] == "knowledge"


def test_top_router_guides_financial_analysis_questions_to_knowledge():
    class FakeSystem:
        knowledge_query_engine = EchoQueryEngine()

        def ensure_knowledge_base_ready(self):
            return None

    old_llm = Settings._llm
    Settings._llm = None
    try:
        router = build_top_router(system=FakeSystem(), llm=FinancialAnalysisSelectingLLM())
    finally:
        Settings._llm = old_llm

    response = router.query("如果我想判断公司收入质量好不好，应该重点看哪些财务信号？")

    assert str(response) == (
        "echo:如果我想判断公司收入质量好不好，应该重点看哪些财务信号？"
    )
    assert response.metadata["selected_query_engine"] == "knowledge_router"
    assert response.metadata["route_type"] == "knowledge"


def test_top_router_without_llm_returns_knowledge_engine_without_selector():
    class FakeSystem:
        knowledge_query_engine = EchoQueryEngine()

        def ensure_knowledge_base_ready(self):
            return None

    old_llm = Settings._llm
    Settings._llm = None
    try:
        router = build_top_router(system=FakeSystem(), llm=None)
    finally:
        Settings._llm = old_llm

    assert isinstance(router, LazyKnowledgeQueryEngine)


def test_knowledge_router_uses_injected_llm_instead_of_falling_back():
    auto_merge_engine = EchoQueryEngine()
    old_llm = Settings._llm
    Settings._llm = None
    try:
        router = _build_knowledge_router(
            auto_merge_engine=auto_merge_engine,
            hyde_engine=None,
            step_back_engine=None,
            llm=StaticLLM(),
        )
    finally:
        Settings._llm = old_llm

    assert isinstance(router, RouterQueryEngine)


def test_knowledge_router_registers_only_core_retrieval_tools():
    auto_merge_engine = EchoQueryEngine()
    hyde_engine = EchoQueryEngine()
    step_back_engine = EchoQueryEngine()
    old_llm = Settings._llm
    Settings._llm = None
    try:
        router = _build_knowledge_router(
            auto_merge_engine=auto_merge_engine,
            hyde_engine=hyde_engine,
            step_back_engine=step_back_engine,
            llm=StaticLLM(),
        )
    finally:
        Settings._llm = old_llm

    assert [metadata.name for metadata in router._metadatas] == ["hyde", "step_back", "auto_merge"]


def test_zero_score_threshold_disables_similarity_filter():
    config = SimpleNamespace(score_threshold=0.0, context_token_budget=2400, neighbor_window=2)

    processors = build_node_postprocessors(config=config, reranker=None, docstore=None)

    assert not any(isinstance(processor, SimilarityPostprocessor) for processor in processors)


def test_prev_next_postprocessor_uses_configured_neighbor_window():
    config = SimpleNamespace(score_threshold=0.0, context_token_budget=2400, neighbor_window=2)

    processors = build_node_postprocessors(config=config, reranker=None, docstore=SimpleDocumentStore())
    prev_next = next(processor for processor in processors if isinstance(processor, PrevNextNodePostprocessor))

    assert prev_next.num_nodes == 2
