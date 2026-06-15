from types import SimpleNamespace

from llama_index.core import Settings
from llama_index.core.base.response.schema import Response
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import CompletionResponse, CustomLLM, LLMMetadata
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import CustomQueryEngine, RetrieverQueryEngine, RouterQueryEngine, TransformQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle

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
    assert str(router.query("客户风险等级如何匹配？")) == "echo:客户风险等级如何匹配？"


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
            summary_engine=None,
            auto_merge_engine=auto_merge_engine,
            sub_question_engine=None,
            hyde_engine=None,
            step_back_engine=None,
            llm=StaticLLM(),
        )
    finally:
        Settings._llm = old_llm

    assert isinstance(router, RouterQueryEngine)


def test_zero_score_threshold_disables_similarity_filter():
    config = SimpleNamespace(score_threshold=0.0, context_token_budget=2400)

    processors = build_node_postprocessors(config=config, reranker=None, docstore=None)

    assert not any(isinstance(processor, SimilarityPostprocessor) for processor in processors)
