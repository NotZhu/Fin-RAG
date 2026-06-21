import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from finrag.retrieval.llamaindex_native import (
    HybridRetrieverUnavailable,
    MilvusNativeHybridRetriever,
    SentenceAwareTokenBudgetPostprocessor,
)


class RecordingVectorIndex:
    def __init__(self, *, enable_sparse=True, results=None):
        self.vector_store = type(
            "VectorStore",
            (),
            {
                "enable_sparse": enable_sparse,
                "hybrid_ranker": "RRFRanker",
                "hybrid_ranker_params": {"k": 71},
            },
        )()
        self.calls = []
        self.results = list(results or [])

    def as_retriever(self, **kwargs):
        self.calls.append(kwargs)

        class Retriever:
            def __init__(self, results):
                self.results = results

            def retrieve(self, query_bundle):
                assert isinstance(query_bundle, QueryBundle)
                return list(self.results)

        return Retriever(self.results)


def test_milvus_native_hybrid_retriever_uses_milvus_hybrid_query_mode():
    node = TextNode(id_="leaf-a", text="客户风险等级", metadata={"knowledge_base_id": "finance"})
    vector_index = RecordingVectorIndex(results=[NodeWithScore(node=node, score=0.8)])

    retriever = MilvusNativeHybridRetriever(
        vector_index=vector_index,
        candidate_k=12,
        top_k=3,
        rrf_k=71,
        filters={"knowledge_base_id": "finance"},
    )
    results = retriever.retrieve("风险等级")

    assert results[0].node.node_id == "leaf-a"
    assert vector_index.calls
    call = vector_index.calls[0]
    assert call["vector_store_query_mode"] is VectorStoreQueryMode.HYBRID
    assert call["similarity_top_k"] == 12
    assert call["sparse_top_k"] == 12
    assert call["hybrid_top_k"] == 12
    assert call["filters"].filters[0].key == "knowledge_base_id"
    assert retriever.last_hybrid_trace == {
        "hybrid_provider": "milvus",
        "hybrid_mode": "native_dense_sparse",
        "hybrid_ranker": "RRFRanker",
        "rrf_k": 71,
        "candidate_k": 12,
    }


def test_milvus_native_hybrid_retriever_requires_sparse_schema():
    vector_index = RecordingVectorIndex(enable_sparse=False)

    with pytest.raises(HybridRetrieverUnavailable, match="hybrid_retriever_unavailable"):
        MilvusNativeHybridRetriever(vector_index=vector_index)


def test_sentence_aware_token_budget_drops_low_ranked_nodes_first():
    nodes = [
        NodeWithScore(node=TextNode(id_="a", text="第一句。第二句。"), score=0.9),
        NodeWithScore(node=TextNode(id_="b", text="第三句。第四句。"), score=0.8),
        NodeWithScore(node=TextNode(id_="c", text="第五句。第六句。"), score=0.1),
    ]
    postprocessor = SentenceAwareTokenBudgetPostprocessor(token_budget=20)

    processed = postprocessor.postprocess_nodes(nodes, query_str="test")

    assert [item.node.node_id for item in processed] == ["a", "b"]


def test_sentence_aware_token_budget_truncates_single_node_on_sentence_boundary():
    node = TextNode(id_="a", text="第一句包含足够内容。第二句应被截断。第三句也不应出现。")
    postprocessor = SentenceAwareTokenBudgetPostprocessor(token_budget=12)

    processed = postprocessor.postprocess_nodes([NodeWithScore(node=node, score=1.0)], query_str="test")

    assert len(processed) == 1
    assert processed[0].node.text.endswith("。")
    assert "第三句" not in processed[0].node.text
    assert processed[0].node.metadata["truncated"] is True