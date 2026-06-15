from llama_index.core.schema import NodeWithScore, TextNode

from finrag.retrieval.reranker import JinaCompatibleReranker, build_reranker


def _result(node_id: str, text: str, score: float = 0.1) -> NodeWithScore:
    return NodeWithScore(node=TextNode(id_=node_id, text=text, metadata={"chunk_id": node_id}), score=score)


def test_build_reranker_only_supports_none_and_jina():
    assert build_reranker("none", "model", "", "", 3) is None
    assert build_reranker("local_bge", "model", "", "", 3) is None
    assert build_reranker("sentence_transformer", "model", "", "", 3) is None


def test_jina_reranker_reorders_by_http_scores():
    calls = []

    class FakeReranker(JinaCompatibleReranker):
        def _post_json(self, payload):
            calls.append(payload)
            return {
                "results": [
                    {"index": 1, "relevance_score": 0.93},
                    {"index": 0, "relevance_score": 0.21},
                ]
            }

    reranker = FakeReranker(
        model="jina-reranker-v2-base-multilingual",
        endpoint="https://rerank.example/v1/rerank",
        api_key="secret",
        top_n=1,
    )

    reranked = reranker.postprocess_nodes([_result("a", "客户风险"), _result("b", "产品匹配")], query_str="风险匹配")

    assert [item.node.node_id for item in reranked] == ["b"]
    assert reranked[0].score == 0.93
    assert calls[0]["model"] == "jina-reranker-v2-base-multilingual"
    assert calls[0]["query"] == "风险匹配"
    assert calls[0]["top_n"] == 1
    assert calls[0]["documents"] == ["客户风险", "产品匹配"]
