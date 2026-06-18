from llama_index.core.schema import TextNode

from finrag.core.config import RAGConfig
from scripts import evaluate_retrieval


def _node(filename: str, chunk_id: str, text: str) -> TextNode:
    return TextNode(
        text=text,
        id_=chunk_id,
        metadata={"filename": filename, "chunk_id": chunk_id, "chunk_level": 3},
    )


def test_retrieval_eval_normalizes_doc_titles_and_reports_hit_chunks():
    cases = [
        {
            "id": "case-1",
            "question": "受益所有人识别路径？",
            "expected_doc_titles": ["反洗钱客户尽职调查与受益所有人识别制度"],
            "expected_keywords": ["受益所有人"],
        }
    ]

    def search_fn(question, top_k):
        return [_node("反洗钱客户尽职调查与受益所有人识别制度.md", "chunk-1", "受益所有人识别路径")]

    report = evaluate_retrieval.evaluate_retrieval_cases(cases, search_fn, top_k=3)

    result = report["cases"][0]["result"]
    assert report["summary"]["retrieval_strategy"] == "milvus_hybrid_retriever"
    assert report["metrics"]["hit_at_1"] == 1.0
    assert report["metrics"]["mrr"] == 1.0
    assert result["ranked_doc_titles"] == ["反洗钱客户尽职调查与受益所有人识别制度"]
    assert result["hit_chunk_ids"] == ["chunk-1"]


def test_retrieval_eval_reports_hit_chunks_from_matching_node_titles():
    cases = [
        {
            "id": "case-duplicate-docs",
            "question": "B 文档的要求是什么？",
            "expected_doc_titles": ["B文档"],
            "expected_keywords": ["命中"],
        }
    ]

    def search_fn(question, top_k):
        return [
            _node("A文档.md", "chunk-a1", "A 文档内容"),
            _node("A文档.md", "chunk-a2", "A 文档另一段"),
            _node("B文档.md", "chunk-b1", "B 文档命中内容"),
        ]

    report = evaluate_retrieval.evaluate_retrieval_cases(cases, search_fn, top_k=3)

    result = report["cases"][0]["result"]
    assert result["ranked_doc_titles"] == ["A文档", "B文档"]
    assert result["ranked_chunk_ids"] == ["chunk-a1", "chunk-a2", "chunk-b1"]
    assert result["hit_chunk_ids"] == ["chunk-b1"]


def test_retrieval_eval_has_fixed_hybrid_retriever_strategy_label():
    assert evaluate_retrieval.RETRIEVAL_STRATEGY == "milvus_hybrid_retriever"


def test_retrieval_eval_human_report_is_readable():
    report = {
        "summary": {"case_count": 1, "top_k": 3, "retrieval_strategy": "milvus_hybrid_retriever"},
        "metrics": {"hit_at_1": 1.0, "hit_at_3": 1.0, "mrr": 1.0, "keyword_coverage": 1.0},
        "cases": [],
    }

    text = evaluate_retrieval.build_human_report(report)

    assert "FinRAG 检索评估报告" in text
    assert "milvus_hybrid_retriever" in text
    assert "| hit_at_1 | 1.0 |" in text


def test_retrieval_eval_search_fn_uses_runtime_finrag_system(monkeypatch, tmp_path):
    calls = []

    class FakeHybridRetriever:
        def __init__(self):
            self.top_k = 3
            self.filters = {}

        def retrieve(self, query_bundle):
            calls.append(("retrieve", str(query_bundle.query_str), self.top_k, self.filters))
            return [_node("理财产品适当性销售与风险揭示规范.md", "chunk-runtime", "客户风险等级匹配")]

    class FakeSystem:
        def __init__(self, config):
            calls.append(("init", config.data_path))
            self.hybrid_retriever = FakeHybridRetriever()

        def ensure_knowledge_base_ready(self):
            calls.append(("ready",))

    monkeypatch.setattr(evaluate_retrieval, "FinRAGSystem", FakeSystem, raising=False)

    search_fn = evaluate_retrieval._build_local_search_fn(RAGConfig(data_path=str(tmp_path)))
    results = search_fn("客户风险等级如何匹配？", 2)

    assert results[0].metadata["chunk_id"] == "chunk-runtime"
    assert calls == [
        ("init", str(tmp_path.resolve())),
        ("ready",),
        ("retrieve", "客户风险等级如何匹配？", 2, {"knowledge_base_id": "finance"}),
    ]
