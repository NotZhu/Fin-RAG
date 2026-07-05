from __future__ import annotations

from llama_index.core.schema import NodeWithScore, TextNode

from finrag.evaluation.metrics import evaluate_retrieval_cases, evaluate_thresholds


def _node(filename: str, chunk_id: str, text: str, score: float = 0.8) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(
            text=text,
            id_=chunk_id,
            metadata={"filename": filename, "chunk_id": chunk_id, "chunk_level": 3},
        ),
        score=score,
    )


def test_retrieval_metrics_score_recall_ranking_keywords_empty_and_latency():
    cases = [
        {
            "id": "cash-flow",
            "question": "经营现金流和授信建议是什么？",
            "relevant_sources": [
                {"filename": "credit_review_report.pdf", "relevance": 3},
                {"filename": "erp_financial_snapshot.json", "relevance": 1},
            ],
            "source_keywords": ["经营现金流", "0.8 亿元", "暂停新增敞口"],
        },
        {
            "id": "empty",
            "question": "不存在的资料？",
            "relevant_sources": [{"filename": "credit_review_report.pdf", "relevance": 3}],
            "source_keywords": ["不存在"],
        },
    ]

    def search_fn(question: str, top_k: int):
        if "不存在" in question:
            return []
        return [
            _node("unrelated_policy.html", "chunk-other", "无关内容"),
            _node("credit_review_report.pdf", "chunk-credit", "经营现金流 0.8 亿元，重度压力下暂停新增敞口"),
            _node("erp_financial_snapshot.json", "chunk-erp", "现金余额和库存快照"),
        ]

    report = evaluate_retrieval_cases(
        cases,
        search_fn,
        top_k=3,
        strategy_name="raw_hybrid",
        latency_fn=lambda question, top_k, results: 30.0 if results else 10.0,
    )

    assert report["summary"] == {"case_count": 2, "top_k": 3, "retrieval_strategy": "raw_hybrid"}
    assert report["metrics"]["recall_at_3"] == 0.5
    assert report["metrics"]["precision_at_3"] == 0.3333
    assert report["metrics"]["hit_at_3"] == 0.5
    assert report["metrics"]["mrr"] == 0.25
    assert report["metrics"]["ndcg_at_3"] == 0.3295
    assert report["metrics"]["source_keyword_coverage"] == 0.5
    assert report["metrics"]["empty_retrieval_rate"] == 0.5
    assert report["metrics"]["avg_latency_ms"] == 20.0
    assert report["cases"][0]["result"]["ranked_sources"] == [
        "unrelated_policy.html",
        "credit_review_report.pdf",
        "erp_financial_snapshot.json",
    ]
    assert report["cases"][0]["result"]["hit_sources"] == [
        "credit_review_report.pdf",
        "erp_financial_snapshot.json",
    ]


def test_thresholds_keep_quality_gate_separate_from_diagnostics():
    gate = evaluate_thresholds(
        {
            "recall_at_5": 0.9,
            "mrr": 0.7,
            "ndcg_at_5": 0.68,
            "source_keyword_coverage": 0.8,
            "empty_retrieval_rate": 0.0,
            "precision_at_5": 0.2,
        },
        {
            "recall_at_5": 0.8,
            "mrr": 0.6,
            "ndcg_at_5": 0.65,
            "source_keyword_coverage": 0.75,
            "empty_retrieval_rate": {"max": 0.05},
        },
    )

    assert gate == {"passed": True, "failures": []}
