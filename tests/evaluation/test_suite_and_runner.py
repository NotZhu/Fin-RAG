from __future__ import annotations

import json
from dataclasses import dataclass

from finrag.evaluation.runner import evaluate_qa_cases, save_report
from finrag.evaluation.suite import DEFAULT_SUITE_PATH, load_eval_suite


def test_load_eval_suite_validates_demo_document_cases(tmp_path):
    suite_path = tmp_path / "demo_documents_suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "name": "demo-documents",
                "knowledge_base_id": "demo-documents",
                "top_k": 5,
                "quality_gates": {
                    "recall_at_5": 0.8,
                    "mrr": 0.6,
                    "ndcg_at_5": 0.65,
                    "source_keyword_coverage": 0.75,
                    "answer_keyword_coverage": 0.75,
                    "empty_retrieval_rate": {"max": 0.05},
                },
                "cases": [
                    {
                        "id": "cash-flow",
                        "question": "经营现金流是多少？",
                        "expected_answer": "经营现金流为 0.8 亿元。",
                        "answer_keywords": ["0.8 亿元"],
                        "source_keywords": ["经营现金流"],
                        "relevant_sources": [{"filename": "credit_review_report.pdf", "relevance": 3}],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    suite = load_eval_suite(suite_path)

    assert suite.name == "demo-documents"
    assert suite.knowledge_base_id == "demo-documents"
    assert suite.top_k == 5
    assert suite.cases[0]["id"] == "cash-flow"
    assert suite.cases[0]["relevant_sources"][0]["filename"] == "credit_review_report.pdf"


def test_load_eval_suite_rejects_cases_without_relevant_sources(tmp_path):
    suite_path = tmp_path / "bad_suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "name": "demo-documents",
                "knowledge_base_id": "demo-documents",
                "cases": [
                    {
                        "id": "bad",
                        "question": "缺少来源？",
                        "answer_keywords": ["关键词"],
                        "source_keywords": ["来源"],
                        "relevant_sources": [],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    try:
        load_eval_suite(suite_path)
    except ValueError as exc:
        assert "relevant_sources" in str(exc)
    else:
        raise AssertionError("load_eval_suite should reject empty relevant_sources")


def test_default_demo_suite_aligns_keywords_with_document_wording():
    suite = load_eval_suite(DEFAULT_SUITE_PATH)
    cases = {case["id"]: case for case in suite.cases}

    assert cases["collateral_ratio"]["source_keywords"] == ["核心客户应收账款池", "50%", "需月度核验"]
    assert "逾期整改" in cases["supplier_esg_action"]["question"]
    assert "整改责任矩阵" in cases["supplier_esg_action"]["question"]
    assert cases["supplier_esg_action"]["answer_keywords"] == [
        "维持合格供应商资格",
        "新增订单需绑定整改节点",
        "暂停新增采购订单",
        "责任部门",
    ]
    assert cases["supplier_esg_action"]["source_keywords"] == [
        "绑定采购准入",
        "整改责任矩阵",
        "绑定责任部门",
    ]


def test_qa_cases_score_answer_sources_route_empty_and_latency():
    @dataclass
    class Source:
        filename: str
        snippet: str
        score: float = 0.91

        def to_dict(self):
            return {"filename": self.filename, "snippet": self.snippet, "score": self.score}

    class Trace:
        route_type = "knowledge"
        final_decision = "generate"
        timings_ms = {"total": 42.5}

        def to_dict(self):
            return {
                "route_type": self.route_type,
                "final_decision": self.final_decision,
                "timings_ms": self.timings_ms,
            }

    class Response:
        answer = "经营现金流为 0.8 亿元，重度压力情景下暂停新增敞口。"
        route_type = "knowledge"
        trace = Trace()
        sources = [Source("credit_review_report.pdf", "经营现金流 0.8 亿元，暂停新增敞口")]

    cases = [
        {
            "id": "cash-flow",
            "question": "经营现金流和授信建议是什么？",
            "answer_keywords": ["0.8 亿元", "暂停新增敞口"],
            "source_keywords": ["经营现金流", "暂停新增敞口"],
            "relevant_sources": [{"filename": "credit_review_report.pdf", "relevance": 3}],
        }
    ]

    report = evaluate_qa_cases(cases, lambda case: Response(), strategy_name="full_router")

    assert report["summary"] == {"case_count": 1, "qa_strategy": "full_router"}
    assert report["metrics"]["answer_keyword_coverage"] == 1.0
    assert report["metrics"]["source_keyword_coverage"] == 1.0
    assert report["metrics"]["route_accuracy"] == 1.0
    assert report["metrics"]["empty_source_rate"] == 0.0
    assert report["metrics"]["avg_total_latency_ms"] == 42.5
    assert report["cases"][0]["result"]["sources"][0]["filename"] == "credit_review_report.pdf"


def test_save_report_uses_stable_default_names(monkeypatch, tmp_path):
    monkeypatch.setattr("finrag.evaluation.runner.DEFAULT_REPORT_DIR", tmp_path)

    full_report = {"summary": {"stages": ["retrieval", "qa"]}, "gate": {"passed": True}}
    retrieval_report = {"summary": {"stages": ["retrieval"]}, "gate": {"passed": True}}
    qa_report = {"summary": {"stages": ["qa"]}, "gate": {"passed": True}}

    assert save_report(full_report) == tmp_path / "latest.json"
    assert save_report(retrieval_report) == tmp_path / "retrieval-latest.json"
    assert save_report(qa_report) == tmp_path / "qa-latest.json"
    assert json.loads((tmp_path / "latest.json").read_text(encoding="utf-8")) == full_report
