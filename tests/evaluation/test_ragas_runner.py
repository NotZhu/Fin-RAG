from __future__ import annotations

from finrag.evaluation.ragas_runner import (
    DEFAULT_RAGAS_BASE_URL,
    RagasDependencyError,
    RagasEvaluationError,
    RagasEvalSettings,
    build_ragas_sample_dicts,
    normalize_ragas_result,
    run_ragas_samples,
)
from finrag.evaluation.suite import EvalSuite


def _suite() -> EvalSuite:
    return EvalSuite(
        name="demo-documents",
        knowledge_base_id="demo-documents",
        top_k=5,
        quality_gates={},
        cases=[
            {
                "id": "cash-flow",
                "question": "经营现金流是多少？",
                "expected_answer": "经营现金流为 0.8 亿元。",
                "answer_keywords": ["0.8 亿元"],
                "source_keywords": ["经营现金流"],
                "relevant_sources": [{"filename": "credit_review_report.pdf", "relevance": 3}],
            },
            {
                "id": "drawdown",
                "question": "新增提款条件是什么？",
                "expected_answer": "需满足经营现金流转正。",
                "answer_keywords": ["经营现金流转正"],
                "source_keywords": ["新增提款"],
                "relevant_sources": [{"filename": "credit_review_report.pdf", "relevance": 3}],
            },
        ],
    )


def _report() -> dict:
    return {
        "summary": {"suite": "demo-documents", "knowledge_base_id": "demo-documents"},
        "qa": {
            "cases": [
                {
                    "id": "cash-flow",
                    "question": "经营现金流是多少？",
                    "result": {
                        "answer": "经营现金流为 0.8 亿元。",
                        "sources": [
                            {"filename": "credit_review_report.pdf", "snippet": "经营现金流为 0.8 亿元。"},
                            {"filename": "empty.txt", "snippet": ""},
                        ],
                    },
                },
                {
                    "id": "drawdown",
                    "question": "新增提款条件是什么？",
                    "result": {
                        "answer": "新增提款需绑定订单。",
                        "sources": [{"filename": "credit_review_report.pdf", "snippet": "新增提款需满足经营现金流转正。"}],
                    },
                },
            ]
        },
    }


def test_build_ragas_sample_dicts_uses_existing_qa_report_and_suite_answers():
    samples = build_ragas_sample_dicts(_report(), _suite(), max_cases=1)

    assert samples == [
        {
            "user_input": "经营现金流是多少？",
            "response": "经营现金流为 0.8 亿元。",
            "retrieved_contexts": ["经营现金流为 0.8 亿元。"],
            "reference": "经营现金流为 0.8 亿元。",
            "metadata": {
                "case_id": "cash-flow",
                "knowledge_base_id": "demo-documents",
                "source_filenames": ["credit_review_report.pdf"],
            },
        }
    ]


def test_build_ragas_sample_dicts_rejects_reports_without_qa_cases():
    try:
        build_ragas_sample_dicts({"retrieval": {"cases": []}}, _suite())
    except ValueError as exc:
        assert "qa.cases" in str(exc)
    else:
        raise AssertionError("missing qa.cases should be rejected")


def test_normalize_ragas_result_aggregates_metric_lists():
    samples = build_ragas_sample_dicts(_report(), _suite())
    raw_result = {
        "faithfulness": [1.0, 0.5],
        "answer_relevancy": [0.8, 0.6],
        "context_recall": [1.0, 1.0],
    }

    report = normalize_ragas_result(raw_result, samples)

    assert report["metrics"] == {
        "faithfulness": 0.75,
        "answer_relevancy": 0.7,
        "context_recall": 1.0,
    }
    assert report["cases"][0]["id"] == "cash-flow"
    assert report["cases"][0]["metrics"]["faithfulness"] == 1.0
    assert report["metric_coverage"] == {
        "faithfulness": {"scored": 2, "missing": 0, "total": 2},
        "answer_relevancy": {"scored": 2, "missing": 0, "total": 2},
        "context_recall": {"scored": 2, "missing": 0, "total": 2},
    }


def test_normalize_ragas_result_marks_missing_metrics_per_case():
    samples = build_ragas_sample_dicts(_report(), _suite())
    raw_result = {
        "faithfulness": [1.0, None],
        "context_recall": [1.0, 0.0],
    }

    report = normalize_ragas_result(raw_result, samples, expected_metrics=["faithfulness", "answer_relevancy", "context_recall"])

    assert report["metrics"] == {"faithfulness": 1.0, "context_recall": 0.5}
    assert report["metric_coverage"]["faithfulness"] == {"scored": 1, "missing": 1, "total": 2}
    assert report["metric_coverage"]["answer_relevancy"] == {"scored": 0, "missing": 2, "total": 2}
    assert report["cases"][1]["missing_metrics"] == ["faithfulness", "answer_relevancy"]


def test_ragas_settings_reads_evaluator_embedding_config_from_env(monkeypatch):
    monkeypatch.setenv("RAGAS_API_KEY", "llm-key")
    monkeypatch.setenv("RAGAS_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("RAGAS_LLM_MODEL", "qwen3.7-max")
    monkeypatch.setenv("RAGAS_EMBEDDING_API_KEY", "embed-key")
    monkeypatch.setenv("RAGAS_EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
    monkeypatch.setenv("RAGAS_EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("RAGAS_ANSWER_RELEVANCY_STRICTNESS", "2")

    settings = RagasEvalSettings.from_env()

    assert settings.api_key == "llm-key"
    assert settings.base_url == DEFAULT_RAGAS_BASE_URL
    assert settings.embedding_api_key == "embed-key"
    assert settings.embedding_base_url == "https://api.siliconflow.cn/v1"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.answer_relevancy_strictness == 2


def test_build_metric_objects_passes_embeddings_and_single_strictness_to_answer_relevancy():
    from finrag.evaluation.ragas_runner import _build_metric_objects

    captured = {}

    class AnswerRelevancyMetric:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class LlmOnlyMetric:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    metrics = _build_metric_objects(
        {
            "answer_relevancy": AnswerRelevancyMetric,
            "faithfulness": LlmOnlyMetric,
        },
        ["answer_relevancy", "faithfulness"],
        evaluator_llm="llm",
        evaluator_embeddings="embeddings",
        settings=RagasEvalSettings(answer_relevancy_strictness=1),
    )

    assert isinstance(metrics[0], AnswerRelevancyMetric)
    assert captured == {"llm": "llm", "embeddings": "embeddings", "strictness": 1}


def test_build_metric_objects_rejects_answer_relevancy_without_embeddings():
    from finrag.evaluation.ragas_runner import _build_metric_objects

    class AnswerRelevancyMetric:
        pass

    try:
        _build_metric_objects(
            {"answer_relevancy": AnswerRelevancyMetric},
            ["answer_relevancy"],
            evaluator_llm="llm",
            evaluator_embeddings=None,
            settings=RagasEvalSettings(),
        )
    except RagasDependencyError as exc:
        assert "RAGAS_EMBEDDING_BASE_URL" in str(exc)
    else:
        raise AssertionError("answer_relevancy should require explicit evaluator embeddings")


def test_run_ragas_samples_reports_clear_message_when_dependency_is_missing(monkeypatch):
    def missing_components():
        raise ImportError("No module named 'ragas'")

    monkeypatch.setattr("finrag.evaluation.ragas_runner._load_ragas_components", missing_components)

    try:
        run_ragas_samples(build_ragas_sample_dicts(_report(), _suite()))
    except RagasDependencyError as exc:
        assert "uv sync --extra eval" in str(exc)
    else:
        raise AssertionError("missing ragas dependency should raise RagasDependencyError")


def test_run_ragas_samples_rejects_reports_without_any_scored_metrics(monkeypatch):
    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            return rows

    class FakeMetric:
        pass

    def fake_evaluate(dataset, *, metrics):
        return {"faithfulness": [None for _ in dataset]}

    monkeypatch.setattr(
        "finrag.evaluation.ragas_runner._load_ragas_components",
        lambda: {"EvaluationDataset": FakeDataset, "evaluate": fake_evaluate, "faithfulness": FakeMetric},
    )

    try:
        run_ragas_samples(
            build_ragas_sample_dicts(_report(), _suite()),
            settings=RagasEvalSettings(metrics=["faithfulness"]),
        )
    except RagasEvaluationError as exc:
        assert "没有产生任何有效指标" in str(exc)
    else:
        raise AssertionError("empty Ragas metrics should fail the semantic evaluation")
