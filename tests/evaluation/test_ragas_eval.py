import builtins
import sys
from types import ModuleType

import pytest

from scripts.evaluate_ragas import build_ragas_rows_from_finrag, load_ragas_rows, run_ragas, save_ragas_report


def test_load_ragas_rows_validates_required_fields(tmp_path):
    dataset = tmp_path / "ragas.jsonl"
    dataset.write_text(
        '{"question":"q","answer":"a","contexts":["c"],"ground_truth":"g"}\n',
        encoding="utf-8",
    )

    rows = load_ragas_rows(dataset)

    assert rows == [{"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "g"}]


def test_load_ragas_rows_rejects_missing_contexts(tmp_path):
    dataset = tmp_path / "ragas.jsonl"
    dataset.write_text('{"question":"q","answer":"a","ground_truth":"g"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="contexts"):
        load_ragas_rows(dataset)


def test_build_ragas_rows_from_finrag_uses_sources_and_ground_truth():
    class FakeResponse:
        answer = "客户风险等级应与产品匹配[1]"
        sources = [type("Source", (), {"snippet": "客户风险等级应与产品风险等级匹配"})()]

    class FakeSystem:
        def ask_question(self, question, **kwargs):
            return FakeResponse()

    cases = [{"question": "客户风险等级如何匹配？", "ground_truth": "客户风险等级应与产品风险等级匹配"}]

    rows = build_ragas_rows_from_finrag(cases, FakeSystem())

    assert rows == [
        {
            "question": "客户风险等级如何匹配？",
            "answer": "客户风险等级应与产品匹配[1]",
            "contexts": ["客户风险等级应与产品风险等级匹配"],
            "ground_truth": "客户风险等级应与产品风险等级匹配",
        }
    ]


def test_save_ragas_report_writes_json(tmp_path):
    output = save_ragas_report([{"faithfulness": 1.0}], output_dir=tmp_path)

    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("[")


def test_run_ragas_requires_official_dependencies(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "datasets" or name == "ragas" or name.startswith("ragas."):
            raise ImportError("missing official evaluator")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="missing official evaluator"):
        run_ragas([{"question": "q", "answer": "a", "contexts": ["a"], "ground_truth": "a"}])


def test_run_ragas_propagates_official_evaluator_errors(monkeypatch):
    datasets_module = ModuleType("datasets")

    class FakeDataset:
        @classmethod
        def from_list(cls, rows):
            return rows

    datasets_module.Dataset = FakeDataset

    ragas_module = ModuleType("ragas")

    def fake_evaluate(*args, **kwargs):
        raise RuntimeError("ragas failed")

    ragas_module.evaluate = fake_evaluate
    metrics_module = ModuleType("ragas.metrics")
    metrics_module.answer_relevancy = object()
    metrics_module.context_precision = object()
    metrics_module.context_recall = object()
    metrics_module.faithfulness = object()

    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "ragas", ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics_module)

    with pytest.raises(RuntimeError, match="ragas failed"):
        run_ragas([{"question": "q", "answer": "a", "contexts": ["a"], "ground_truth": "a"}])
