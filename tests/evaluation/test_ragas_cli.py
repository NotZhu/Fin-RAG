from __future__ import annotations

import json

from finrag.evaluation.ragas_runner import RagasEvaluationError
from scripts.evaluate_demo_documents_ragas import main


def test_ragas_cli_dry_run_writes_sample_preview(tmp_path):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "name": "demo-documents",
                "knowledge_base_id": "demo-documents",
                "top_k": 5,
                "quality_gates": {},
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
    report_path = tmp_path / "latest.json"
    report_path.write_text(
        json.dumps(
            {
                "qa": {
                    "cases": [
                        {
                            "id": "cash-flow",
                            "question": "经营现金流是多少？",
                            "result": {
                                "answer": "经营现金流为 0.8 亿元。",
                                "sources": [{"filename": "credit_review_report.pdf", "snippet": "经营现金流为 0.8 亿元。"}],
                            },
                        }
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "ragas-preview.json"

    code = main(
        [
            "--suite",
            str(suite_path),
            "--from-report",
            str(report_path),
            "--report-output",
            str(output_path),
            "--dry-run",
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["summary"] == {"case_count": 1, "qa_strategy": "ragas_sample_preview"}
    assert payload["cases"][0]["sample"]["reference"] == "经营现金流为 0.8 亿元。"


def test_ragas_cli_does_not_overwrite_existing_report_when_evaluation_fails(monkeypatch, tmp_path):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "name": "demo-documents",
                "knowledge_base_id": "demo-documents",
                "top_k": 5,
                "quality_gates": {},
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
    source_report_path = tmp_path / "latest.json"
    source_report_path.write_text(
        json.dumps(
            {
                "qa": {
                    "cases": [
                        {
                            "id": "cash-flow",
                            "question": "经营现金流是多少？",
                            "result": {
                                "answer": "经营现金流为 0.8 亿元。",
                                "sources": [{"filename": "credit_review_report.pdf", "snippet": "经营现金流为 0.8 亿元。"}],
                            },
                        }
                    ]
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "ragas-latest.json"
    output_path.write_text('{"metrics":{"faithfulness":1.0}}', encoding="utf-8")
    monkeypatch.setattr(
        "scripts.evaluate_demo_documents_ragas.run_ragas_samples",
        lambda samples: (_ for _ in ()).throw(RagasEvaluationError("Ragas 没有产生任何有效指标")),
    )

    code = main(
        [
            "--suite",
            str(suite_path),
            "--from-report",
            str(source_report_path),
            "--report-output",
            str(output_path),
        ]
    )

    assert code == 2
    assert output_path.read_text(encoding="utf-8") == '{"metrics":{"faithfulness":1.0}}'
