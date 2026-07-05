"""demo-documents 评测套件的执行流程"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

from llama_index.core.schema import QueryBundle

from finrag.application.system import FinRAGSystem
from finrag.core.config import RAGConfig
from finrag.evaluation.metrics import evaluate_retrieval_cases, evaluate_thresholds, text_keyword_coverage
from finrag.evaluation.suite import DEFAULT_DATA_PATH, DEFAULT_KNOWLEDGE_BASE_ID, DEFAULT_REPORT_DIR, EvalSuite


def demo_config_from_env() -> RAGConfig:
    """构建固定指向 output/demo-documents 的评测配置"""
    config = RAGConfig.from_env()
    config.data_path = str(DEFAULT_DATA_PATH.resolve())
    config.knowledge_base_id = DEFAULT_KNOWLEDGE_BASE_ID
    return RAGConfig(**config.to_dict())


def build_demo_system(config: RAGConfig | None = None) -> FinRAGSystem:
    """创建并预热用于 demo-documents 评测的 FinRAGSystem"""
    resolved = config or demo_config_from_env()
    system = FinRAGSystem(resolved)
    system.ensure_knowledge_base_ready(resolved.knowledge_base_id)
    return system


def build_retrieval_search_fn(system: Any, knowledge_base_id: str) -> Callable[[str, int], Sequence[Any]]:
    """返回 demo 套件默认使用的原始混合检索函数"""

    def search(question: str, top_k: int) -> Sequence[Any]:
        retriever = getattr(system, "hybrid_retriever", None)
        if retriever is None:
            raise RuntimeError("FinRAG hybrid_retriever 尚未就绪")
        retriever.top_k = max(int(top_k), 1)
        retriever.filters = {"knowledge_base_id": knowledge_base_id}
        return retriever.retrieve(QueryBundle(question))

    return search


def evaluate_qa_cases(
    cases: Sequence[Mapping[str, Any]],
    ask_fn: Callable[[Mapping[str, Any]], Any],
    *,
    strategy_name: str = "full_router",
) -> Dict[str, Any]:
    """基于回答、来源片段、路由、空来源率和延迟评测 QA 质量"""
    metric_sums = {
        "answer_keyword_coverage": 0.0,
        "source_keyword_coverage": 0.0,
        "route_accuracy": 0.0,
        "empty_source_rate": 0.0,
        "avg_total_latency_ms": 0.0,
    }
    reports = []
    for case in cases:
        started = time.perf_counter()
        response = ask_fn(case)
        measured_latency = (time.perf_counter() - started) * 1000
        trace = _trace_to_dict(getattr(response, "trace", None))
        timings_ms = trace.get("timings_ms") or {}
        total_latency = float(timings_ms.get("total") or measured_latency)
        sources = [_source_to_dict(source) for source in (getattr(response, "sources", []) or [])]
        answer = str(getattr(response, "answer", "") or "")
        route_type = str(getattr(response, "route_type", "") or trace.get("route_type") or "")
        expected_route = str(case.get("expected_route_type") or "knowledge")
        answer_coverage = text_keyword_coverage(answer, case.get("answer_keywords", []))
        source_text = "\n".join(str(source.get("snippet") or "") for source in sources)
        source_coverage = text_keyword_coverage(source_text, case.get("source_keywords", []))
        route_accuracy = 1.0 if route_type == expected_route else 0.0
        empty_source = 1.0 if not sources else 0.0
        metric_sums["answer_keyword_coverage"] += answer_coverage
        metric_sums["source_keyword_coverage"] += source_coverage
        metric_sums["route_accuracy"] += route_accuracy
        metric_sums["empty_source_rate"] += empty_source
        metric_sums["avg_total_latency_ms"] += total_latency
        reports.append(
            {
                "id": case["id"],
                "question": case["question"],
                "result": {
                    "route_type": route_type,
                    "answer": answer,
                    "answer_keyword_coverage": round(answer_coverage, 4),
                    "source_keyword_coverage": round(source_coverage, 4),
                    "empty_source": bool(empty_source),
                    "sources": sources,
                    "total_latency_ms": round(total_latency, 2),
                },
            }
        )
    count = len(cases)
    return {
        "summary": {"case_count": count, "qa_strategy": strategy_name},
        "metrics": {metric: round(value / count, 4) if count else 0.0 for metric, value in metric_sums.items()},
        "cases": reports,
    }


def evaluate_suite(
    suite: EvalSuite,
    *,
    system: Any | None = None,
    stages: Sequence[str] = ("retrieval", "qa"),
) -> Dict[str, Any]:
    """运行选定的 demo-documents 确定性评测阶段"""
    resolved_system = system or build_demo_system()
    report: Dict[str, Any] = {
        "summary": {
            "suite": suite.name,
            "knowledge_base_id": suite.knowledge_base_id,
            "top_k": suite.top_k,
            "stages": list(stages),
        }
    }
    merged_metrics: Dict[str, Any] = {}
    if "retrieval" in stages:
        retrieval = evaluate_retrieval_cases(
            suite.cases,
            build_retrieval_search_fn(resolved_system, suite.knowledge_base_id),
            top_k=suite.top_k,
            strategy_name="raw_hybrid",
        )
        report["retrieval"] = retrieval
        merged_metrics.update(retrieval.get("metrics", {}))
    if "qa" in stages:
        qa = evaluate_qa_cases(
            suite.cases,
            lambda case: resolved_system.ask_question(
                str(case["question"]),
                knowledge_base_id=suite.knowledge_base_id,
                return_sources=True,
                return_trace=True,
            ),
        )
        report["qa"] = qa
        merged_metrics.update(qa.get("metrics", {}))
    report["gate"] = evaluate_thresholds(merged_metrics, suite.quality_gates)
    return report


def save_report(report: Mapping[str, Any], path: Path | None = None) -> Path:
    """将评测报告保存为 UTF-8 JSON"""
    output_path = path or _default_report_path(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _default_report_path(report: Mapping[str, Any]) -> Path:
    stages = set((report.get("summary") or {}).get("stages") or [])
    if stages == {"retrieval"}:
        return DEFAULT_REPORT_DIR / "retrieval-latest.json"
    if stages == {"qa"}:
        return DEFAULT_REPORT_DIR / "qa-latest.json"
    return DEFAULT_REPORT_DIR / "latest.json"


def _source_to_dict(source: Any) -> Dict[str, Any]:
    if hasattr(source, "to_dict"):
        return dict(source.to_dict())
    return {
        "filename": getattr(source, "filename", ""),
        "snippet": getattr(source, "snippet", ""),
        "score": getattr(source, "score", None),
    }


def _trace_to_dict(trace: Any) -> Dict[str, Any]:
    if trace is None:
        return {}
    if hasattr(trace, "to_dict"):
        return dict(trace.to_dict())
    if isinstance(trace, Mapping):
        return dict(trace)
    return {
        "route_type": getattr(trace, "route_type", ""),
        "final_decision": getattr(trace, "final_decision", ""),
        "timings_ms": getattr(trace, "timings_ms", {}) or {},
    }
