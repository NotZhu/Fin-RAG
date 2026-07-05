"""demo-documents RAG 评测套件使用的确定性指标"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from finrag.core.node_schema import NodeWithScore, TextNode

DEFAULT_RETRIEVAL_STRATEGY = "raw_hybrid"


def _coerce_node(item: TextNode | NodeWithScore) -> TextNode:
    return item.node if isinstance(item, NodeWithScore) else item


def source_name(value: Any) -> str:
    """返回用于指标匹配的稳定来源文件名"""
    return Path(str(value or "").strip()).name


def extract_source_names(nodes: Iterable[TextNode | NodeWithScore]) -> List[str]:
    """按检索顺序提取去重后的来源文件名"""
    names: List[str] = []
    seen: set[str] = set()
    for item in nodes:
        node = _coerce_node(item)
        metadata = node.metadata or {}
        name = source_name(metadata.get("filename") or metadata.get("title_hint") or "unknown")
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def extract_chunk_ids(nodes: Iterable[TextNode | NodeWithScore]) -> List[str]:
    """按检索顺序提取分块标识"""
    chunk_ids: List[str] = []
    for item in nodes:
        node = _coerce_node(item)
        metadata = node.metadata or {}
        chunk_ids.append(str(metadata.get("chunk_id") or node.node_id))
    return chunk_ids


def text_keyword_coverage(text: str, expected_keywords: Sequence[str]) -> float:
    """计算普通文本中的关键词覆盖率"""
    keywords = [str(keyword) for keyword in expected_keywords if str(keyword)]
    if not keywords:
        return 0.0
    return sum(1 for keyword in keywords if keyword in (text or "")) / len(keywords)


def source_keyword_coverage(nodes: Iterable[TextNode | NodeWithScore], expected_keywords: Sequence[str]) -> float:
    """计算检索来源文本中的关键词覆盖率"""
    combined = "\n".join(_coerce_node(item).text for item in nodes)
    return text_keyword_coverage(combined, expected_keywords)


def relevant_source_weights(case: Mapping[str, Any]) -> Dict[str, float]:
    """返回单条用例的来源文件名到分级相关性的映射"""
    weights: Dict[str, float] = {}
    for source in case.get("relevant_sources", []):
        if not isinstance(source, Mapping):
            continue
        name = source_name(source.get("filename"))
        if not name:
            continue
        weights[name] = max(float(source.get("relevance", 1) or 1), weights.get(name, 0.0))
    return weights


def recall_at_k(ranked_sources: Sequence[str], relevant_sources: Mapping[str, float], k: int) -> float:
    if not relevant_sources:
        return 0.0
    hits = {source for source in ranked_sources[:k] if source in relevant_sources}
    return len(hits) / len(relevant_sources)


def precision_at_k(ranked_sources: Sequence[str], relevant_sources: Mapping[str, float], k: int) -> float:
    top_k = list(ranked_sources[:k])
    if not top_k:
        return 0.0
    return sum(1 for source in top_k if source in relevant_sources) / len(top_k)


def hit_at_k(ranked_sources: Sequence[str], relevant_sources: Mapping[str, float], k: int) -> float:
    return 1.0 if any(source in relevant_sources for source in ranked_sources[:k]) else 0.0


def reciprocal_rank(ranked_sources: Sequence[str], relevant_sources: Mapping[str, float]) -> float:
    for rank, source in enumerate(ranked_sources, 1):
        if source in relevant_sources:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_sources: Sequence[str], relevant_sources: Mapping[str, float], k: int) -> float:
    if not relevant_sources:
        return 0.0
    gains = [float(relevant_sources.get(source, 0.0)) for source in ranked_sources[:k]]
    dcg = _dcg(gains)
    ideal = sorted((float(value) for value in relevant_sources.values()), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def _dcg(gains: Sequence[float]) -> float:
    return sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))


def evaluate_retrieval_cases(
    cases: Sequence[Mapping[str, Any]],
    search_fn: Callable[[str, int], Sequence[TextNode | NodeWithScore]],
    top_k: int = 5,
    *,
    strategy_name: str = DEFAULT_RETRIEVAL_STRATEGY,
    latency_fn: Callable[[str, int, Sequence[TextNode | NodeWithScore]], float] | None = None,
) -> Dict[str, Any]:
    """评测检索召回、排序、来源覆盖率、空召回率和延迟"""
    top_k = max(int(top_k), 1)
    metric_sums = {
        f"recall_at_{top_k}": 0.0,
        f"precision_at_{top_k}": 0.0,
        f"hit_at_{top_k}": 0.0,
        "mrr": 0.0,
        f"ndcg_at_{top_k}": 0.0,
        "source_keyword_coverage": 0.0,
        "empty_retrieval_rate": 0.0,
        "avg_latency_ms": 0.0,
    }
    reports: List[Dict[str, Any]] = []
    for case in cases:
        question = str(case["question"])
        started = time.perf_counter()
        results = list(search_fn(question, top_k))
        measured_latency = (time.perf_counter() - started) * 1000
        latency_ms = float(latency_fn(question, top_k, results)) if latency_fn is not None else measured_latency
        ranked_sources = extract_source_names(results)
        relevant = relevant_source_weights(case)
        hit_sources = [source for source in ranked_sources[:top_k] if source in relevant]
        recall = recall_at_k(ranked_sources, relevant, top_k)
        precision = precision_at_k(ranked_sources, relevant, top_k)
        hit = hit_at_k(ranked_sources, relevant, top_k)
        mrr = reciprocal_rank(ranked_sources, relevant)
        ndcg = ndcg_at_k(ranked_sources, relevant, top_k)
        coverage = source_keyword_coverage(results, case.get("source_keywords", []))
        empty = 1.0 if not results else 0.0
        metric_sums[f"recall_at_{top_k}"] += recall
        metric_sums[f"precision_at_{top_k}"] += precision
        metric_sums[f"hit_at_{top_k}"] += hit
        metric_sums["mrr"] += mrr
        metric_sums[f"ndcg_at_{top_k}"] += ndcg
        metric_sums["source_keyword_coverage"] += coverage
        metric_sums["empty_retrieval_rate"] += empty
        metric_sums["avg_latency_ms"] += latency_ms
        reports.append(
            {
                "id": case["id"],
                "question": question,
                "relevant_sources": [dict(source) for source in case.get("relevant_sources", [])],
                "source_keywords": list(case.get("source_keywords", [])),
                "result": {
                    "ranked_sources": ranked_sources,
                    "ranked_chunk_ids": extract_chunk_ids(results),
                    "hit_sources": hit_sources,
                    f"recall_at_{top_k}": round(recall, 4),
                    f"precision_at_{top_k}": round(precision, 4),
                    f"hit_at_{top_k}": round(hit, 4),
                    "mrr": round(mrr, 4),
                    f"ndcg_at_{top_k}": round(ndcg, 4),
                    "source_keyword_coverage": round(coverage, 4),
                    "empty_retrieval": bool(empty),
                    "latency_ms": round(latency_ms, 2),
                },
            }
        )
    count = len(cases)
    return {
        "summary": {"case_count": count, "top_k": top_k, "retrieval_strategy": strategy_name},
        "metrics": {metric: round(value / count, 4) if count else 0.0 for metric, value in metric_sums.items()},
        "cases": reports,
    }


def evaluate_thresholds(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> Dict[str, Any]:
    """评估指标门禁；裸数字表示下限，{max: x} 表示上限"""
    failures: List[Dict[str, Any]] = []
    for metric, rule in thresholds.items():
        actual = float(metrics.get(metric, 0.0) or 0.0)
        if isinstance(rule, Mapping):
            if "max" in rule and actual > float(rule["max"]):
                failures.append({"metric": metric, "operator": "<=", "expected": rule["max"], "actual": actual})
            if "min" in rule and actual < float(rule["min"]):
                failures.append({"metric": metric, "operator": ">=", "expected": rule["min"], "actual": actual})
        elif actual < float(rule):
            failures.append({"metric": metric, "operator": ">=", "expected": rule, "actual": actual})
    return {"passed": not failures, "failures": failures}
