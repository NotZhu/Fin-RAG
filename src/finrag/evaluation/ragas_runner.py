"""基于已有 demo-documents 报告的可选 Ragas 语义评测"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from finrag.evaluation.suite import DEFAULT_KNOWLEDGE_BASE_ID, EvalSuite

DEFAULT_RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_recall")
DEFAULT_RAGAS_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class RagasDependencyError(RuntimeError):
    """Ragas 可选依赖未安装或评测 LLM 依赖不可用"""


class RagasEvaluationError(RuntimeError):
    """Ragas 已运行但没有得到可用评测分数"""


@dataclass(frozen=True)
class RagasEvalSettings:
    """Ragas 语义评测运行配置"""

    metrics: Sequence[str] = DEFAULT_RAGAS_METRICS
    llm_model: str = "qwen3.7-max"
    api_key: str = ""
    base_url: str = DEFAULT_RAGAS_BASE_URL
    embedding_model: str = "BAAI/bge-m3"
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    answer_relevancy_strictness: int = 1

    @classmethod
    def from_env(cls) -> "RagasEvalSettings":
        """从环境变量构建 Ragas 评测配置"""
        metrics = tuple(
            item.strip()
            for item in os.getenv("RAGAS_METRICS", ",".join(DEFAULT_RAGAS_METRICS)).split(",")
            if item.strip()
        )
        return cls(
            metrics=metrics or DEFAULT_RAGAS_METRICS,
            llm_model=os.getenv("RAGAS_LLM_MODEL") or os.getenv("RAG_LLM_MODEL") or "qwen3.7-max",
            api_key=os.getenv("RAGAS_API_KEY") or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
            base_url=os.getenv("RAGAS_BASE_URL") or os.getenv("OPENAI_BASE_URL") or DEFAULT_RAGAS_BASE_URL,
            embedding_model=os.getenv("RAGAS_EMBEDDING_MODEL") or os.getenv("RAG_EMBEDDING_MODEL") or "BAAI/bge-m3",
            embedding_api_key=os.getenv("RAGAS_EMBEDDING_API_KEY") or os.getenv("EMBEDDING_API_KEY") or "",
            embedding_base_url=os.getenv("RAGAS_EMBEDDING_BASE_URL") or os.getenv("EMBEDDING_BASE_URL") or "",
            answer_relevancy_strictness=_env_int("RAGAS_ANSWER_RELEVANCY_STRICTNESS", 1),
        )


def build_ragas_sample_dicts(
    report: Mapping[str, Any],
    suite: EvalSuite,
    *,
    max_cases: int | None = None,
) -> List[Dict[str, Any]]:
    """将确定性 QA 报告转换为 Ragas 单轮评测样本"""
    qa_cases = list(((report.get("qa") or {}).get("cases") or []))
    if not qa_cases:
        raise ValueError("Ragas 评测需要包含 report.qa.cases 的完整 QA 报告")
    expected_by_id = {str(case["id"]): case for case in suite.cases}
    samples: List[Dict[str, Any]] = []
    for qa_case in qa_cases:
        case_id = str(qa_case.get("id") or "").strip()
        expected_case = expected_by_id.get(case_id)
        if expected_case is None:
            raise ValueError(f"报告中的 case {case_id!r} 不存在于评测 suite")
        reference = str(expected_case.get("expected_answer") or "").strip()
        if not reference:
            raise ValueError(f"评测 suite 中 case {case_id!r} 缺少 expected_answer")
        result = dict(qa_case.get("result") or {})
        contexts, source_filenames = _source_contexts(result.get("sources") or [])
        samples.append(
            {
                "user_input": str(qa_case.get("question") or expected_case.get("question") or "").strip(),
                "response": str(result.get("answer") or "").strip(),
                "retrieved_contexts": contexts,
                "reference": reference,
                "metadata": {
                    "case_id": case_id,
                    "knowledge_base_id": suite.knowledge_base_id or DEFAULT_KNOWLEDGE_BASE_ID,
                    "source_filenames": source_filenames,
                },
            }
        )
        if max_cases is not None and len(samples) >= max(int(max_cases), 0):
            break
    if not samples:
        raise ValueError("Ragas 评测没有可用样本")
    return samples


def run_ragas_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    settings: RagasEvalSettings | None = None,
) -> Dict[str, Any]:
    """运行 Ragas 语义评测并返回统一 JSON 报告"""
    resolved = settings or RagasEvalSettings.from_env()
    try:
        components = _load_ragas_components()
    except ImportError as exc:
        raise RagasDependencyError("缺少 Ragas 可选依赖，请先运行 uv sync --extra eval") from exc
    dataset = _build_ragas_dataset(components["EvaluationDataset"], samples)
    evaluator_llm = _build_evaluator_llm(resolved)
    evaluator_embeddings = _build_evaluator_embeddings(resolved)
    metrics = _build_metric_objects(
        components,
        resolved.metrics,
        evaluator_llm=evaluator_llm,
        evaluator_embeddings=evaluator_embeddings,
        settings=resolved,
    )
    result = components["evaluate"](dataset, metrics=metrics)
    report = normalize_ragas_result(result, samples, expected_metrics=resolved.metrics)
    report["summary"]["metrics"] = list(resolved.metrics)
    report["summary"]["llm_model"] = resolved.llm_model
    report["summary"]["embedding_model"] = resolved.embedding_model
    _ensure_report_has_scored_metrics(report)
    return report


def normalize_ragas_result(
    result: Any,
    samples: Sequence[Mapping[str, Any]],
    *,
    expected_metrics: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """将不同版本 Ragas 返回值归一化为稳定 JSON 结构"""
    records = _result_records(result, len(samples))
    metric_names = _merge_metric_names(expected_metrics or [], _metric_names(records))
    case_reports = []
    metric_values: Dict[str, List[float]] = {name: [] for name in metric_names}
    for index, sample in enumerate(samples):
        record = records[index] if index < len(records) else {}
        case_metrics: Dict[str, float] = {}
        missing_metrics: List[str] = []
        for name in metric_names:
            value = _numeric(record.get(name))
            if value is None:
                missing_metrics.append(name)
                continue
            rounded = round(value, 4)
            case_metrics[name] = rounded
            metric_values[name].append(value)
        metadata = dict(sample.get("metadata") or {})
        case_report = {
            "id": metadata.get("case_id", str(index + 1)),
            "question": sample.get("user_input", ""),
            "metrics": case_metrics,
        }
        if missing_metrics:
            case_report["missing_metrics"] = missing_metrics
        case_reports.append(case_report)
    total = len(samples)
    return {
        "summary": {"case_count": len(samples), "qa_strategy": "ragas_semantic"},
        "metrics": {
            name: round(sum(values) / len(values), 4)
            for name, values in metric_values.items()
            if values
        },
        "metric_coverage": {
            name: {"scored": len(values), "missing": total - len(values), "total": total}
            for name, values in metric_values.items()
        },
        "cases": case_reports,
    }


def _source_contexts(sources: Sequence[Any]) -> tuple[List[str], List[str]]:
    contexts: List[str] = []
    filenames: List[str] = []
    for source in sources:
        source_dict = dict(source) if isinstance(source, Mapping) else {}
        snippet = str(source_dict.get("snippet") or "").strip()
        if not snippet:
            continue
        contexts.append(snippet)
        filename = str(source_dict.get("filename") or "").strip()
        if filename:
            filenames.append(filename)
    return contexts, filenames


def _load_ragas_components() -> Dict[str, Any]:
    from ragas import EvaluationDataset, evaluate

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy

    return {
        "EvaluationDataset": EvaluationDataset,
        "evaluate": evaluate,
        "faithfulness": Faithfulness,
        "answer_relevancy": ResponseRelevancy,
        "response_relevancy": ResponseRelevancy,
        "context_recall": LLMContextRecall,
    }


def _build_ragas_dataset(dataset_class: Any, samples: Sequence[Mapping[str, Any]]) -> Any:
    rows = [dict(sample) for sample in samples]
    from_list = getattr(dataset_class, "from_list", None)
    if callable(from_list):
        return from_list(rows)
    return dataset_class(rows)


def _build_evaluator_llm(settings: RagasEvalSettings) -> Any:
    if not settings.api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RagasDependencyError("缺少 langchain-openai，请先运行 uv sync --extra eval") from exc
    try:
        from ragas.llms import LangchainLLMWrapper
    except ImportError:
        from ragas.llms.base import LangchainLLMWrapper

    chat = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.api_key,
        base_url=settings.base_url or None,
        temperature=0,
    )
    return LangchainLLMWrapper(chat)


def _build_evaluator_embeddings(settings: RagasEvalSettings) -> Any:
    if not settings.embedding_base_url or not settings.embedding_api_key:
        return None
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise RagasDependencyError("缺少 langchain-openai，请先运行 uv sync --extra eval") from exc
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError:
        from ragas.embeddings.base import LangchainEmbeddingsWrapper

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        tiktoken_enabled=False,
    )
    return LangchainEmbeddingsWrapper(embeddings)


def _build_metric_objects(
    components: Mapping[str, Any],
    metric_names: Sequence[str],
    *,
    evaluator_llm: Any,
    evaluator_embeddings: Any,
    settings: RagasEvalSettings,
) -> List[Any]:
    metrics: List[Any] = []
    for name in metric_names:
        key = str(name).strip()
        factory = components.get(key)
        if factory is None:
            raise ValueError(f"不支持的 Ragas 指标: {key}")
        if key in {"answer_relevancy", "response_relevancy"}:
            metrics.append(_instantiate_answer_relevancy_metric(factory, evaluator_llm, evaluator_embeddings, settings))
            continue
        metrics.append(_instantiate_metric(factory, evaluator_llm))
    return metrics


def _instantiate_answer_relevancy_metric(
    factory: Any,
    evaluator_llm: Any,
    evaluator_embeddings: Any,
    settings: RagasEvalSettings,
) -> Any:
    if evaluator_embeddings is None:
        raise RagasDependencyError(
            "Ragas answer_relevancy 需要显式配置 RAGAS_EMBEDDING_BASE_URL/RAGAS_EMBEDDING_API_KEY，"
            "也可复用 EMBEDDING_BASE_URL/EMBEDDING_API_KEY；否则 Ragas 可能回退到 OpenAI 官方 embedding。"
        )
    strictness = max(int(settings.answer_relevancy_strictness), 1)
    if not callable(factory):
        metric = factory
        if hasattr(metric, "llm"):
            metric.llm = evaluator_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = evaluator_embeddings
        if hasattr(metric, "strictness"):
            metric.strictness = strictness
        return metric
    try:
        return factory(llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=strictness)
    except TypeError:
        metric = _instantiate_metric(factory, evaluator_llm)
        if hasattr(metric, "embeddings"):
            metric.embeddings = evaluator_embeddings
        if hasattr(metric, "strictness"):
            metric.strictness = strictness
        return metric


def _instantiate_metric(factory: Any, evaluator_llm: Any) -> Any:
    if not callable(factory):
        return factory
    if evaluator_llm is not None:
        try:
            return factory(llm=evaluator_llm)
        except TypeError:
            metric = factory()
            if hasattr(metric, "llm"):
                metric.llm = evaluator_llm
            return metric
    return factory()


def _result_records(result: Any, sample_count: int) -> List[Dict[str, Any]]:
    if isinstance(result, Mapping):
        return _records_from_mapping(result, sample_count)
    to_pandas = getattr(result, "to_pandas", None)
    if callable(to_pandas):
        frame = to_pandas()
        to_dict = getattr(frame, "to_dict", None)
        if callable(to_dict):
            return [dict(row) for row in to_dict(orient="records")]
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        return _records_from_mapping(to_dict(), sample_count)
    return []


def _records_from_mapping(payload: Mapping[str, Any], sample_count: int) -> List[Dict[str, Any]]:
    if "cases" in payload and isinstance(payload["cases"], Sequence):
        return [dict(row) for row in payload["cases"] if isinstance(row, Mapping)]
    records = [dict() for _ in range(sample_count)]
    for key, value in payload.items():
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, item in enumerate(value[:sample_count]):
                records[index][key] = item
        elif sample_count == 1:
            records[0][key] = value
    return records


def _metric_names(records: Sequence[Mapping[str, Any]]) -> List[str]:
    ignored = {"user_input", "response", "retrieved_contexts", "reference", "metadata"}
    names: List[str] = []
    for record in records:
        for key, value in record.items():
            if key in ignored or _numeric(value) is None or key in names:
                continue
            names.append(str(key))
    return names


def _merge_metric_names(expected: Sequence[str], actual: Sequence[str]) -> List[str]:
    names: List[str] = []
    for source in (expected, actual):
        for name in source:
            key = str(name).strip()
            if key and key not in names:
                names.append(key)
    return names


def _ensure_report_has_scored_metrics(report: Mapping[str, Any]) -> None:
    coverage = report.get("metric_coverage") or {}
    scored = 0
    for item in coverage.values():
        if isinstance(item, Mapping):
            scored += int(item.get("scored") or 0)
    if scored > 0:
        return
    raise RagasEvaluationError(
        "Ragas 没有产生任何有效指标；请检查评测 LLM/embedding 配置、API 额度或服务错误，"
        "本次不会写入正式 Ragas 报告。"
    )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number
