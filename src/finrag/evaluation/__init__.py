"""FinRAG 的 demo-documents 评测辅助接口"""

from .metrics import (
    evaluate_retrieval_cases,
    evaluate_thresholds,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    text_keyword_coverage,
)
from .runner import evaluate_qa_cases, evaluate_suite, save_report
from .ragas_runner import RagasEvaluationError, build_ragas_sample_dicts, normalize_ragas_result, run_ragas_samples
from .suite import DEFAULT_SUITE_PATH, EvalSuite, load_eval_suite

__all__ = [
    "DEFAULT_SUITE_PATH",
    "EvalSuite",
    "evaluate_qa_cases",
    "evaluate_retrieval_cases",
    "evaluate_suite",
    "evaluate_thresholds",
    "RagasEvaluationError",
    "build_ragas_sample_dicts",
    "hit_at_k",
    "load_eval_suite",
    "ndcg_at_k",
    "normalize_ragas_result",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "run_ragas_samples",
    "save_report",
    "text_keyword_coverage",
]
