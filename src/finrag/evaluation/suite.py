"""demo-documents 确定性评测套件加载逻辑"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

from finrag.core.config import PROJECT_ROOT

DEFAULT_SUITE_PATH = PROJECT_ROOT / "datasets" / "eval" / "demo_documents_suite.json"
DEFAULT_KNOWLEDGE_BASE_ID = "demo-documents"
DEFAULT_DATA_PATH = PROJECT_ROOT / "output"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "eval" / DEFAULT_KNOWLEDGE_BASE_ID


@dataclass(frozen=True)
class EvalSuite:
    """已校验的 demo-documents 评测套件"""

    name: str
    knowledge_base_id: str
    top_k: int
    quality_gates: Dict[str, Any]
    cases: List[Dict[str, Any]]


def load_eval_suite(path: Path = DEFAULT_SUITE_PATH) -> EvalSuite:
    """加载并校验 JSON 评测套件"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    name = str(payload.get("name") or DEFAULT_KNOWLEDGE_BASE_ID)
    knowledge_base_id = str(payload.get("knowledge_base_id") or DEFAULT_KNOWLEDGE_BASE_ID)
    top_k = max(int(payload.get("top_k") or 5), 1)
    cases = list(payload.get("cases") or [])
    if not cases:
        raise ValueError(f"{path} 缺少 cases")
    validated_cases = [_validate_case(path, index, case) for index, case in enumerate(cases, 1)]
    return EvalSuite(
        name=name,
        knowledge_base_id=knowledge_base_id,
        top_k=top_k,
        quality_gates=dict(payload.get("quality_gates") or {}),
        cases=validated_cases,
    )


def _validate_case(path: Path, index: int, case: Mapping[str, Any]) -> Dict[str, Any]:
    case_id = str(case.get("id") or "").strip()
    question = str(case.get("question") or "").strip()
    if not case_id or not question:
        raise ValueError(f"{path}:cases[{index}] 缺少 id/question")
    relevant_sources = list(case.get("relevant_sources") or [])
    if not relevant_sources:
        raise ValueError(f"{path}:cases[{index}] relevant_sources 不能为空")
    for source in relevant_sources:
        if not isinstance(source, Mapping) or not str(source.get("filename") or "").strip():
            raise ValueError(f"{path}:cases[{index}] relevant_sources 缺少 filename")
    if not list(case.get("answer_keywords") or []):
        raise ValueError(f"{path}:cases[{index}] answer_keywords 不能为空")
    if not list(case.get("source_keywords") or []):
        raise ValueError(f"{path}:cases[{index}] source_keywords 不能为空")
    normalized = dict(case)
    normalized["id"] = case_id
    normalized["question"] = question
    normalized["relevant_sources"] = [dict(source) for source in relevant_sources]
    normalized["answer_keywords"] = [str(keyword) for keyword in case.get("answer_keywords", [])]
    normalized["source_keywords"] = [str(keyword) for keyword in case.get("source_keywords", [])]
    return normalized
