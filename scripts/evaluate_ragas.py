"""FinRAG 答案的可选 Ragas 评估入口

该入口保持可选：核心检索测试不依赖 Ragas，生产评估可安装声明的依赖，
并基于包含 question、answer、contexts、ground_truth 字段的 JSONL 用例运行
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def load_ragas_rows(path: Path) -> List[Dict[str, Any]]:
    """
    从 JSONL 文件加载 Ragas 评估输入
    Args:
        path: 评估集文件路径
    Returns:
        评估输入行列表
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # 解析 JSON 行
            row = json.loads(line)
            missing = {"question", "answer", "contexts", "ground_truth"} - set(row)
            if missing:
                raise ValueError(f"{path}:{line_number} 缺少字段: {sorted(missing)}")
            rows.append(row)
    return rows


def build_ragas_rows_from_finrag(cases: List[Dict[str, Any]], system: Any) -> List[Dict[str, Any]]:
    """
    调用 FinRAG 系统生成 Ragas 所需的答案和上下文行
    Args:
        cases: 包含 question 和期望答案字段的用例列表
        system: FinRAGSystem 或兼容对象
    Returns:
        Ragas 评估输入行列表
    """
    rows = []
    for case in cases:
        question = case["question"]
        response = system.ask_question(question, return_sources=True, return_trace=True)
        sources = getattr(response, "sources", []) or []
        rows.append(
            {
                "question": question,
                "answer": getattr(response, "answer", ""),
                "contexts": [source.snippet for source in sources if getattr(source, "snippet", "")],
                "ground_truth": case.get("ground_truth") or case.get("expected_answer") or "",
            }
        )
    return rows


def run_ragas(rows: List[Dict[str, Any]]) -> Any:
    """
    运行 Ragas 官方评估
    Args:
        rows: Ragas 评估输入行
    Returns:
        Ragas 结果对象
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    # 从 Ragas 评估输入行创建 Ragas 数据集
    dataset = Dataset.from_list(rows)
    # 运行 Ragas 评估
    return evaluate(
        dataset,
        # 评估指标，包括答案相关性、上下文精度、上下文召回率和答案真实性
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )


def save_ragas_report(rows: Any, output_dir: Path = Path("reports") / "ragas") -> Path:
    """
    将 Ragas 或本地评估报告保存为 JSON 文件
    Args:
        rows: Ragas 结果对象或普通行列表
        output_dir: 报告输出目录
    Returns:
        写出的报告文件路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-ragas.json"
    if hasattr(rows, "to_pandas"):
        payload = json.loads(rows.to_pandas().to_json(force_ascii=False, orient="records"))
    else:
        payload = rows
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main():
    """
    解析命令行参数并运行 Ragas 评估
    """
    parser = argparse.ArgumentParser(description="运行 FinRAG Ragas 评估")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--generate", action="store_true", help="从 FinRAG 自动生成 answer/context 后再评估")
    parser.add_argument("--save-report", action="store_true", help="保存 JSON 报告到 reports/ragas")
    args = parser.parse_args()

    # 加载 Ragas 评估输入行
    rows = load_ragas_rows(args.dataset)
    # 如果指定了生成 answer/context 后再评估，调用 FinRAG 系统
    if args.generate:
        from finrag.application.system import FinRAGSystem
        rows = build_ragas_rows_from_finrag(rows, FinRAGSystem())
    result = run_ragas(rows)
    # 如果指定了保存报告路径，写出 JSON 文件
    if args.save_report:
        save_ragas_report(result)
    # 打印评估结果
    print(result.to_pandas().to_json(force_ascii=False, orient="records") if args.json else result)


if __name__ == "__main__":
    main()
