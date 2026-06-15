"""FinRAG 检索评估入口"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finrag.core.config import RAGConfig  # noqa: E402
from finrag.core.node_schema import NodeWithScore, TextNode  # noqa: E402
from finrag.application.system import FinRAGSystem  # noqa: E402

DEFAULT_EVAL_SET = PROJECT_ROOT / "datasets" / "eval" / "finance_smoke_eval_set.jsonl"
RETRIEVAL_STRATEGY = "milvus_hybrid_retriever"


def configure_utf8_stdio(stdout=None, stderr=None):
    """
    将标准输出和错误流配置为 UTF-8 编码
    Args:
        stdout: 可选标准输出流
        stderr: 可选标准错误流
    """
    import sys

    for stream in (stdout or sys.stdout, stderr or sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")


def load_eval_cases(path: Path) -> List[Dict[str, Any]]:
    """
    从 JSONL 文件加载检索评估用例
    Args:
        path: 评估集文件路径
    Returns:
        评估用例字典列表
    """
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            # 检查必填字段是否存在
            if not case.get("id") or not case.get("question") or not case.get("expected_doc_titles"):
                raise ValueError(f"{path}:{line_number} 缺少 id/question/expected_doc_titles")
            cases.append(case)
    return cases


def _coerce_node(item: TextNode | NodeWithScore) -> TextNode:
    """
    将节点或带分数节点统一转换为 TextNode
    Args:
        item: TextNode 或 NodeWithScore
    Returns:
        提取出的 TextNode
    """
    return item.node if isinstance(item, NodeWithScore) else item


def normalize_doc_title(title: str) -> str:
    """
    去除文档标题中的常见文件后缀
    Args:
        title: 原始标题或文件名
    Returns:
        规范化后的文档标题
    """
    suffix = Path(title).suffix.lower()
    if suffix in {".md", ".txt", ".pdf", ".docx"}:
        return title[: -len(suffix)]
    return title


def extract_doc_titles(nodes: Iterable[TextNode | NodeWithScore]) -> List[str]:
    """
    从检索结果中提取去重后的文档标题
    Args:
        nodes: 检索结果节点序列
    Returns:
        按首次出现顺序排列的文档标题列表
    """
    titles = []
    seen = set()
    for item in nodes:
        node = _coerce_node(item)
        metadata = node.metadata or {}
        title = normalize_doc_title(str(metadata.get("title_hint") or metadata.get("filename") or "未知文档"))
        if title not in seen:
            titles.append(title)
            seen.add(title)
    return titles


def extract_chunk_ids(nodes: Iterable[TextNode | NodeWithScore]) -> List[str]:
    """
    从检索结果中提取分块 ID
    Args:
        nodes: 检索结果节点序列
    Returns:
        与输入顺序一致的 chunk_id 列表
    """
    chunk_ids = []
    for item in nodes:
        node = _coerce_node(item)
        metadata = node.metadata or {}
        chunk_ids.append(str(metadata.get("chunk_id") or node.node_id))
    return chunk_ids


def extract_hit_chunk_ids(nodes: Iterable[TextNode | NodeWithScore], expected_doc_titles: List[str]) -> List[str]:
    """
    从命中预期文档的检索节点中提取分块 ID
    Args:
        nodes: 检索结果节点序列
        expected_doc_titles: 期望命中的文档标题列表
    Returns:
        所有命中的分块 ID 列表
    """
    expected = {normalize_doc_title(str(title)) for title in expected_doc_titles}
    hit_chunk_ids = []
    for item in nodes:
        node = _coerce_node(item)
        metadata = node.metadata or {}
        title = normalize_doc_title(str(metadata.get("title_hint") or metadata.get("filename") or "未知文档"))
        if title in expected:
            hit_chunk_ids.append(str(metadata.get("chunk_id") or node.node_id))
    return hit_chunk_ids


def reciprocal_rank(ranked_doc_titles: List[str], expected_doc_titles: List[str]) -> float:
    """
    计算首个命中文档的倒数排名
    Args:
        ranked_doc_titles: 检索返回的文档标题排序
        expected_doc_titles: 期望命中的文档标题列表
    Returns:
        reciprocal rank 分数
    """
    expected = {normalize_doc_title(title) for title in expected_doc_titles}
    for rank, title in enumerate(ranked_doc_titles, 1):
        if title in expected:
            return 1.0 / rank
    return 0.0


def keyword_coverage(nodes: Iterable[TextNode | NodeWithScore], expected_keywords: List[str]) -> float:
    """
    计算预期关键词在检索文本中的覆盖率
    Args:
        nodes: 检索结果节点序列
        expected_keywords: 预期出现的关键词列表
    Returns:
        关键词覆盖率
    """
    if not expected_keywords:
        return 0.0
    # 合并所有节点文本
    combined = "\n".join(_coerce_node(item).text for item in nodes)
    # 计算关键词覆盖率
    return sum(1 for keyword in expected_keywords if keyword in combined) / len(expected_keywords)


def _hit_at(ranked_doc_titles: List[str], expected_doc_titles: List[str], k: int) -> float:
    """
    计算前 k 个文档中是否命中任一期望文档
    Args:
        ranked_doc_titles: 检索返回的文档标题排序
        expected_doc_titles: 期望命中的文档标题列表
        k: 截断位置
    Returns:
        命中返回 1.0，否则返回 0.0
    """
    expected = {normalize_doc_title(title) for title in expected_doc_titles}
    return 1.0 if any(title in expected for title in ranked_doc_titles[:k]) else 0.0


def evaluate_retrieval_cases(
    cases: List[Dict[str, Any]],
    search_fn: Callable[[str, int], List[TextNode | NodeWithScore]],
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    对固定 Milvus 混合检索器结果计算命中率、MRR 和关键词覆盖率
    Args:
        cases: 评估用例列表
        search_fn: 执行检索的回调函数
        top_k: 每个问题返回的候选数量
    Returns:
        包含汇总指标和逐用例结果的报告字典
    """
    metric_sums = {"hit_at_1": 0.0, f"hit_at_{top_k}": 0.0, "mrr": 0.0, "keyword_coverage": 0.0}
    reports = []
    for case in cases:
        results = search_fn(case["question"], top_k)
        # 从检索结果中提取去重后的文档标题
        titles = extract_doc_titles(results)
        # 从检索结果中提取分块 ID
        chunk_ids = extract_chunk_ids(results)
        # 从检索结果中提取命中分块 ID
        hit_chunk_ids = extract_hit_chunk_ids(results, case["expected_doc_titles"])
        # 计算命中率
        hit_at_1 = _hit_at(titles, case["expected_doc_titles"], 1)
        hit_at_k = _hit_at(titles, case["expected_doc_titles"], top_k)
        # 计算 MRR
        mrr = reciprocal_rank(titles, case["expected_doc_titles"])
        # 计算关键词覆盖率
        coverage = keyword_coverage(results, case.get("expected_keywords", []))
        # 累加指标值
        metric_sums["hit_at_1"] += hit_at_1
        metric_sums[f"hit_at_{top_k}"] += hit_at_k
        metric_sums["mrr"] += mrr
        metric_sums["keyword_coverage"] += coverage
        # 构建报告
        report = {
            "id": case["id"],
            "question": case["question"],
            "expected_doc_titles": case["expected_doc_titles"],
            "result": {
                "ranked_doc_titles": titles,
                "ranked_chunk_ids": chunk_ids,
                "hit_chunk_ids": hit_chunk_ids,
                "hit_at_1": hit_at_1,
                f"hit_at_{top_k}": hit_at_k,
                "mrr": mrr,
                "keyword_coverage": coverage,
            },
        }
        reports.append(report)
    count = len(cases)
    # 计算平均指标值
    return {
        "summary": {"case_count": count, "top_k": top_k, "retrieval_strategy": RETRIEVAL_STRATEGY},
        "metrics": {metric: round(value / count, 4) if count else 0.0 for metric, value in metric_sums.items()},
        "cases": reports,
    }


def build_human_report(report: Dict[str, Any]) -> str:
    """
    将检索评估报告渲染为 Markdown 文本
    Args:
        report: evaluate_retrieval_cases 返回的报告字典
    Returns:
        可读的 Markdown 报告文本
    """
    summary = report.get("summary", {})
    lines = [
        "# FinRAG 检索评估报告",
        "",
        f"case_count={summary.get('case_count', 0)} top_k={summary.get('top_k', 0)} retrieval_strategy={summary.get('retrieval_strategy', RETRIEVAL_STRATEGY)}",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    top_k_key = f"hit_at_{summary.get('top_k', 3)}"
    metrics = report.get("metrics", {})
    lines.extend(
        [
            f"| hit_at_1 | {metrics.get('hit_at_1', 0.0)} |",
            f"| {top_k_key} | {metrics.get(top_k_key, 0.0)} |",
            f"| mrr | {metrics.get('mrr', 0.0)} |",
            f"| keyword_coverage | {metrics.get('keyword_coverage', 0.0)} |",
        ]
    )
    return "\n".join(lines)


def _build_local_search_fn(config: RAGConfig):
    """
    基于本地数据和配置构造评估检索函数
    Args:
        config: RAG 运行配置
    Returns:
        接收 question、top_k 的检索函数
    """
    system = FinRAGSystem(config)
    system.ensure_knowledge_base_ready()

    def hybrid_search(question: str, top_k: int):
        """
        执行单次本地检索评估
        Args:
            question: 评估问题
            top_k: 返回候选数量
        Returns:
            检索结果节点列表
        """
        # 检查混合检索器是否就绪
        if getattr(system, "hybrid_retriever", None) is not None:
            # 设置混合检索器参数
            system.hybrid_retriever.top_k = top_k
            system.hybrid_retriever.filters = {"knowledge_base_id": config.knowledge_base_id}
            # 执行混合检索
            from llama_index.core.schema import QueryBundle
            return system.hybrid_retriever.retrieve(QueryBundle(question))
        raise RuntimeError("FinRAG 混合检索器尚未就绪")
    # 返回混合检索函数
    return hybrid_search


def main():
    """
    解析命令行参数并运行检索评估
    """
    # 确保 UTF-8 输出
    configure_utf8_stdio()
    # 命令行参数解析器
    parser = argparse.ArgumentParser(description="运行 FinRAG 检索评估")
    # 评估集路径
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    # 返回候选数量
    parser.add_argument("--top-k", type=int, default=3)
    # 是否输出 JSON 格式报告
    parser.add_argument("--json", action="store_true")
    # 报告输出路径
    parser.add_argument("--report-output", type=Path)
    # 解析命令行参数
    args = parser.parse_args()

    # 加载评估集
    cases = load_eval_cases(args.eval_set)
    # 运行评估
    report = evaluate_retrieval_cases(cases, _build_local_search_fn(RAGConfig.from_env()), top_k=args.top_k)
    if args.report_output is not None:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(build_human_report(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else build_human_report(report))


if __name__ == "__main__":
    main()
