"""FinRAG 命令行入口"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from .application.system import FinRAGSystem

logger = logging.getLogger(__name__)


def run_evaluation_report(args: argparse.Namespace):
    """
    根据命令行参数运行 Ragas 评估并可选写出报告
    Args:
        args: argparse 解析后的 eval 子命令参数
    Returns:
        Ragas 评估报告对象
    """
    from scripts.evaluate_ragas import load_ragas_rows, run_ragas
    # 加载 Ragas 评估输入行
    rows = load_ragas_rows(args.dataset)
    # 运行 Ragas 评估
    result = run_ragas(rows)
    # 如果指定了输出报告路径，写出 JSON 文件
    if args.output is not None:
        # 将 Ragas 结果对象转换为 JSON 字符串
        payload = json.loads(result.to_pandas().to_json(force_ascii=False, orient="records"))
        # 确保输出目录存在
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # 写出 JSON 文件
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _provider_name(report) -> str:
    """
    从评估报告中提取评估提供方名称
    Args:
        report: Ragas 或本地评估报告对象
    Returns:
        provider 字段值，缺失时返回 unknown
    """
    provider = getattr(report, "provider", None)
    if provider:
        return str(provider)
    rows = getattr(report, "rows", None) or []
    if rows and isinstance(rows[0], dict):
        return str(rows[0].get("provider") or "unknown")
    return "unknown"


def _print_key_values(payload: dict) -> None:
    """
    将字典以 key=value 形式输出到终端
    Args:
        payload: 待输出的键值字典
    """
    print(" ".join(f"{key}={value}" for key, value in payload.items()))


def main(argv: Sequence[str] | None = None) -> int:
    """
    启动交互客户端，或执行资料库维护子命令
    Args:
        argv: 命令行参数列表
    Returns:
        进程退出码
    """
    # 初始化 argparse 解析器
    parser = argparse.ArgumentParser(prog="finrag")
    # 添加子命令解析器
    subparsers = parser.add_subparsers(dest="command")
    # 添加重建索引子命令
    subparsers.add_parser("rebuild", help="从源文档全量重建 PostgreSQL/BM25/Milvus 索引")
    # 添加评估子命令
    eval_parser = subparsers.add_parser("eval", help="运行 Ragas 评估报告")
    # 添加评估数据集路径参数
    eval_parser.add_argument("--dataset", required=True, type=Path)
    # 添加输出报告路径参数
    eval_parser.add_argument("--output", type=Path)

    # 解析命令行参数
    args = parser.parse_args(argv)
    try:
        if args.command == "rebuild":
            result = FinRAGSystem().rebuild_from_sources()
            _print_key_values(result)
            return 0
        if args.command == "eval":
            report = run_evaluation_report(args)
            print(f"provider={_provider_name(report)}")
            return 0
        # 没有指定子命令时启动交互客户端
        FinRAGSystem().run_interactive()
        return 0
    except Exception as exc:
        logger.error("系统运行出错: %s", exc)
        print(f"系统错误: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
