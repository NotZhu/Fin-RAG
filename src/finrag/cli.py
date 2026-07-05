"""FinRAG 命令行入口"""

from __future__ import annotations

import argparse
import logging
from typing import Sequence

from .application.system import FinRAGSystem
from .core.config import validate_knowledge_base_id

logger = logging.getLogger(__name__)


def _knowledge_base_id_argument(value: str) -> str:
    try:
        return validate_knowledge_base_id(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


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
    rebuild_parser = subparsers.add_parser("rebuild", help="从源文档全量重建 PostgreSQL 节点和 Milvus 索引")
    rebuild_parser.add_argument(
        "--knowledge-base-id",
        "--kb",
        dest="knowledge_base_id",
        required=True,
        type=_knowledge_base_id_argument,
        help="要全量重建的知识库 ID",
    )
    # 解析命令行参数
    args = parser.parse_args(argv)
    try:
        if args.command == "rebuild":
            result = FinRAGSystem().rebuild_from_sources(args.knowledge_base_id)
            _print_key_values(result)
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
