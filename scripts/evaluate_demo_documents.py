"""运行 demo-documents 的确定性 RAG 评测套件"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finrag.evaluation.runner import evaluate_suite, save_report  # noqa: E402
from finrag.evaluation.suite import DEFAULT_SUITE_PATH, load_eval_suite  # noqa: E402


def configure_utf8_stdio() -> None:
    """在 Windows 终端中强制使用 UTF-8，避免中文报告输出乱码。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """构建评测脚本命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 demo-documents RAG 质量评测")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument("--stage", action="append", choices=["retrieval", "qa"], default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """加载测试集、执行评测并按质量门禁返回退出码。"""
    configure_utf8_stdio()
    args = build_parser().parse_args(argv)
    suite = load_eval_suite(args.suite)
    report = evaluate_suite(suite, stages=tuple(args.stage or ["retrieval", "qa"]))
    output = save_report(report, args.report_output)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"report={output}")
    return 0 if report.get("gate", {}).get("passed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
