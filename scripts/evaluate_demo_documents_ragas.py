"""基于最新 demo-documents 报告运行可选 Ragas 语义评测"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finrag.evaluation.ragas_runner import (  # noqa: E402
    RagasDependencyError,
    RagasEvaluationError,
    build_ragas_sample_dicts,
    run_ragas_samples,
)
from finrag.evaluation.suite import DEFAULT_REPORT_DIR, DEFAULT_SUITE_PATH, load_eval_suite  # noqa: E402


def configure_utf8_stdio() -> None:
    """在 Windows 终端中强制使用 UTF-8，避免中文输出乱码"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """构建 Ragas 语义评测脚本命令行参数"""
    parser = argparse.ArgumentParser(description="运行 demo-documents Ragas 语义评测")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE_PATH)
    parser.add_argument("--from-report", type=Path, default=DEFAULT_REPORT_DIR / "latest.json")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只生成 Ragas 输入样本预览，不调用评测 LLM")
    return parser


def main(argv: list[str] | None = None) -> int:
    """加载确定性 QA 报告，转换样本并按需运行 Ragas"""
    configure_utf8_stdio()
    load_project_env()
    args = build_parser().parse_args(argv)
    suite = load_eval_suite(args.suite)
    source_report = json.loads(args.from_report.read_text(encoding="utf-8"))
    samples = build_ragas_sample_dicts(source_report, suite, max_cases=args.max_cases)
    if args.dry_run:
        report = _preview_report(samples)
        default_name = "ragas-preview.json"
    else:
        try:
            report = run_ragas_samples(samples)
        except (RagasDependencyError, RagasEvaluationError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        default_name = "ragas-latest.json"
    output = _save_report(report, args.report_output or DEFAULT_REPORT_DIR / default_name)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"report={output}")
    return 0


def load_project_env() -> None:
    """加载项目根目录 .env，已有环境变量保持优先"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


def _preview_report(samples: list[dict]) -> dict:
    return {
        "summary": {"case_count": len(samples), "qa_strategy": "ragas_sample_preview"},
        "cases": [{"id": sample.get("metadata", {}).get("case_id", str(index + 1)), "sample": sample} for index, sample in enumerate(samples)],
    }


def _save_report(report: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    raise SystemExit(main())
