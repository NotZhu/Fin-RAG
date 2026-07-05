"""预览 Docling 解析结果和每篇源文档的三层节点"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_DEMO_DIR_NAME = "demo-documents"


@dataclass
class FilePreview:
    """单个源文件的解析和节点预览结果。"""

    source: Path
    relative_path: str
    out_dir: Path
    status: str
    documents: int = 0
    all_nodes: int = 0
    leaf_nodes: int = 0
    root_nodes: int = 0
    section_nodes: int = 0
    indexed_leaf_nodes: int = 0
    error: str = ""
    files: list[str] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        """转换为顶层 summary 中的文件条目。"""
        return {
            "source": self.source.as_posix(),
            "relative_path": self.relative_path,
            "out_dir": self.out_dir.as_posix(),
            "status": self.status,
            "documents": self.documents,
            "all_nodes": self.all_nodes,
            "leaf_nodes": self.leaf_nodes,
            "root_nodes": self.root_nodes,
            "section_nodes": self.section_nodes,
            "indexed_leaf_nodes": self.indexed_leaf_nodes,
            "error": self.error,
            "files": self.files,
        }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="查看文件或目录经 Docling 和三层节点构建后的结果")
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=None,
        help="待解析文件或目录；省略时默认使用 output/demo-documents",
    )
    parser.add_argument("--knowledge-base-id", default="debug", help="调试用知识库 ID")
    parser.add_argument("--data-root", type=Path, default=None, help="数据根目录，默认目录源为自身、文件源为父目录")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出目录，默认 output/ingestion-preview/<源文件或目录名>",
    )
    return parser.parse_args()


def main() -> int:
    """执行批量 Docling JSON 解析预览和三层节点预览。"""
    args = parse_args()
    ensure_src_path()
    ensure_local_model_cache()

    source = resolve_source(args.source)
    if args.source is None:
        print_step(f"未提供 source，默认使用: {source}")

    if not source.exists():
        print_error(f"源路径不存在: {source}")
        return 2

    data_root = resolve_data_root(source, args.data_root)
    out_dir = resolve_out_dir(source, args.out)
    source_files = discover_sources(source)
    if not source_files:
        print_error(f"未找到支持格式文件: {source}")
        return 2

    clean_preview_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    documents_root = out_dir / "01_documents"
    documents_root.mkdir(parents=True, exist_ok=True)

    results: list[FilePreview] = []
    for index, source_file in enumerate(source_files, 1):
        relative_path = safe_relative_path(source_file, data_root)
        document_out_dir = documents_root / document_preview_dir_name(index, source_file)
        results.append(
            inspect_source_file(
                source_file,
                relative_path=relative_path,
                document_out_dir=document_out_dir,
                data_root=data_root,
                knowledge_base_id=args.knowledge_base_id,
            )
        )

    parsed_results = [result for result in results if result.status == "ok"]
    if not parsed_results:
        print_error(f"Docling 未解析出任何 Document: {source}")
        print_error("请查看上方 Docling 日志；常见原因包括模型/内存问题、文件格式异常或解析依赖缺失。")
        return 1

    write_top_level_outputs(out_dir, source, data_root, results)
    print_success(out_dir, results)
    return 0


def inspect_source_file(
    source: Path,
    *,
    relative_path: str,
    document_out_dir: Path,
    data_root: Path,
    knowledge_base_id: str,
) -> FilePreview:
    """解析单个源文件，写出该文件独立的三层节点预览。"""
    from finrag.ingestion.docling_loader import load_docling_documents
    from finrag.indexing.nodes import DataPreparationModule

    print_step(f"正在使用 Docling 解析: {source}")
    documents = load_docling_documents(
        source,
        knowledge_base_id=knowledge_base_id,
        data_root=data_root,
    )
    if not documents:
        print_error(f"Docling 未解析出任何 Document: {source}")
        return FilePreview(source, relative_path, document_out_dir, "empty")

    print_step(f"Docling Document 数量: {len(documents)} ({source.name})")
    module = DataPreparationModule(
        str(data_root),
        knowledge_base_id=knowledge_base_id,
    )
    all_nodes, leaf_nodes = module._build_hierarchical_nodes(documents)
    print_step(f"三层节点: all={len(all_nodes)}, leaf={len(leaf_nodes)} ({source.name})")

    document_out_dir.mkdir(parents=True, exist_ok=True)
    files = write_file_preview(
        document_out_dir,
        source=source,
        relative_path=relative_path,
        data_root=data_root,
        documents=documents,
        all_nodes=all_nodes,
        leaf_nodes=leaf_nodes,
    )
    return FilePreview(
        source=source,
        relative_path=relative_path,
        out_dir=document_out_dir,
        status="ok",
        documents=len(documents),
        all_nodes=len(all_nodes),
        leaf_nodes=len(leaf_nodes),
        root_nodes=count_level(all_nodes, 1),
        section_nodes=count_level(all_nodes, 2),
        indexed_leaf_nodes=count_level(leaf_nodes, 3),
        files=files,
    )


def write_file_preview(
    out_dir: Path,
    *,
    source: Path,
    relative_path: str,
    data_root: Path,
    documents: Sequence[object],
    all_nodes: Sequence[object],
    leaf_nodes: Sequence[object],
) -> list[str]:
    """写出单篇源文件的所有预览文件。"""
    summary = {
        "source": source.as_posix(),
        "relative_path": relative_path,
        "data_root": data_root.as_posix(),
        "documents": len(documents),
        "all_nodes": len(all_nodes),
        "leaf_nodes": len(leaf_nodes),
        "root_nodes": count_level(all_nodes, 1),
        "section_nodes": count_level(all_nodes, 2),
        "indexed_leaf_nodes": count_level(leaf_nodes, 3),
    }
    write_json(out_dir / "00_document_summary.json", summary)
    write_docling_documents(out_dir, documents)
    (out_dir / "02_three_level_tree.md").write_text(
        format_three_level_tree(source, all_nodes),
        encoding="utf-8",
    )
    (out_dir / "03_leaf_nodes_for_milvus.md").write_text(format_nodes(leaf_nodes), encoding="utf-8")
    write_relationships_csv(out_dir / "04_relationships.csv", all_nodes)
    return [
        "00_document_summary.json",
        "01_docling_documents.md",
        "02_three_level_tree.md",
        "03_leaf_nodes_for_milvus.md",
        "04_relationships.csv",
    ]


def write_top_level_outputs(out_dir: Path, source: Path, data_root: Path, results: Sequence[FilePreview]) -> None:
    """写出批量预览的顶层索引和总览。"""
    ok_results = [result for result in results if result.status == "ok"]
    summary = {
        "source": source.as_posix(),
        "data_root": data_root.as_posix(),
        "out_dir": out_dir.as_posix(),
        "files_total": len(results),
        "files_parsed": len(ok_results),
        "files_empty": sum(1 for result in results if result.status == "empty"),
        "documents": sum(result.documents for result in ok_results),
        "all_nodes": sum(result.all_nodes for result in ok_results),
        "leaf_nodes": sum(result.leaf_nodes for result in ok_results),
        "root_nodes": sum(result.root_nodes for result in ok_results),
        "section_nodes": sum(result.section_nodes for result in ok_results),
        "indexed_leaf_nodes": sum(result.indexed_leaf_nodes for result in ok_results),
        "files": [result.to_summary() for result in results],
    }
    write_json(out_dir / "00_summary.json", summary)
    (out_dir / "00_index.md").write_text(format_index(summary, results), encoding="utf-8")
    (out_dir / "README.md").write_text(preview_readme(source, out_dir), encoding="utf-8")


def resolve_source(source: Path | None) -> Path:
    """解析默认源路径。"""
    if source is not None:
        return source.resolve()
    return (PROJECT_ROOT / "output" / DEFAULT_DEMO_DIR_NAME).resolve()


def resolve_data_root(source: Path, explicit_data_root: Path | None) -> Path:
    """解析用于稳定相对路径和 document_id 的数据根目录。"""
    if explicit_data_root is not None:
        return explicit_data_root.resolve()
    if source.is_dir():
        return source.resolve()
    return source.parent.resolve()


def resolve_out_dir(source: Path, explicit_out: Path | None) -> Path:
    """解析输出目录。"""
    if explicit_out is not None:
        return explicit_out.resolve()
    return (PROJECT_ROOT / "output" / "ingestion-preview" / source.stem).resolve()


def discover_sources(source: Path) -> list[Path]:
    """发现需要预览的支持格式源文件。"""
    from finrag.ingestion.parsers import SUPPORTED_SUFFIXES

    if source.is_file():
        return [source.resolve()] if source.suffix.lower() in SUPPORTED_SUFFIXES else []
    return [
        path.resolve()
        for path in sorted(source.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]


def clean_preview_dir(out_dir: Path) -> None:
    """清理脚本生成的旧预览文件，避免新旧结果混在一起。"""
    if not out_dir.exists():
        return
    for filename in ("00_summary.json", "00_index.md", "README.md"):
        path = out_dir / filename
        if path.is_file():
            path.unlink()
    documents_dir = out_dir / "01_documents"
    if documents_dir.exists() and documents_dir.is_dir():
        shutil.rmtree(documents_dir)


def write_docling_documents(out_dir: Path, documents: Iterable[object]) -> None:
    """保存 Docling JSON Document 原文和可读版本。"""
    sections: list[str] = []
    for index, document in enumerate(documents, 1):
        text = str(getattr(document, "text", "") or "")
        metadata = dict(getattr(document, "metadata", {}) or {})
        sections.append(
            "\n".join(
                [
                    f"# Docling Document {index}",
                    "",
                    "## Metadata",
                    "",
                    fenced_json(metadata),
                    "",
                    "## Text",
                    "",
                    fenced_text(pretty_json_text(text), "json" if is_json_text(text) else "text"),
                ]
            )
        )
        if is_json_text(text):
            (out_dir / f"01_docling_document_{index:03d}.json").write_text(
                pretty_json_text(text),
                encoding="utf-8",
            )
    (out_dir / "01_docling_documents.md").write_text("\n\n---\n\n".join(sections), encoding="utf-8")


def format_three_level_tree(source: Path, nodes: Sequence[object]) -> str:
    """按 root/section/leaf 层级展示单篇文档的全部节点。"""
    node_list = list(nodes)
    children_by_parent = children_map(node_list)
    roots = root_nodes(node_list)
    lines = [
        f"# {source.name}",
        "",
        "这个文件展示该源文档最终进入 PostgreSQL docstore 的三层节点；Milvus dense/BM25 hybrid 索引只使用 L3 leaf。",
        "",
        "```text",
        "L1 root    = 文档根节点",
        "L2 section = 章节聚合节点",
        "L3 leaf    = 检索索引节点",
        "```",
        "",
    ]
    for root in roots:
        append_node_block(lines, root, children_by_parent, depth=0)
    return "\n".join(lines).rstrip() + "\n"


def append_node_block(
    lines: list[str],
    node: object,
    children_by_parent: dict[str, list[object]],
    *,
    depth: int,
) -> None:
    """递归写入一个节点和它的子节点。"""
    metadata = node_metadata(node)
    level = int(metadata.get("chunk_level") or 0)
    heading = "#" * min(depth + 2, 6)
    role = {1: "root", 2: "section", 3: "leaf"}.get(level, "node")
    title = node_title(node)
    children = children_by_parent.get(str(getattr(node, "node_id", "")), [])
    lines.extend(
        [
            f"{heading} L{level} {role}: {title}",
            "",
            f"- node_id: `{getattr(node, 'node_id', '')}`",
            f"- parent_chunk_id: `{metadata.get('parent_chunk_id', '')}`",
            f"- root_chunk_id: `{metadata.get('root_chunk_id', '')}`",
            f"- chunk_idx: `{metadata.get('chunk_idx', '')}`",
            f"- children: `{len(children)}`",
        ]
    )
    if metadata.get("section_title"):
        lines.append(f"- section_title: `{metadata['section_title']}`")
    if metadata.get("page_number"):
        lines.append(f"- page_number: `{metadata['page_number']}`")
    lines.extend(
        [
            "",
            "<details>",
            "<summary>Metadata</summary>",
            "",
            fenced_json(metadata),
            "",
            "</details>",
            "",
            "#### Text",
            "",
            fenced_text(node_text_for_tree(node, level), "text"),
            "",
        ]
    )
    for child in children:
        append_node_block(lines, child, children_by_parent, depth=depth + 1)


def format_nodes(nodes: Iterable[object]) -> str:
    """将 LlamaIndex 叶子节点格式化为便于人工查看的 Markdown。"""
    sections: list[str] = []
    for index, node in enumerate(nodes, 1):
        metadata = node_metadata(node)
        text = str(getattr(node, "text", "") or "")
        sections.append(
            "\n".join(
                [
                    f"# Leaf Chunk {index}",
                    "",
                    f"- node_id: `{getattr(node, 'node_id', '')}`",
                    f"- parent_chunk_id: `{metadata.get('parent_chunk_id', '')}`",
                    f"- root_chunk_id: `{metadata.get('root_chunk_id', '')}`",
                    f"- section_title: `{metadata.get('section_title', '')}`",
                    f"- page_number: `{metadata.get('page_number', '')}`",
                    "",
                    fenced_json(metadata),
                    "",
                    "## Text",
                    "",
                    text,
                ]
            )
        )
    return "\n\n---\n\n".join(sections)


def write_relationships_csv(path: Path, nodes: Iterable[object]) -> None:
    """保存节点关系表，方便检查父子关系是否正确。"""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "node_id",
                "chunk_level",
                "chunk_idx",
                "parent_chunk_id",
                "root_chunk_id",
                "children_count",
                "document_id",
                "filename",
                "section_title",
                "page_number",
            ],
        )
        writer.writeheader()
        children_by_parent = children_map(list(nodes))
        for node in nodes:
            metadata = node_metadata(node)
            writer.writerow(
                {
                    "node_id": getattr(node, "node_id", ""),
                    "chunk_level": metadata.get("chunk_level", ""),
                    "chunk_idx": metadata.get("chunk_idx", ""),
                    "parent_chunk_id": metadata.get("parent_chunk_id", ""),
                    "root_chunk_id": metadata.get("root_chunk_id", ""),
                    "children_count": len(children_by_parent.get(str(getattr(node, "node_id", "")), [])),
                    "document_id": metadata.get("document_id", ""),
                    "filename": metadata.get("filename", ""),
                    "section_title": metadata.get("section_title", ""),
                    "page_number": metadata.get("page_number", ""),
                }
            )


def format_index(summary: dict[str, Any], results: Sequence[FilePreview]) -> str:
    """生成顶层预览索引。"""
    lines = [
        "# Ingestion Preview Index",
        "",
        f"- source: `{summary['source']}`",
        f"- data_root: `{summary['data_root']}`",
        f"- files: `{summary['files_parsed']}/{summary['files_total']}` parsed",
        f"- documents: `{summary['documents']}`",
        f"- all_nodes: `{summary['all_nodes']}`",
        f"- indexed_leaf_nodes: `{summary['indexed_leaf_nodes']}`",
        "",
        "| 文件 | 状态 | Documents | L1 | L2 | L3 | 三层树 | Leaf 节点 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for result in results:
        tree_link = link_to(result.out_dir / "02_three_level_tree.md", summary["out_dir"])
        leaf_link = link_to(result.out_dir / "03_leaf_nodes_for_milvus.md", summary["out_dir"])
        lines.append(
            "| "
            f"`{result.relative_path}` | "
            f"{result.status} | "
            f"{result.documents} | "
            f"{result.root_nodes} | "
            f"{result.section_nodes} | "
            f"{result.indexed_leaf_nodes} | "
            f"[tree]({tree_link}) | "
            f"[leaf]({leaf_link}) |"
        )
    return "\n".join(lines).rstrip() + "\n"


def preview_readme(source: Path, out_dir: Path) -> str:
    """生成预览目录说明。"""
    return "\n".join(
        [
            "# Ingestion Preview",
            "",
            f"- 源路径：`{source.as_posix()}`",
            f"- 输出目录：`{out_dir.as_posix()}`",
            "",
            "## 顶层文件",
            "",
            "- `00_summary.json`：批量预览的总统计和每个文件的状态。",
            "- `00_index.md`：面向人工阅读的入口，链接到每篇文档的三层节点树。",
            "- `01_documents/<序号>_<文件名>/`：每篇源文档的独立预览目录。",
            "",
            "## 单篇文档目录",
            "",
            "- `00_document_summary.json`：该文档的节点数量统计。",
            "- `01_docling_documents.md`：DoclingReader 产出的 LlamaIndex Document。",
            "- `01_docling_document_001.json`：Docling JSON 原文，存在 JSON 文本时生成。",
            "- `02_three_level_tree.md`：root/section/leaf 三层节点长相。",
            "- `03_leaf_nodes_for_milvus.md`：最终写入 Milvus dense/BM25 hybrid 索引的 L3 leaf 节点。",
            "- `04_relationships.csv`：节点 ID、父节点、根节点和章节页码关系表。",
            "",
        ]
    )


def print_success(out_dir: Path, results: Sequence[FilePreview]) -> None:
    """打印预览生成结果。"""
    print("解析预览已生成:", flush=True)
    print(f"- {out_dir / '00_summary.json'}", flush=True)
    print(f"- {out_dir / '00_index.md'}", flush=True)
    for result in results:
        if result.status == "ok":
            print(f"- {result.relative_path}: {result.out_dir / '02_three_level_tree.md'}", flush=True)


def children_map(nodes: Sequence[object]) -> dict[str, list[object]]:
    """根据 metadata.parent_chunk_id 构建父子映射。"""
    mapping: dict[str, list[object]] = {}
    for node in nodes:
        metadata = node_metadata(node)
        node_id = str(getattr(node, "node_id", ""))
        parent_id = str(metadata.get("parent_chunk_id") or "")
        if parent_id and parent_id != node_id:
            mapping.setdefault(parent_id, []).append(node)
    return mapping


def root_nodes(nodes: Sequence[object]) -> list[object]:
    """找出文档根节点。"""
    roots = [node for node in nodes if int(node_metadata(node).get("chunk_level") or 0) == 1]
    if roots:
        return roots
    return [
        node
        for node in nodes
        if str(node_metadata(node).get("parent_chunk_id") or "") in {"", str(getattr(node, "node_id", ""))}
    ]


def count_level(nodes: Iterable[object], level: int) -> int:
    """统计指定层级节点数量。"""
    return sum(1 for node in nodes if int(node_metadata(node).get("chunk_level") or 0) == level)


def node_metadata(node: object) -> dict[str, Any]:
    """读取节点 metadata。"""
    return dict(getattr(node, "metadata", {}) or {})


def node_title(node: object) -> str:
    """生成节点标题。"""
    metadata = node_metadata(node)
    level = int(metadata.get("chunk_level") or 0)
    if level == 1:
        return str(metadata.get("filename") or "document")
    if level == 2:
        return str(metadata.get("section_title") or "section")
    return compact_text(str(getattr(node, "text", "") or ""), limit=80) or "leaf"


def node_text_for_tree(node: object, level: int) -> str:
    """树视图中文本预览：L1/L2 截断，L3 保留完整正文。"""
    text = str(getattr(node, "text", "") or "")
    if level >= 3:
        return text
    return compact_text(text, limit=700)


def document_preview_dir_name(index: int, source: Path) -> str:
    """生成稳定、可读的单篇文档预览目录名。"""
    return f"{index:03d}_{safe_filename(source.stem)}"


def safe_filename(value: str) -> str:
    """清理 Windows 和 Markdown 都友好的文件名片段。"""
    name = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._")
    return name or "document"


def safe_relative_path(path: Path, root: Path) -> str:
    """尽量生成相对于 data_root 的路径。"""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def link_to(path: Path, root: str | Path) -> str:
    """生成 Markdown 相对链接。"""
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def compact_text(text: str, *, limit: int) -> str:
    """压缩文本预览。"""
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def fenced_json(value: Any) -> str:
    """生成 JSON Markdown 代码块。"""
    return fenced_text(json.dumps(value, ensure_ascii=False, indent=2), "json")


def fenced_text(text: str, language: str) -> str:
    """生成 Markdown 代码块。"""
    return f"```{language}\n{text}\n```"


def is_json_text(text: str) -> bool:
    """判断文本是否为 JSON。"""
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def pretty_json_text(text: str) -> str:
    """格式化 JSON 文本；非 JSON 原样返回。"""
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        return text


def write_json(path: Path, value: Any) -> None:
    """写入 UTF-8 JSON。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def print_step(message: str) -> None:
    """打印可立即刷新的脚本进度。"""
    print(f"[inspect] {message}", flush=True)


def print_error(message: str) -> None:
    """打印可立即刷新的错误信息。"""
    print(message, file=sys.stderr, flush=True)


def ensure_src_path() -> None:
    """确保脚本可在未安装包的源码目录中直接运行。"""
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))


def ensure_local_model_cache() -> None:
    """将 Docling/HuggingFace 模型缓存放到工作区内，避免系统目录权限问题。"""
    cache_root = PROJECT_ROOT / ".debug" / "model-cache"
    huggingface_home = cache_root / "huggingface"
    torch_home = cache_root / "torch"
    os.environ.setdefault("HF_HOME", str(huggingface_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(huggingface_home / "hub"))
    os.environ.setdefault("TORCH_HOME", str(torch_home))
    huggingface_home.mkdir(parents=True, exist_ok=True)
    torch_home.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
