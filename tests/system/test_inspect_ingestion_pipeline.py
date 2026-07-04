import json
import sys
import zipfile
from types import ModuleType

import fitz
from openpyxl import load_workbook
from PIL import Image
from llama_index.core import Document

from scripts import generate_docling_demo_pdf, inspect_ingestion_pipeline


class FakePreviewNode:
    def __init__(self, node_id, level, text, metadata=None, parent=None):
        self.node_id = node_id
        self.text = text
        self.parent_node = parent
        self.child_nodes = []
        self.metadata = {
            "chunk_id": node_id,
            "chunk_level": level,
            "parent_chunk_id": parent.node_id if parent else node_id,
            "root_chunk_id": parent.metadata["root_chunk_id"] if parent else node_id,
            "chunk_idx": 0,
            **(metadata or {}),
        }


def _install_empty_docling_modules(monkeypatch):
    fake_loader_module = ModuleType("finrag.ingestion.docling_loader")
    fake_loader_module.load_docling_documents = lambda *args, **kwargs: []

    fake_nodes_module = ModuleType("finrag.indexing.nodes")

    class UnusedDataPreparationModule:
        def __init__(self, *args, **kwargs):
            raise AssertionError("empty Docling results should stop before node building")

    fake_nodes_module.DataPreparationModule = UnusedDataPreparationModule
    monkeypatch.setitem(sys.modules, "finrag.ingestion.docling_loader", fake_loader_module)
    monkeypatch.setitem(sys.modules, "finrag.indexing.nodes", fake_nodes_module)


def _install_preview_modules(monkeypatch):
    fake_loader_module = ModuleType("finrag.ingestion.docling_loader")

    def load_docling_documents(path, *, knowledge_base_id, data_root=None):
        source = path.resolve()
        return [
            Document(
                text=f"Docling JSON for {source.name}",
                metadata={
                    "document_id": f"doc-{source.stem}",
                    "knowledge_base_id": knowledge_base_id,
                    "source_path": str(source),
                    "filename": source.name,
                    "file_type": source.suffix.lower().lstrip("."),
                    "parser_name": "docling",
                },
            )
        ]

    fake_loader_module.load_docling_documents = load_docling_documents

    fake_nodes_module = ModuleType("finrag.indexing.nodes")

    class FakeDataPreparationModule:
        def __init__(self, data_path, *, knowledge_base_id):
            self.data_path = data_path
            self.knowledge_base_id = knowledge_base_id

        def _build_hierarchical_nodes(self, documents):
            document = documents[0]
            metadata = dict(document.metadata)
            filename = metadata["filename"]
            root = FakePreviewNode(
                f"{filename}-root",
                1,
                f"{filename} 文档根节点",
                metadata,
            )
            section = FakePreviewNode(
                f"{filename}-section",
                2,
                f"{filename} 授信审批章节",
                metadata | {"section_title": "授信审批"},
                parent=root,
            )
            leaf = FakePreviewNode(
                f"{filename}-leaf",
                3,
                f"{filename} 叶子节点正文",
                metadata | {"section_title": "授信审批", "page_number": 1},
                parent=section,
            )
            root.child_nodes = [section]
            section.child_nodes = [leaf]
            return [root, section, leaf], [leaf]

    fake_nodes_module.DataPreparationModule = FakeDataPreparationModule
    monkeypatch.setitem(sys.modules, "finrag.ingestion.docling_loader", fake_loader_module)
    monkeypatch.setitem(sys.modules, "finrag.indexing.nodes", fake_nodes_module)


def test_inspect_pipeline_reports_empty_docling_result_without_success(monkeypatch, tmp_path, capsys):
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"%PDF mocked")
    out_dir = tmp_path / "preview"

    _install_empty_docling_modules(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect_ingestion_pipeline.py",
            str(source),
            "--out",
            str(out_dir),
        ],
    )

    exit_code = inspect_ingestion_pipeline.main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "正在使用 Docling 解析" in captured.out
    assert "解析预览已生成" not in captured.out
    assert "Docling 未解析出任何 Document" in captured.err
    assert not (out_dir / "00_summary.json").exists()


def test_inspect_pipeline_uses_demo_documents_dir_when_source_is_omitted(monkeypatch, tmp_path, capsys):
    project_root = tmp_path / "project"
    demo_dir = project_root / "output" / "demo-documents"
    demo_dir.mkdir(parents=True)
    (demo_dir / "finrag_docling_demo.md").write_text("# mocked", encoding="utf-8")

    monkeypatch.setattr(inspect_ingestion_pipeline, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(inspect_ingestion_pipeline, "SRC_ROOT", project_root / "src")
    _install_empty_docling_modules(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["inspect_ingestion_pipeline.py"])

    exit_code = inspect_ingestion_pipeline.main()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "未提供 source，默认使用" in captured.out
    assert str(demo_dir.resolve()) in captured.out
    assert "Docling 未解析出任何 Document" in captured.err


def test_inspect_pipeline_writes_readable_three_level_preview_for_each_source_file(monkeypatch, tmp_path):
    source_dir = tmp_path / "documents"
    source_dir.mkdir()
    (source_dir / "credit.md").write_text("# 授信审批\n正文", encoding="utf-8")
    (source_dir / "risk.txt").write_text("风险监测正文", encoding="utf-8")
    out_dir = tmp_path / "preview"

    _install_preview_modules(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "inspect_ingestion_pipeline.py",
            str(source_dir),
            "--out",
            str(out_dir),
            "--knowledge-base-id",
            "kb-test",
        ],
    )

    exit_code = inspect_ingestion_pipeline.main()

    assert exit_code == 0
    summary = json.loads((out_dir / "00_summary.json").read_text(encoding="utf-8"))
    assert summary["files_total"] == 2
    assert summary["files_parsed"] == 2
    assert summary["documents"] == 2
    assert summary["root_nodes"] == 2
    assert summary["section_nodes"] == 2
    assert summary["indexed_leaf_nodes"] == 2

    document_dirs = sorted((out_dir / "01_documents").iterdir())
    assert [path.name for path in document_dirs] == ["001_credit", "002_risk"]

    credit_tree = (document_dirs[0] / "02_three_level_tree.md").read_text(encoding="utf-8")
    assert "# credit.md" in credit_tree
    assert "L1 root" in credit_tree
    assert "L2 section" in credit_tree
    assert "L3 leaf" in credit_tree
    assert "授信审批" in credit_tree
    assert "credit.md 叶子节点正文" in credit_tree

    index = (out_dir / "00_index.md").read_text(encoding="utf-8")
    assert "credit.md" in index
    assert "risk.txt" in index
    assert "001_credit/02_three_level_tree.md" in index


def test_demo_generator_writes_multiple_document_formats(monkeypatch, tmp_path):
    out_dir = tmp_path / "demo-documents"
    monkeypatch.setattr(
        generate_docling_demo_pdf,
        "save_demo_pdf",
        lambda output: output.write_bytes(b"%PDF mocked"),
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_docling_demo_pdf.py", "--out-dir", str(out_dir)],
    )

    exit_code = generate_docling_demo_pdf.main()

    assert exit_code == 0
    generated_suffixes = {path.suffix for path in out_dir.iterdir() if path.is_file()}
    assert {".pdf", ".md", ".txt", ".csv", ".json", ".html", ".docx", ".xlsx", ".pptx", ".png"}.issubset(
        generated_suffixes
    )
    docling_json = json.loads((out_dir / "finrag_docling_demo.json").read_text(encoding="utf-8"))
    assert docling_json["schema_name"] == "DoclingDocument"

    with zipfile.ZipFile(out_dir / "finrag_docling_demo.docx") as docx:
        names = set(docx.namelist())
        document_xml = docx.read("word/document.xml").decode("utf-8")
        styles_xml = docx.read("word/styles.xml").decode("utf-8")
        rels_xml = docx.read("word/_rels/document.xml.rels").decode("utf-8")
    assert "word/styles.xml" in names
    assert 'w:styleId="Heading1"' in styles_xml
    assert 'Target="styles.xml"' in rels_xml
    assert document_xml.count('w:pStyle w:val="Heading1"') >= 6
    assert "<w:tblGrid>" in document_xml
    assert 'w:tcW w:w="' in document_xml

    with zipfile.ZipFile(out_dir / "finrag_docling_demo.xlsx") as workbook:
        workbook_names = set(workbook.namelist())
        workbook_xml = "\n".join(
            workbook.read(name).decode("utf-8", errors="ignore")
            for name in workbook_names
            if name.endswith(".xml")
        )
    assert "xl/worksheets/sheet1.xml" in workbook_names
    assert "xl/worksheets/sheet2.xml" in workbook_names
    assert "经营指标" in workbook_xml
    assert "整改跟踪" in workbook_xml
    workbook = load_workbook(out_dir / "finrag_docling_demo.xlsx", read_only=True, data_only=True)
    assert workbook["经营指标"]["A1"].value == "指标"
    assert list(workbook["经营指标"].iter_rows(min_row=1, max_row=1, values_only=True))[0] == (
        "指标",
        "2025H1",
        "2026H1",
        "同比变化",
        "风险提示",
    )

    with zipfile.ZipFile(out_dir / "finrag_docling_demo.pptx") as deck:
        deck_names = set(deck.namelist())
        slide_text = deck.read("ppt/slides/slide1.xml").decode("utf-8")
    assert "ppt/slides/slide1.xml" in deck_names
    assert "华东智造集团" in slide_text
    assert "授信风险监测" in slide_text

    with Image.open(out_dir / "finrag_docling_demo_scan.png") as image:
        assert image.size == generate_docling_demo_pdf.PAGE_SIZE


def test_demo_pdf_contains_extractable_text(tmp_path):
    pdf_path = tmp_path / "demo.pdf"

    generate_docling_demo_pdf.save_demo_pdf(pdf_path)

    with fitz.open(pdf_path) as pdf:
        extracted_text = "\n".join(page.get_text() for page in pdf)
    normalized_text = " ".join(extracted_text.split())
    normalized_title = " ".join(generate_docling_demo_pdf.REPORT_TITLE.split())
    assert normalized_title in normalized_text
    assert "一、执行摘要" in extracted_text
