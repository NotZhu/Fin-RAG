import pytest
from llama_index.core import Document

from finrag.ingestion.parsers import (
    SUPPORTED_SUFFIXES,
    ParserRegistry,
    compute_content_hash,
    load_documents,
)
from tests.support.fakes import MemoryDocumentRegistry


def _registry():
    return MemoryDocumentRegistry("postgresql://test")


def test_parser_registry_uses_stable_metadata_without_forced_inference(tmp_path):
    source = tmp_path / "messy_001.md"
    source.write_text("第一条\n客户风险等级应与产品风险等级匹配", encoding="utf-8")

    parser = ParserRegistry.default()
    documents = parser.load(source, knowledge_base_id="kb-finance")

    assert len(documents) == 1
    doc = documents[0]
    assert isinstance(doc, Document)
    metadata = doc.metadata
    assert doc.text.startswith("第一条")
    assert metadata["knowledge_base_id"] == "kb-finance"
    assert metadata["filename"] == "messy_001.md"
    assert metadata["file_type"] == "md"
    assert metadata["page_number"] is None
    assert "title_hint" not in metadata
    assert "doc_title" not in metadata
    assert "regulatory_topic" not in metadata
    assert "department" not in metadata
    assert "source_name" not in metadata
    assert "page" not in metadata


def test_parser_registry_does_not_infer_markdown_heading_metadata(tmp_path):
    source = tmp_path / "risk_policy.md"
    source.write_text("# 适当性管理办法\n\n客户风险等级应与产品风险等级匹配", encoding="utf-8")

    parser = ParserRegistry.default()
    documents = parser.load(source, knowledge_base_id="kb-finance")

    metadata = documents[0].metadata
    assert metadata["knowledge_base_id"] == "kb-finance"
    assert "title_hint" not in metadata


def test_supported_suffixes_include_enterprise_office_web_data_formats_without_ocr_images():
    assert {
        ".md",
        ".txt",
        ".pdf",
        ".docx",
        ".csv",
        ".json",
        ".html",
        ".htm",
        ".xlsx",
        ".pptx",
    }.issubset(SUPPORTED_SUFFIXES)
    assert not {".png", ".jpg", ".jpeg", ".tif", ".tiff"} & SUPPORTED_SUFFIXES


def test_docx_parser_preserves_tables_as_markdown(tmp_path):
    docx_lib = pytest.importorskip("docx")
    source = tmp_path / "credit_policy.docx"
    document = docx_lib.Document()
    document.add_heading("授信政策", level=1)
    document.add_paragraph("以下为行业准入参数。")
    table = document.add_table(rows=2, cols=3)
    table.cell(0, 0).text = "行业"
    table.cell(0, 1).text = "准入等级"
    table.cell(0, 2).text = "抵质押率上限"
    table.cell(1, 0).text = "制造业"
    table.cell(1, 1).text = "A"
    table.cell(1, 2).text = "70%"
    document.save(source)

    parsed = ParserRegistry.default().load(source, knowledge_base_id="kb-risk")

    assert len(parsed) == 1
    assert "授信政策" in parsed[0].text
    assert "| 行业 | 准入等级 | 抵质押率上限 |" in parsed[0].text
    assert "| 制造业 | A | 70% |" in parsed[0].text


def test_docx_parser_preserves_paragraph_and_table_order(tmp_path):
    docx_lib = pytest.importorskip("docx")
    source = tmp_path / "ordered_policy.docx"
    document = docx_lib.Document()
    document.add_paragraph("一、授信政策说明")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "客户类型"
    table.cell(0, 1).text = "额度上限"
    table.cell(1, 0).text = "小微企业"
    table.cell(1, 1).text = "500 万元"
    document.add_paragraph("二、表后审批要求")
    document.save(source)

    parsed = ParserRegistry.default().load(source, knowledge_base_id="kb-risk")

    assert len(parsed) == 1
    text = parsed[0].text
    before_index = text.index("一、授信政策说明")
    table_index = text.index("| 客户类型 | 额度上限 |")
    after_index = text.index("二、表后审批要求")
    assert before_index < table_index < after_index


def test_csv_parser_outputs_markdown_table(tmp_path):
    source = tmp_path / "metrics.csv"
    source.write_text("指标,2025,2026预算\n营业收入,12000,15000\n毛利率,31%,33%\n", encoding="utf-8-sig")

    parsed = ParserRegistry.default().load(source, knowledge_base_id="finance")

    assert len(parsed) == 1
    assert parsed[0].metadata["file_type"] == "csv"
    assert "| 指标 | 2025 | 2026预算 |" in parsed[0].text
    assert "| 毛利率 | 31% | 33% |" in parsed[0].text


def test_json_parser_outputs_readable_markdown(tmp_path):
    source = tmp_path / "risk_case.json"
    source.write_text(
        '{"case_id":"RC-2026-001","customer":{"name":"华东设备","rating":"A-"},"limits":[{"product":"流贷","amount":3000}]}',
        encoding="utf-8",
    )

    parsed = ParserRegistry.default().load(source, knowledge_base_id="risk")

    assert len(parsed) == 1
    assert "# risk_case.json" in parsed[0].text
    assert "- case_id: RC-2026-001" in parsed[0].text
    assert "customer.name: 华东设备" in parsed[0].text
    assert "limits[0].amount: 3000" in parsed[0].text


def test_html_parser_preserves_headings_lists_and_tables(tmp_path):
    source = tmp_path / "review.html"
    source.write_text(
        """
        <html><body>
          <h1>合同审查清单</h1>
          <p>适用于 SaaS 订阅合同。</p>
          <ul><li>确认数据出境条款</li><li>确认违约责任上限</li></ul>
          <table><tr><th>条款</th><th>风险</th></tr><tr><td>自动续费</td><td>中</td></tr></table>
        </body></html>
        """,
        encoding="utf-8",
    )

    parsed = ParserRegistry.default().load(source, knowledge_base_id="legal")

    assert len(parsed) == 1
    assert "# 合同审查清单" in parsed[0].text
    assert "- 确认数据出境条款" in parsed[0].text
    assert "| 条款 | 风险 |" in parsed[0].text
    assert "| 自动续费 | 中 |" in parsed[0].text


def test_xlsx_parser_outputs_each_sheet_as_markdown_table(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    source = tmp_path / "budget.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "经营预算"
    sheet.append(["科目", "2025实际", "2026预算"])
    sheet.append(["营业收入", 12000, 15000])
    sheet.append(["研发费用率", "8%", "9%"])
    second = workbook.create_sheet("现金流")
    second.append(["项目", "金额"])
    second.append(["经营性现金流", 2600])
    workbook.save(source)

    parsed = ParserRegistry.default().load(source, knowledge_base_id="finance")

    assert len(parsed) == 1
    assert "## 经营预算" in parsed[0].text
    assert "| 科目 | 2025实际 | 2026预算 |" in parsed[0].text
    assert "| 营业收入 | 12000 | 15000 |" in parsed[0].text
    assert "## 现金流" in parsed[0].text
    assert "| 经营性现金流 | 2600 |" in parsed[0].text


def test_pptx_parser_preserves_slide_text_and_tables(tmp_path):
    pptx = pytest.importorskip("pptx")
    source = tmp_path / "committee.pptx"
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "风险委员会汇报"
    textbox = slide.shapes.add_textbox(100000, 900000, 5000000, 800000)
    textbox.text = "本月新增预警客户 12 户。"
    table_shape = slide.shapes.add_table(2, 2, 100000, 1800000, 4000000, 1000000)
    table = table_shape.table
    table.cell(0, 0).text = "风险项"
    table.cell(0, 1).text = "等级"
    table.cell(1, 0).text = "逾期迁徙"
    table.cell(1, 1).text = "高"
    presentation.save(source)

    parsed = ParserRegistry.default().load(source, knowledge_base_id="risk")

    assert len(parsed) == 1
    assert "## Slide 1" in parsed[0].text
    assert "风险委员会汇报" in parsed[0].text
    assert "本月新增预警客户 12 户" in parsed[0].text
    assert "| 风险项 | 等级 |" in parsed[0].text
    assert "| 逾期迁徙 | 高 |" in parsed[0].text


def test_image_formats_are_not_registered_without_ocr(tmp_path):
    source = tmp_path / "scanned.png"
    source.write_bytes(b"mocked")

    assert ParserRegistry.default().load(source, knowledge_base_id="risk") == []


def test_document_registry_tracks_lifecycle_and_deduplicates_by_hash(tmp_path):
    registry = _registry()
    first_hash = "sha256:first"

    first = registry.upsert_uploaded(
        source_path=tmp_path / "policy.txt",
        filename="policy.txt",
        file_type="txt",
        content_hash=first_hash,
        knowledge_base_id="kb-finance",
    )
    duplicate = registry.find_by_hash(first_hash, "kb-finance")
    registry.mark_indexed(first.document_id, chunk_count=3)

    assert duplicate is not None
    assert duplicate.document_id == first.document_id
    assert registry.get(first.document_id).status == "indexed"
    assert registry.get(first.document_id).chunk_count == 3
    assert registry.get(first.document_id).filename == "policy.txt"


def test_document_registry_keeps_same_filename_until_system_retires_old_document(tmp_path):
    registry = _registry()
    first_path = tmp_path / "policy-v1.txt"
    second_path = tmp_path / "policy-v2.txt"
    first_path.write_text("旧制度", encoding="utf-8")
    second_path.write_text("新制度", encoding="utf-8")

    first = registry.upsert_uploaded(
        source_path=first_path,
        filename="policy.txt",
        file_type="txt",
        content_hash="sha256:v1",
        knowledge_base_id="kb-finance",
    )
    second = registry.upsert_uploaded(
        source_path=second_path,
        filename="policy.txt",
        file_type="txt",
        content_hash="sha256:v2",
        knowledge_base_id="kb-finance",
    )

    assert second.document_id != first.document_id
    assert registry.get(first.document_id).status == "uploaded"
    assert registry.get(second.document_id).status == "uploaded"
    assert first_path.exists()


def test_load_documents_uses_registry_and_filters_current_knowledge_base_id(tmp_path):
    kb_a = tmp_path / "kb-a-policy.txt"
    kb_b = tmp_path / "kb-b-policy.txt"
    deleted = tmp_path / "deleted.txt"
    kb_a.write_text("A 资料库 客户尽调", encoding="utf-8")
    kb_b.write_text("B 资料库 理财适当性", encoding="utf-8")
    deleted.write_text("已删除 旧制度", encoding="utf-8")
    registry = _registry()
    rec_a = registry.upsert_uploaded(
        source_path=kb_a,
        filename="a.txt",
        file_type="txt",
        content_hash=compute_content_hash(kb_a),
        knowledge_base_id="kb-a",
    )
    rec_b = registry.upsert_uploaded(
        source_path=kb_b,
        filename="b.txt",
        file_type="txt",
        content_hash=compute_content_hash(kb_b),
        knowledge_base_id="kb-b",
    )
    rec_deleted = registry.upsert_uploaded(
        source_path=deleted,
        filename="deleted.txt",
        file_type="txt",
        content_hash=compute_content_hash(deleted),
        knowledge_base_id="kb-a",
    )
    registry.mark_indexed(rec_a.document_id, chunk_count=1)
    registry.mark_indexed(rec_b.document_id, chunk_count=1)
    registry.mark_deleted(rec_deleted.document_id)

    kb_a_documents = load_documents(tmp_path, knowledge_base_id="kb-a", document_registry=registry)
    kb_b_documents = load_documents(tmp_path, knowledge_base_id="kb-b", document_registry=registry)

    assert {doc.metadata["knowledge_base_id"] for doc in kb_a_documents} == {"kb-a"}
    assert {doc.metadata["filename"] for doc in kb_a_documents} == {"a.txt"}
    assert {doc.metadata["knowledge_base_id"] for doc in kb_b_documents} == {"kb-b"}
    assert {doc.metadata["filename"] for doc in kb_b_documents} == {"b.txt"}


def test_load_documents_skips_registry_source_paths_outside_data_root(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    external = tmp_path / "external.md"
    external.write_text("# 外部文件\n不应被索引", encoding="utf-8")
    registry = _registry()
    registry.upsert_uploaded(
        source_path=external,
        filename="external.md",
        file_type="md",
        content_hash=compute_content_hash(external),
        knowledge_base_id="kb-finance",
    )

    documents = load_documents(data_root, knowledge_base_id="kb-finance", document_registry=registry)

    assert documents == []


def test_load_documents_accepts_registered_paths_inside_data_root(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    source = data_root / "policy.md"
    source.write_text("# 制度\n客户尽调", encoding="utf-8")
    registry = _registry()
    record = registry.upsert_uploaded(
        source_path=source,
        filename="policy.md",
        file_type="md",
        content_hash=compute_content_hash(source),
        knowledge_base_id="kb-finance",
    )
    registry.mark_indexed(record.document_id, chunk_count=1)

    documents = load_documents(data_root, knowledge_base_id="kb-finance", document_registry=registry)

    assert len(documents) == 1
    assert documents[0].metadata["document_id"] == record.document_id
    assert documents[0].metadata["filename"] == "policy.md"


def test_document_registry_public_list_hides_internal_paths_and_hashes(tmp_path):
    source = tmp_path / "policy.txt"
    source.write_text("制度内容", encoding="utf-8")
    registry = _registry()
    record = registry.upsert_uploaded(
        source_path=source,
        filename="policy.txt",
        file_type="txt",
        content_hash=compute_content_hash(source),
        knowledge_base_id="kb-finance",
    )
    registry.mark_indexed(record.document_id, chunk_count=1)

    public_record = registry.list_public()[0]

    assert "source_path" not in public_record
    assert "content_hash" not in public_record
    assert public_record["filename"] == "policy.txt"
    assert public_record["status"] == "indexed"
