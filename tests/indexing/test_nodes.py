import logging

import pytest
from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode, TransformComponent

import finrag.indexing.nodes as nodes_module
from finrag.indexing.nodes import DataPreparationModule, HierarchyBuilder, MetadataTransform, build_ingestion_pipeline


class StaticEmbedding(BaseEmbedding):
    def _get_query_embedding(self, query: str):
        return [0.1, 0.2, 0.3]

    def _get_text_embedding(self, text: str):
        return [0.1, 0.2, 0.3]

    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)


class FakeDoclingNodeParser(TransformComponent):
    @classmethod
    def class_name(cls) -> str:
        return "fake_docling_node_parser"

    def __call__(self, nodes, **kwargs):
        return [
            TextNode(text=node.text, metadata=dict(node.metadata or {}))
            if isinstance(node, Document)
            else node
            for node in nodes
        ]

    def get_nodes_from_documents(self, documents, **kwargs):
        return [
            TextNode(
                text=document.text,
                metadata=dict(document.metadata or {}),
            )
            for document in documents
        ]


def test_nodes_module_reuses_parser_supported_suffixes():
    assert not hasattr(DataPreparationModule, "make_text_node")


def test_data_preparation_module_rejects_legacy_chunking_arguments(tmp_path):
    with pytest.raises(TypeError, match="chunk_size"):
        DataPreparationModule(str(tmp_path), chunk_size=80)

    module = DataPreparationModule(str(tmp_path))

    assert not hasattr(module, "chunk_size")
    assert not hasattr(module, "chunk_overlap")

def test_build_ingestion_pipeline_accepts_metadata_transform(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes_module, "_make_docling_node_parser", lambda: FakeDoclingNodeParser())
    module = DataPreparationModule(str(tmp_path))

    pipeline = build_ingestion_pipeline(
        module,
        StaticEmbedding(),
    )

    assert any(isinstance(transform, MetadataTransform) for transform in pipeline.transformations)
    assert all(isinstance(transform, TransformComponent) for transform in pipeline.transformations)


def test_build_ingestion_pipeline_uses_docling_then_hierarchy_then_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes_module, "_make_docling_node_parser", lambda: FakeDoclingNodeParser())
    module = DataPreparationModule(str(tmp_path))

    pipeline = build_ingestion_pipeline(
        module,
        StaticEmbedding(),
    )

    hierarchy_index = next(index for index, transform in enumerate(pipeline.transformations) if isinstance(transform, HierarchyBuilder))
    metadata_index = next(
        index
        for index, transform in enumerate(pipeline.transformations)
        if isinstance(transform, MetadataTransform)
    )
    assert hierarchy_index < metadata_index
    assert not any(transform.__class__.__name__ == "MarkdownElementNodeParser" for transform in pipeline.transformations)


def test_ingestion_pipeline_keeps_table_text_without_markdown_element_wrapper(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(nodes_module, "_make_docling_node_parser", lambda: FakeDoclingNodeParser())
    module = DataPreparationModule(str(tmp_path))
    pipeline = build_ingestion_pipeline(
        module,
        StaticEmbedding(),
    )
    assert getattr(pipeline, "vector_store", None) is None
    assert getattr(pipeline, "docstore", None) is None
    document = Document(
        text="# 指标\n\n| 指标 | 值 |\n| --- | --- |\n| 收入 | 100 |",
        metadata={
            "document_id": "doc-table",
            "knowledge_base_id": "finance",
            "filename": "table.md",
            "file_type": "md",
        },
    )

    caplog.set_level(logging.WARNING)
    nodes = pipeline.run(documents=[document], show_progress=False)

    assert nodes
    assert "Structured response error" not in caplog.text
    assert not any("with the following columns" in node.text for node in nodes if isinstance(node, TextNode))


def test_nodes_build_three_level_chunks_and_index_only_l3(tmp_path):
    source = tmp_path / "policy.md"
    source.write_text("# 适当性制度\n客户风险等级应与产品风险等级匹配，并进行风险揭示。\n" * 30, encoding="utf-8")
    module = DataPreparationModule(str(tmp_path), knowledge_base_id="kb-finance")
    documents = module.load_documents()
    module._parse_docling_leaf_nodes = lambda docs: [
        TextNode(text=doc.text, metadata=dict(doc.metadata or {}) | {"headings": ["适当性制度"]})
        for doc in docs
    ]
    all_nodes, nodes = module._build_hierarchical_nodes(documents)
    module.load_prepared_nodes(all_nodes)

    assert nodes
    assert all(isinstance(node, TextNode) for node in nodes)
    assert all(node.metadata["chunk_level"] == 3 for node in nodes)


def test_nodes_chunk_one_registered_document_without_loading_others(tmp_path):
    from types import SimpleNamespace

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text(("第一份资料 客户风险等级匹配\n" * 50), encoding="utf-8")
    second.write_text(("第二份资料 销售留痕管理\n" * 50), encoding="utf-8")

    module = DataPreparationModule(str(tmp_path), knowledge_base_id="kb-finance")
    module.load_documents()
    assert len(module.documents) == 2
    module._parse_docling_leaf_nodes = lambda docs: [
        TextNode(text=doc.text, metadata=dict(doc.metadata or {}) | {"headings": ["单文档"]})
        for doc in docs
    ]

    record = SimpleNamespace(
        document_id="doc-test",
        source_path=str(first),
        filename="first.txt",
        file_type="txt",
        knowledge_base_id="kb-finance",
        status="indexed",
    )
    all_nodes, leaf_nodes = module.chunk_single_document(record)
    assert leaf_nodes
    assert all(isinstance(node, TextNode) for node in leaf_nodes)


def test_nodes_chunk_one_registered_document_keeps_markdown_table_leaf(tmp_path, monkeypatch):
    from types import SimpleNamespace

    source = tmp_path / "annual.md"
    source.write_text(
        "\n\n".join(
            [
                "## 主要经营指标",
                "\n".join(
                    [
                        "| 指标 | 2024年 | 2025年 |",
                        "| --- | --- | --- |",
                        "| 营业收入 | 115.0 | 128.6 |",
                        "| 归母净利润 | 13.1 | 14.2 |",
                        "| 综合毛利率 | 41.8% | 42.6% |",
                        "| 经营现金流 | 15.4 | 18.7 |",
                        "| 应收账款周转天数 | 69 | 76 |",
                    ]
                ),
                "后续说明 " * 80,
            ]
        ),
        encoding="utf-8",
    )
    record = SimpleNamespace(
        document_id="doc-table",
        source_path=str(source),
        filename="annual.md",
        file_type="md",
        knowledge_base_id="kb-finance",
        status="uploaded",
    )

    module = DataPreparationModule(str(tmp_path), knowledge_base_id="kb-finance")
    monkeypatch.setattr(nodes_module, "load_docling_documents", lambda path, *, knowledge_base_id, data_root=None: [
        Document(
            text=source.read_text(encoding="utf-8"),
            metadata={
                "document_id": record.document_id,
                "knowledge_base_id": knowledge_base_id,
                "filename": record.filename,
                "file_type": record.file_type,
                "parser_name": "docling",
            },
        )
    ])
    module._parse_docling_leaf_nodes = lambda docs: [
        TextNode(text=doc.text, metadata=dict(doc.metadata or {}) | {"headings": ["主要经营指标"]})
        for doc in docs
    ]
    _all_nodes, leaf_nodes = module.chunk_single_document(record)

    assert any(
        all(value in node.text for value in ["营业收入", "115.0", "归母净利润", "13.1", "经营现金流", "15.4", "18.7"])
        for node in leaf_nodes
    )


def test_hierarchy_builder_keeps_only_needed_docling_metadata(tmp_path):
    document_metadata = {
        "document_id": "doc-docling",
        "knowledge_base_id": "kb-finance",
        "source_path": str(tmp_path / "annual.pdf"),
        "filename": "annual.pdf",
        "file_type": "pdf",
        "parser_name": "docling",
    }
    leaf = TextNode(
        text="授信风险矩阵表格",
        metadata={
            **document_metadata,
            "headings": ["二、授信风险矩阵"],
            "doc_items": [{"prov": [{"page_no": 3, "bbox": {"l": 1}}]}],
            "bbox": {"l": 1},
            "origin": {"filename": "annual.pdf"},
        },
    )

    all_nodes = list(HierarchyBuilder()([leaf]))
    leaves = [node for node in all_nodes if isinstance(node, TextNode) and not node.child_nodes]

    assert len(all_nodes) == 3
    assert len(leaves) == 1
    assert leaves[0].metadata["section_title"] == "二、授信风险矩阵"
    assert leaves[0].metadata["page_number"] == 3
    for key in ("headings", "doc_items", "bbox", "origin", "element_type"):
        assert key not in leaves[0].metadata


def test_docling_page_and_section_metadata_reaches_leaf_nodes(tmp_path):
    module = DataPreparationModule(str(tmp_path), knowledge_base_id="kb-finance")
    document = Document(
        text="管理层讨论与分析显示，营业收入增长来自财富管理业务和机构客户服务。" * 8,
        metadata={
            "document_id": "doc-docling",
            "knowledge_base_id": "kb-finance",
            "filename": "annual.pdf",
            "file_type": "pdf",
            "parser_name": "docling",
        },
    )
    module._parse_docling_leaf_nodes = lambda docs: [
        TextNode(
            text=doc.text,
            metadata=dict(doc.metadata or {})
            | {
                "headings": ["管理层讨论与分析"],
                "doc_items": [{"prov": [{"page_no": 2, "bbox": {"l": 1}}]}],
                "bbox": {"l": 1},
            },
        )
        for doc in docs
    ]

    _all_nodes, leaf_nodes = module._build_hierarchical_nodes([document])

    assert leaf_nodes
    assert {node.metadata["page_number"] for node in leaf_nodes} == {2}
    assert {node.metadata["section_title"] for node in leaf_nodes} == {"管理层讨论与分析"}
    assert {node.metadata["parser_name"] for node in leaf_nodes} == {"docling"}
    for key in ("element_type", "bbox", "doc_items", "origin", "headings"):
        assert all(key not in node.metadata for node in leaf_nodes)


def test_nodes_no_longer_expose_removed_context_methods():
    assert not hasattr(DataPreparationModule, "chunk_documents")
    assert not hasattr(DataPreparationModule, "get_context_nodes")
    assert not hasattr(DataPreparationModule, "_deterministic_auto_merge_nodes")
    assert not hasattr(DataPreparationModule, "_merge_leaf_groups_to_parents")
    assert not hasattr(DataPreparationModule, "_merge_parent_groups_to_roots")
    assert not hasattr(DataPreparationModule, "_source_get")
    assert not hasattr(DataPreparationModule, "_apply_token_budget")
    assert not hasattr(DataPreparationModule, "last_auto_merge_trace")
