import logging

from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.schema import TextNode, TransformComponent

from finrag.indexing.nodes import (
    DataPreparationModule,
    FinRAGMetadataTransform,
    build_ingestion_pipeline,
)


class StaticEmbedding(BaseEmbedding):
    def _get_query_embedding(self, query: str):
        return [0.1, 0.2, 0.3]

    def _get_text_embedding(self, text: str):
        return [0.1, 0.2, 0.3]

    async def _aget_query_embedding(self, query: str):
        return self._get_query_embedding(query)


def test_nodes_module_reuses_parser_supported_suffixes():
    assert not hasattr(DataPreparationModule, "make_text_node")


def test_build_ingestion_pipeline_accepts_finrag_metadata_transform(tmp_path):
    module = DataPreparationModule(str(tmp_path), chunk_size=80, chunk_overlap=10)

    pipeline = build_ingestion_pipeline(
        module,
        StaticEmbedding(),
    )

    assert any(isinstance(transform, FinRAGMetadataTransform) for transform in pipeline.transformations)
    assert all(isinstance(transform, TransformComponent) for transform in pipeline.transformations)


def test_build_ingestion_pipeline_extracts_markdown_elements_before_chunking(tmp_path):
    module = DataPreparationModule(str(tmp_path), chunk_size=80, chunk_overlap=10)

    pipeline = build_ingestion_pipeline(
        module,
        StaticEmbedding(),
    )

    markdown_index = next(
        index
        for index, transform in enumerate(pipeline.transformations)
        if isinstance(transform, MarkdownElementNodeParser)
    )
    metadata_index = next(
        index
        for index, transform in enumerate(pipeline.transformations)
        if isinstance(transform, FinRAGMetadataTransform)
    )
    assert markdown_index < metadata_index


def test_ingestion_pipeline_handles_markdown_tables_without_llm_warning(tmp_path, caplog):
    module = DataPreparationModule(str(tmp_path), chunk_size=80, chunk_overlap=10)
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


def test_nodes_build_three_level_chunks_and_index_only_l3(tmp_path):
    source = tmp_path / "policy.md"
    source.write_text("# 适当性制度\n客户风险等级应与产品风险等级匹配，并进行风险揭示。\n" * 30, encoding="utf-8")
    module = DataPreparationModule(str(tmp_path), knowledge_base_id="kb-finance", chunk_size=80, chunk_overlap=10)
    documents = module.load_documents()
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


def test_nodes_chunk_one_registered_document_keeps_markdown_table_leaf(tmp_path):
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

    module = DataPreparationModule(str(tmp_path), knowledge_base_id="kb-finance", chunk_size=300, chunk_overlap=60)
    _all_nodes, leaf_nodes = module.chunk_single_document(record)

    assert any(
        all(value in node.text for value in ["营业收入", "115.0", "归母净利润", "13.1", "经营现金流", "15.4", "18.7"])
        for node in leaf_nodes
    )


def test_nodes_no_longer_expose_removed_context_methods():
    assert not hasattr(DataPreparationModule, "chunk_documents")
    assert not hasattr(DataPreparationModule, "get_context_nodes")
    assert not hasattr(DataPreparationModule, "_deterministic_auto_merge_nodes")
    assert not hasattr(DataPreparationModule, "_merge_leaf_groups_to_parents")
    assert not hasattr(DataPreparationModule, "_merge_parent_groups_to_roots")
    assert not hasattr(DataPreparationModule, "_source_get")
    assert not hasattr(DataPreparationModule, "_apply_token_budget")
    assert not hasattr(DataPreparationModule, "last_auto_merge_trace")
