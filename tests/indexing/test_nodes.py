from llama_index.core.schema import TextNode

from finrag.indexing.nodes import DataPreparationModule


def test_nodes_module_reuses_parser_supported_suffixes():
    assert not hasattr(DataPreparationModule, "make_text_node")


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


def test_nodes_no_longer_expose_removed_context_methods():
    assert not hasattr(DataPreparationModule, "chunk_documents")
    assert not hasattr(DataPreparationModule, "get_context_nodes")
    assert not hasattr(DataPreparationModule, "_deterministic_auto_merge_nodes")
    assert not hasattr(DataPreparationModule, "_merge_leaf_groups_to_parents")
    assert not hasattr(DataPreparationModule, "_merge_parent_groups_to_roots")
    assert not hasattr(DataPreparationModule, "_source_get")
    assert not hasattr(DataPreparationModule, "_apply_token_budget")
    assert not hasattr(DataPreparationModule, "last_auto_merge_trace")
