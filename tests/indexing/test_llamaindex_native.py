import os

import pytest
from llama_index.core import Document
from llama_index.core.schema import NodeWithScore, TextNode

from finrag.indexing import DataPreparationModule, IndexConstructionModule

requires_live_vector_stack = pytest.mark.skipif(
    os.getenv("FINRAG_RUN_INTEGRATION") != "1",
    reason="需要运行中的 Milvus 和 DashScope embedding",
)


def test_nodes_use_llamaindex_native_documents_and_nodes(tmp_path):
    source = tmp_path / "policy.md"
    source.write_text("# 适当性制度\n客户风险等级应与产品风险等级匹配", encoding="utf-8")

    module = DataPreparationModule(str(tmp_path), knowledge_base_id="kb-finance", chunk_size=80, chunk_overlap=10)
    documents = module.load_documents()
    all_nodes, nodes = module._build_hierarchical_nodes(documents)
    module.load_prepared_nodes(all_nodes)

    assert documents
    assert isinstance(documents[0], Document)
    assert nodes
    assert all(isinstance(node, TextNode) for node in nodes)
    assert all(node.metadata["chunk_level"] == 3 for node in nodes)


@requires_live_vector_stack
def test_milvus_index_uses_llamaindex_dense_vector_store(tmp_path):
    nodes = [
        TextNode(
            id_="n1",
            text=
            "客户风险等级应与产品风险等级匹配",
            metadata={"chunk_id": "n1", "knowledge_base_id": "kb-finance", "document_id": "doc-1"},
        )
    ]
    module = IndexConstructionModule()

    index = module.build_vector_index(nodes)

    assert module.vector_store is not None
    assert module.vector_store.collection_name == "finrag_leaf_nodes"
    assert module.vector_store.embedding_field == "dense_embedding"
    assert index is module.index


@requires_live_vector_stack
def test_retrieval_outputs_llamaindex_node_with_score(tmp_path):
    nodes = [
        TextNode(text="反洗钱 客户尽职调查 受益所有人", id_="a", metadata={"chunk_id": "a", "knowledge_base_id": "kb-finance"}),
        TextNode(text="员工报销 发票", id_="b", metadata={"chunk_id": "b", "knowledge_base_id": "kb-finance"}),
    ]
    module = IndexConstructionModule()
    vector_index = module.build_vector_index(nodes)

    results = vector_index.as_retriever(similarity_top_k=1).retrieve("受益所有人识别")

    assert results
    assert isinstance(results[0], NodeWithScore)
    assert isinstance(results[0].node, TextNode)
    assert results[0].score is not None
