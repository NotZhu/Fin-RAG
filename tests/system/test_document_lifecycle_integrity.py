import os
from pathlib import Path

import pytest

from finrag.core.config import RAGConfig
from finrag.application.system import FinRAGSystem

pytestmark = pytest.mark.skipif(
    os.getenv("FINRAG_RUN_INTEGRATION") != "1",
    reason="需要运行中的 Milvus 和 DashScope embedding",
)


def _system(tmp_path):
    return FinRAGSystem(
        RAGConfig(
            data_path=str(tmp_path / "data"),
            reranker_provider="none",
        )
    )


def test_same_filename_reupload_removes_old_source_from_index(tmp_path):
    first = tmp_path / "first.md"
    first.write_text("# 账户制度\n旧内容 旧客户资料", encoding="utf-8")
    second = tmp_path / "second.md"
    second.write_text("# 账户制度\n新内容 新授权书", encoding="utf-8")
    system = _system(tmp_path)

    first_record = system.ingest_uploaded_file(first, "账户制度.md", "kb-finance")
    second_record = system.ingest_uploaded_file(second, "账户制度.md", "kb-finance")

    assert first_record["document_id"] != second_record["document_id"]
    first_internal_record = system.document_registry.get(first_record["document_id"])
    assert not Path(first_internal_record.source_path).exists()
    assert system.retrieval_module is not None
    old_hits = system.retrieval_module.hybrid_search("旧客户资料", top_k=3, filters={"knowledge_base_id": "kb-finance"})
    new_hits = system.retrieval_module.hybrid_search("新授权书", top_k=3, filters={"knowledge_base_id": "kb-finance"})
    assert not old_hits or all("旧客户资料" not in hit.node.get_content() for hit in old_hits)
    assert new_hits and "新授权书" in new_hits[0].node.get_content()


def test_uploaded_documents_are_isolated_by_knowledge_base_id(tmp_path):
    finance = tmp_path / "finance.md"
    finance.write_text("# 理财制度\n客户风险等级匹配", encoding="utf-8")
    ops = tmp_path / "ops.md"
    ops.write_text("# 运营制度\n印鉴卡变更流程", encoding="utf-8")
    system = _system(tmp_path)

    system.ingest_uploaded_file(finance, "理财制度.md", "kb-finance")
    system.ingest_uploaded_file(ops, "运营制度.md", "kb-ops")

    assert system.retrieval_module is not None
    finance_hits = system.retrieval_module.hybrid_search("印鉴卡变更流程", top_k=3, filters={"knowledge_base_id": "kb-finance"})
    ops_hits = system.retrieval_module.hybrid_search("印鉴卡变更流程", top_k=3, filters={"knowledge_base_id": "kb-ops"})

    assert not finance_hits or all("印鉴卡" not in hit.node.get_content() for hit in finance_hits)
    assert ops_hits and "印鉴卡" in ops_hits[0].node.get_content()


def test_second_upload_uses_incremental_indexing_without_full_rebuild(monkeypatch, tmp_path):
    first = tmp_path / "first.md"
    first.write_text("# 第一份\n客户风险等级匹配", encoding="utf-8")
    second = tmp_path / "second.md"
    second.write_text("# 第二份\n销售留痕管理", encoding="utf-8")
    system = _system(tmp_path)

    system.ingest_uploaded_file(first, "first.md", "kb-finance")

    from finrag.indexing.milvus import IndexConstructionModule

    def fail_full_rebuild(*args, **kwargs):
        raise AssertionError("增量上传不应触发全量重建")

    monkeypatch.setattr(IndexConstructionModule, "build_vector_index", fail_full_rebuild)

    system.ingest_uploaded_file(second, "second.md", "kb-finance")

    hits = system.retrieval_module.hybrid_search("客户风险等级", top_k=3, filters={"knowledge_base_id": "kb-finance"})
    assert hits and "客户风险等级" in hits[0].node.get_content()


def test_delete_document_removes_only_target_document_from_incremental_index(tmp_path):
    first = tmp_path / "first.md"
    first.write_text("# 第一份\n独有术语甲 客户风险等级匹配", encoding="utf-8")
    second = tmp_path / "second.md"
    second.write_text("# 第二份\n独有术语乙 销售留痕管理", encoding="utf-8")
    system = _system(tmp_path)

    first_record = system.ingest_uploaded_file(first, "first.md", "kb-finance")
    system.ingest_uploaded_file(second, "second.md", "kb-finance")

    system.delete_document(first_record["document_id"], "kb-finance")

    old_hits = system.retrieval_module.hybrid_search("独有术语甲", top_k=3, filters={"knowledge_base_id": "kb-finance"})
    new_hits = system.retrieval_module.hybrid_search("独有术语乙", top_k=3, filters={"knowledge_base_id": "kb-finance"})
    assert not old_hits or all("独有术语甲" not in hit.node.get_content() for hit in old_hits)
    assert new_hits and "独有术语乙" in new_hits[0].node.get_content()


def test_same_filename_reupload_failure_keeps_old_document_searchable(monkeypatch, tmp_path):
    first = tmp_path / "first.md"
    first.write_text("# 账户制度\n旧客户资料保留", encoding="utf-8")
    second = tmp_path / "second.md"
    second.write_text("# 账户制度\n新授权书解析失败", encoding="utf-8")
    system = _system(tmp_path)
    first_record = system.ingest_uploaded_file(first, "账户制度.md", "kb-finance")

    prepared = system.prepare_uploaded_file(second, "账户制度.md", "kb-finance")

    from finrag.indexing.nodes import DataPreparationModule

    def fail_single_document(*args, **kwargs):
        raise RuntimeError("解析失败")

    monkeypatch.setattr(DataPreparationModule, "chunk_single_document", fail_single_document)

    with pytest.raises(RuntimeError, match="解析失败"):
        system.index_registered_document(prepared["document_id"])

    old_record = system.document_registry.get(first_record["document_id"])
    new_record = system.document_registry.get(prepared["document_id"])
    hits = system.retrieval_module.hybrid_search("旧客户资料保留", top_k=3, filters={"knowledge_base_id": "kb-finance"})
    assert old_record.status == "indexed"
    assert Path(old_record.source_path).exists()
    assert new_record.status == "failed"
    assert hits and "旧客户资料保留" in hits[0].node.get_content()


def test_restart_restores_retrieval_from_milvus_and_postgres_node_store(monkeypatch, tmp_path):
    source = tmp_path / "restart.md"
    source.write_text("# 重启制度\n重启恢复节点关系和关键词检索", encoding="utf-8")
    system = _system(tmp_path)
    system.ingest_uploaded_file(source, "restart.md", "kb-finance")

    from finrag.indexing.nodes import DataPreparationModule

    def fail_full_parse(*args, **kwargs):
        raise AssertionError("重启时应从 PostgreSQL 恢复节点，而不是执行全量切分")

    monkeypatch.setattr(DataPreparationModule, "chunk_documents", fail_full_parse)

    restarted = _system(tmp_path)
    restarted.initialize_system()
    restarted.build_knowledge_base()

    hits = restarted.retrieval_module.hybrid_search("重启恢复节点关系", top_k=3, filters={"knowledge_base_id": "kb-finance"})
    assert hits and "重启恢复节点关系" in hits[0].node.get_content()


def test_system_data_module_uses_persistent_node_store_for_context_recovery(tmp_path):
    system = _system(tmp_path)

    system.initialize_system()

    assert system.data_module is not None
    assert system.data_module.node_store is system.node_store


def test_upload_rejects_knowledge_base_id_that_escapes_data_root(tmp_path):
    source = tmp_path / "policy.md"
    source.write_text("# 制度\n客户尽调", encoding="utf-8")
    system = _system(tmp_path)

    try:
        system.prepare_uploaded_file(source, "policy.md", "../outside")
    except ValueError as exc:
        assert "knowledge_base_id" in str(exc)
    else:
        raise AssertionError("非法 knowledge_base_id 应被拒绝")

    assert not (tmp_path / "outside").exists()


def test_delete_document_does_not_unlink_registry_source_outside_data_root(tmp_path):
    external = tmp_path / "external.md"
    external.write_text("# 外部文件\n不应被删除", encoding="utf-8")
    system = _system(tmp_path)
    record = system.document_registry.upsert_uploaded(
        source_path=external,
        filename="external.md",
        file_type="md",
        content_hash="sha256:external",
        knowledge_base_id="kb-finance",
    )
    system.document_registry.mark_indexed(record.document_id, chunk_count=1)

    result = system.delete_document(record.document_id, "kb-finance")

    assert result["status"] == "deleted"
    assert external.exists()


def test_index_registered_document_marks_record_failed_when_indexing_fails(tmp_path):
    source = tmp_path / "policy.md"
    source.write_text("# 制度\n客户尽调", encoding="utf-8")
    system = _system(tmp_path)
    prepared = system.prepare_uploaded_file(source, "policy.md", "kb-finance")

    def fail_single_document(*args, **kwargs):
        raise RuntimeError("索引失败")

    system.initialize_system()
    system.data_module.chunk_single_document = fail_single_document

    with pytest.raises(RuntimeError, match="索引失败"):
        system.index_registered_document(prepared["document_id"])

    record = system.document_registry.get(prepared["document_id"])
    assert record.status == "failed"
    assert "RuntimeError: 索引失败" == record.last_error


def test_reindex_document_marks_record_failed_when_indexing_fails(tmp_path):
    source = tmp_path / "policy.md"
    source.write_text("# 制度\n客户尽调", encoding="utf-8")
    system = _system(tmp_path)
    prepared = system.prepare_uploaded_file(source, "policy.md", "kb-finance")
    system.document_registry.mark_indexed(prepared["document_id"], chunk_count=1)

    def fail_single_document(*args, **kwargs):
        raise RuntimeError("重建索引失败")

    system.initialize_system()
    system.data_module.chunk_single_document = fail_single_document

    with pytest.raises(RuntimeError, match="重建索引失败"):
        system.reindex_document(prepared["document_id"], "kb-finance")

    record = system.document_registry.get(prepared["document_id"])
    assert record.status == "failed"
    assert "RuntimeError: 重建索引失败" == record.last_error
