"""文档生命周期服务类"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from finrag.core.config import validate_knowledge_base_id
from finrag.ingestion import compute_content_hash


class DocumentLifecycleService:
    """文档生命周期服务类，负责管理文档的注册、索引、删除和重新索引流程"""

    def __init__(self, system: Any):
        self.system = system

    def prepare_uploaded_file(
        self,
        file_path: Path,
        filename: str,
        knowledge_base_id: str,
    ) -> dict:
        """
        准备上传的文件，将其复制到管理的待处理源区域
        Args:
            file_path: 上传文件的本地路径(临时文件目录 upload_root/<uuid>.<suffix>)
            filename: 原始文件名
            knowledge_base_id: 关联的知识库ID
        Returns:
            包含文档ID的响应字典
        """
        system = self.system
        with system._write_lock:
            # 验证知识库ID
            knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
            # 确保数据路径存在
            Path(system.config.data_path).mkdir(parents=True, exist_ok=True)

            # 源文件服务实例
            source_files = system._managed_source_files()
            # 生成安全的源文件名
            safe_filename = source_files.safe_source_filename(filename, file_path.suffix)
            # 计算文件内容哈希
            content_hash = compute_content_hash(file_path)
            # 创建或更新文档记录
            record = system.document_registry.upsert_uploaded(
                source_path=file_path, # 上传文件的本地路径
                filename=safe_filename, # 安全的源文件名
                file_type=Path(safe_filename).suffix.lower().lstrip(".") or file_path.suffix.lower().lstrip("."), # 文件类型
                content_hash=content_hash, # 文件内容哈希
                knowledge_base_id=knowledge_base_id, # 关联的知识库ID
            )
            
            # 如果记录已索引，说明文件是重复上传，直接返回
            if record.status == "indexed":
                # 返回文档记录
                return system._public_document(record.document_id)

            # 如果记录未索引，复制文件到待处理源区域
            # 计算待处理源文件路径 data/.pending/<document_id>/<filename>
            target_path = source_files.pending_source_path(record)
            # 确保待处理源文件路径存在
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 如果文件路径不是待处理源文件路径，复制文件
            if file_path.resolve() != target_path.resolve():
                # 复制文件到待处理源文件路径
                shutil.copyfile(file_path, target_path)
                # 更新文档记录的源文件路径
                system.document_registry.update_source_path(record.document_id, str(target_path))
            # 返回文档记录
            return system._public_document(record.document_id)

    def index_registered_document(self, document_id: str) -> dict:
        """
        索引已注册的文档并提升其管理的源文件
        """
        system = self.system
        with system._write_lock:
            # 标记文档为解析中
            system.document_registry.mark_parsing(document_id)
            try:
                # 索引文档
                system._index_document_locked(document_id, retire_replacements=True)
                # 提升文档源文件到管理目录
                system._managed_source_files().promote_document_source_file(system.document_registry.get(document_id))
                # 返回文档记录
                return system._public_document(document_id)
            except Exception as exc:
                # 标记文档为索引失败
                system.document_registry.mark_failed(document_id, system._format_exception(exc))
                raise

    def ingest_uploaded_file(
        self,
        file_path: Path,
        filename: str,
        knowledge_base_id: str,
    ) -> dict:
        """
        索引上传的文件并提升其管理的源文件
        Args:
            file_path: 上传文件的本地路径(临时文件目录)
            filename: 原始文件名
            knowledge_base_id: 关联的知识库ID
        Returns:
            包含文档ID的响应字典
        """
        system = self.system
        with system._write_lock:
            # 准备上传文件
            prepared = self.prepare_uploaded_file(file_path, filename, knowledge_base_id)
            if prepared["status"] == "indexed":
                return prepared
            try:
                # 索引文档
                return self.index_registered_document(prepared["document_id"])
            except Exception as exc:
                # 标记文档为索引失败
                system.document_registry.mark_failed(prepared["document_id"], system._format_exception(exc))
                raise

    def delete_document(self, document_id: str) -> dict:
        """
        删除指定文档及其索引数据
        Args:
            document_id: 需要删除的文档 ID
        Returns:
            公开的文档记录字典
        """
        system = self.system
        with system._write_lock:
            # 获取文档记录
            record = system.document_registry.get(document_id)
            # 确保增量索引准备就绪
            system._ensure_incremental_index_ready_locked()
            # 删除管理源文件
            system._managed_source_files().delete_managed_source_file(record.source_path)
            # 删除文档索引条目
            system._delete_document_index_entries_locked(document_id)
            # 标记文档为已删除
            system.document_registry.mark_deleted(document_id)
            # 刷新文档索引
            system._reload_from_store_and_refresh_locked()
            # 返回文档记录
            return system._public_document(document_id)

    def reindex_document(self, document_id: str) -> dict:
        """
        对指定文档重新解析并重建索引
        Args:
            document_id: 需要重建索引的文档 ID
        Returns:
            公开的文档记录字典
        """
        system = self.system
        with system._write_lock:
            # 标记文档为解析中
            system.document_registry.mark_parsing(document_id)
            try: 
                # 重建索引
                return system._index_document_locked(document_id, retire_replacements=False)
            except Exception as exc:
                # 标记文档为索引失败
                system.document_registry.mark_failed(document_id, system._format_exception(exc))
                raise
