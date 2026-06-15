"""Managed source file path and promotion helpers."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from finrag.core.config import RAGConfig
from finrag.ingestion import is_path_within

logger = logging.getLogger(__name__)


class ManagedSourceFileService:
    """托管源文件服务类，负责管理文档源文件的路径和迁移"""

    def __init__(self, config: RAGConfig, document_registry: Any):
        """
        初始化托管源文件服务类
        Args:
            config: 配置对象
            document_registry: 文档注册表对象
        """
        self.config = config
        self.document_registry = document_registry

    def delete_managed_source_file(self, source_path: str | Path) -> None:
        """
        删除托管目录下的文档源文件
        Args:
            source_path: 文档源文件路径
        """
        path = Path(source_path)
        data_root = Path(self.config.data_path)
        if not is_path_within(path, data_root):
            logger.warning("跳过删除托管目录外的文档源文件: %s", path)
            return
        # 删除文件
        if path.exists():
            path.unlink()

    def pending_source_path(self, record: Any) -> Path:
        """返回待处理源文件路径"""
        return Path(self.config.data_path) / ".pending" / record.document_id / self.safe_source_filename(record.filename)

    def final_source_path(self, record: Any) -> Path:
        """
        返回文档源文件的最终托管路径
        Args:
            record: 文档记录对象
        Returns:
            文档源文件的最终托管路径
        """
        filename = self.safe_source_filename(record.filename)
        final_path = Path(self.config.data_path) / filename
        # 检查是否有其他文档使用相同的文件名
        for other in self.document_registry.records.values():
            if other.document_id == record.document_id or other.status == "deleted":
                continue
            # 检查其他文档的源文件路径是否与当前文档的最终托管路径相同
            if Path(other.source_path).resolve() == final_path.resolve():
                return Path(self.config.data_path) / record.knowledge_base_id / filename
        return final_path

    def promote_document_source_file(self, record: Any) -> None:
        """
        提升文档源文件到管理目录
        Args:
            record: 文档记录对象
        """
        # 取出当前文档源文件路径 data/.pending/<document_id>/<filename>
        source_path = Path(record.source_path)
        # 计算索引成功后的最终保存路径 data/<filename> ，如果其他文档使用相同的文件名 data/<knowledge_base_id>/<filename>
        final_path = self.final_source_path(record)
        # 如果当前路径已经是最终路径，就不用移动
        if source_path.resolve() == final_path.resolve():
            return
        # 确保最终路径的目录存在，否则创建
        final_path.parent.mkdir(parents=True, exist_ok=True)
        # 如果最终路径已存在旧文件，先删除
        if final_path.exists():
            final_path.unlink()
        # 移动文件到最终路径
        shutil.move(str(source_path), str(final_path))
        try:
            # 删除原来的空目录
            source_path.parent.rmdir()
        except OSError:
            pass
        # 更新文档记录的源文件路径
        self.document_registry.update_source_path(record.document_id, str(final_path))

    @staticmethod
    def safe_source_filename(filename: str | None, fallback_suffix: str = "") -> str:
        """
        返回安全的源文件名，添加回退后缀（如果需要）
        Args:
            filename: 原始文件名
            fallback_suffix: 回退后缀（可选）
        Returns:
            安全的源文件名
        """
        # 原始文件名
        raw_filename = (filename or "upload").strip()
        # 安全文件名
        safe_filename = PureWindowsPath(PurePosixPath(raw_filename).name).name or "upload"
        # 如果安全文件名没有后缀，且指定了回退后缀
        if not Path(safe_filename).suffix and fallback_suffix:
            # 添加回退后缀
            safe_filename = f"{safe_filename}{fallback_suffix.lower()}"
        return safe_filename
