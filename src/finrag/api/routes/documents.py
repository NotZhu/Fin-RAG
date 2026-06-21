"""文档生命周期路由"""

from __future__ import annotations

import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath

from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile

from finrag.api.errors import _build_document_not_found_response, _build_error_response
from finrag.api.rag_service import RAGService
from finrag.core.config import RAGConfig, validate_knowledge_base_id
from finrag.ingestion.parsers import SUPPORTED_SUFFIXES
from finrag.storage.knowledge_base_registry import KnowledgeBaseArchivedError

_DEFAULT_CONFIG = RAGConfig() # 路由层默认配置快照
DEFAULT_UPLOAD_DIR = Path(_DEFAULT_CONFIG.upload_dir) # 默认上传临时目录
MAX_UPLOAD_BYTES = _DEFAULT_CONFIG.max_upload_bytes # 默认单文件上传大小上限
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024 # 上传流式读取块大小


def _safe_upload_filename(filename: str | None) -> str:
    """
    从客户端上传文件名中提取安全的基础文件名
    Args:
        filename: 客户端提交的原始文件名
    Returns:
        去除路径信息后的文件名，包含扩展名，空值时返回 upload
    """
    raw_filename = (filename or "upload").strip()
    return PureWindowsPath(PurePosixPath(raw_filename).name).name or "upload"


def register_document_routes(app: FastAPI, service: RAGService, upload_dir: str | Path | None = None) -> None:
    """
    注册文档上传、列表、删除和重建索引路由
    Args:
        app: 需要注册路由的 FastAPI 应用实例
        service: 访问 RAG 系统的服务适配器
        upload_dir: 上传文件的临时落盘目录
    """
    # 上传目录重写
    upload_root_override = Path(upload_dir) if upload_dir is not None else None

    def _resolve_knowledge_base_id(value: str) -> str:
        """
        从请求参数中解析知识库 ID，确保其安全且存在
        Args:
            value: 请求中的 knowledge_base_id 字段
        Returns:
            通过校验的知识库 ID
        """
        return validate_knowledge_base_id(value)

    def _archived_response(knowledge_base_id: str, request_id: str):
        """
        构建知识库已归档的错误响应
        Args:
            knowledge_base_id: 知识库 ID
            request_id: 当前 HTTP 请求 ID
        Returns:
            包含错误信息的统一响应响应字典
        """
        return _build_error_response(
            code="knowledge_base_archived",
            message=f"知识库 {knowledge_base_id!r} 已归档",
            request_id=request_id,
            status_code=409,
        )

    @app.get("/knowledge-bases/{knowledge_base_id}/documents")
    def list_knowledge_base_documents(knowledge_base_id: str, request: Request):
        """
        返回指定知识库的公开文档列表
        Args:
            knowledge_base_id: 知识库 ID
            request: 当前 HTTP 请求
        Returns:
            包含 documents 列表的响应字典
        """
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
        system = service.get_system()
        try:
            resolved_knowledge_base_id = _resolve_knowledge_base_id(knowledge_base_id)
        except ValueError as exc:
            return _build_error_response(
                code="invalid_knowledge_base_id",
                message=str(exc),
                request_id=request_id,
                status_code=400,
            )
        return {"documents": system.list_documents(resolved_knowledge_base_id)}

    @app.post("/knowledge-bases/{knowledge_base_id}/documents/upload")
    async def upload_knowledge_base_document(
        knowledge_base_id: str,
        request: Request,
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        async_index: bool = Form(False),
    ) -> dict:
        """
        接收上传文档并写入指定知识库
        Args:
            knowledge_base_id: 知识库 ID
            request: 当前 HTTP 请求
            background_tasks: FastAPI 后台任务队列
            file: 上传的文档文件
            async_index: 是否异步执行索引构建
        Returns:
            文档注册或索引结果；失败时返回统一错误响应
        """
        # 从请求状态获取或生成唯一请求 ID
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
        system = service.get_system()
        # 获取当前配置
        config = getattr(system, "config", None) or _DEFAULT_CONFIG
        # 上传目录配置参数
        configured_upload_dir = Path(config.upload_dir)
        # 最大上传大小配置参数
        max_upload_bytes = int(config.max_upload_bytes)
        # 上传目录重写或默认配置
        upload_root = upload_root_override or configured_upload_dir
        # 验证并处理请求中的资料库 ID
        try:
            knowledge_base_id = _resolve_knowledge_base_id(knowledge_base_id)
        except ValueError as exc:
            # 处理无效的资料库 ID
            return _build_error_response(
                code="invalid_knowledge_base_id",
                message=str(exc),
                request_id=request_id,
                status_code=400,
            )
        
        # 提取安全的基础文件名
        safe_filename = _safe_upload_filename(file.filename)
        # 获取文件扩展名并检查是否支持上传
        suffix = Path(safe_filename).suffix
        if suffix.lower() not in SUPPORTED_SUFFIXES:
            return _build_error_response(
                code="unsupported_file_type",
                message=f"仅支持 {sorted(SUPPORTED_SUFFIXES)} 文件",
                request_id=request_id,
                status_code=400,
            )
        
        # 创建上传目录（如果不存在）
        upload_root.mkdir(parents=True, exist_ok=True)
        # 生成临时文件路径
        temp_path = upload_root / f"{uuid.uuid4().hex}{suffix}"
        # 已读取的字节数
        total = 0

        with temp_path.open("wb") as output: # 以二进制写入模式打开临时文件
            # 异步读取上传文件内容
            while chunk := await file.read(UPLOAD_READ_CHUNK_BYTES):
                total += len(chunk)
                # 检查文件大小是否超过最大限制
                if total > max_upload_bytes:
                    # 关闭临时文件
                    output.close()
                    # 删除已读取的临时文件
                    temp_path.unlink(missing_ok=True)
                    return _build_error_response(
                        code="file_too_large",
                        message=f"文件大小不能超过 {max_upload_bytes} 字节",
                        request_id=request_id,
                        status_code=400,
                    )
                # 写入当前读取的块到临时文件
                output.write(chunk)
            
        # 检查上传文件是否为空
        if total == 0:
            # 删除空文件
            temp_path.unlink(missing_ok=True)
            return _build_error_response(
                code="empty_file",
                message="上传文件不能为空",
                request_id=request_id,
                status_code=400,
            )

        try:
            # 异步索引文档
            if async_index:
                # 准备上传文件的公开文档记录
                prepared = system.prepare_uploaded_file(temp_path, safe_filename, knowledge_base_id)
                
                # 检查文档未索引
                if prepared["status"] != "indexed":
                    # 把文档索引任务添加到后台任务队列
                    background_tasks.add_task(system.index_registered_document, prepared["document_id"])
                return prepared
            # 同步索引文档
            return system.ingest_uploaded_file(temp_path, safe_filename, knowledge_base_id)
        except KnowledgeBaseArchivedError:
            # 知识库已归档
            return _archived_response(knowledge_base_id, request_id)
        finally:
            # 删除临时文件
            temp_path.unlink(missing_ok=True)

    @app.delete("/knowledge-bases/{knowledge_base_id}/documents/{document_id}")
    def delete_knowledge_base_document(knowledge_base_id: str, document_id: str, request: Request):
        """
        删除指定知识库中的文档及其索引数据
        Args:
            knowledge_base_id: 知识库 ID
            document_id: 需要删除的文档 ID
            request: 当前 HTTP 请求
        Returns:
            删除结果；文档不存在或不属于当前知识库时返回 404 错误响应
        """
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
        system = service.get_system()
        try:
            resolved_knowledge_base_id = _resolve_knowledge_base_id(knowledge_base_id)
        except ValueError as exc:
            return _build_error_response(
                code="invalid_knowledge_base_id",
                message=str(exc),
                request_id=request_id,
                status_code=400,
            )
        try:
            return system.delete_document(document_id, resolved_knowledge_base_id)
        except KnowledgeBaseArchivedError:
            # 知识库已归档
            return _archived_response(resolved_knowledge_base_id, request_id)
        except KeyError:
            # 文档不存在
            return _build_document_not_found_response(document_id, request_id)

    @app.post("/knowledge-bases/{knowledge_base_id}/documents/{document_id}/reindex")
    def reindex_knowledge_base_document(knowledge_base_id: str, document_id: str, request: Request):
        """
        对指定知识库中的文档重新解析并重建索引
        Args:
            knowledge_base_id: 知识库 ID
            document_id: 需要重建索引的文档 ID
            request: 当前 HTTP 请求
        Returns:
            重建索引结果；文档不存在或不属于当前知识库时返回 404 错误响应
        """
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
        system = service.get_system()
        try:
            resolved_knowledge_base_id = _resolve_knowledge_base_id(knowledge_base_id)
        except ValueError as exc:
            return _build_error_response(
                code="invalid_knowledge_base_id",
                message=str(exc),
                request_id=request_id,
                status_code=400,
            )
        try:
            return system.reindex_document(document_id, resolved_knowledge_base_id)
        except KnowledgeBaseArchivedError:
            # 知识库已归档
            return _archived_response(resolved_knowledge_base_id, request_id)
        except KeyError:
            return _build_document_not_found_response(document_id, request_id)
