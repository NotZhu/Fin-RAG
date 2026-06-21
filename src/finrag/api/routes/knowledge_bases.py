"""知识库管理路由"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, Request, status

from finrag.api.errors import _build_error_response
from finrag.api.rag_service import RAGService
from finrag.api.schemas import CreateKnowledgeBaseRequest
from finrag.core.config import validate_knowledge_base_id
from finrag.storage.knowledge_base_registry import (
    DuplicateKnowledgeBaseError,
    KnowledgeBaseNotFoundError,
    ProtectedKnowledgeBaseError,
)


def register_knowledge_base_routes(app: FastAPI, service: RAGService) -> None:
    """
    注册知识库创建和列表路由
    Args:
        app: FastAPI 应用
        service: RAG 服务适配器
    """

    def _request_id(request: Request) -> str:
        """
        从请求状态获取或生成唯一请求 ID
        Args:
            request: 当前 HTTP 请求
        Returns:
            唯一请求 ID
        """
        return getattr(request.state, "request_id", None) or uuid.uuid4().hex

    def _resolve_knowledge_base_id(value: str) -> str:
        """
        验证并返回解析后的知识库 ID
        Args:
            value: 输入的知识库 ID
        Returns:
            验证后的知识库 ID
        Raises:
            ValueError: 如果输入 ID 无效
        """
        return validate_knowledge_base_id(value)

    def _not_found(knowledge_base_id: str, request_id: str):
        """
        构建知识库不存在的错误响应
        Args:
            knowledge_base_id: 知识库 ID
            request_id: 当前 HTTP 请求 ID
        Returns:
            包含错误信息的统一响应响应字典
        """
        return _build_error_response(
            code="knowledge_base_not_found",
            message=f"知识库 {knowledge_base_id!r} 不存在",
            request_id=request_id,
            status_code=404,
        )

    def _protected(knowledge_base_id: str, request_id: str):
        """
        构建默认知识库保护的错误响应
        Args:
            knowledge_base_id: 知识库 ID
            request_id: 当前 HTTP 请求 ID
        Returns:
            包含错误信息的统一响应响应字典
        """
        return _build_error_response(
            code="default_knowledge_base_protected",
            message=f"默认知识库 {knowledge_base_id!r} 不允许归档或删除",
            request_id=request_id,
            status_code=409,
        )

    @app.get("/knowledge-bases")
    def list_knowledge_bases() -> dict:
        """
        列出所有知识库
        Returns:
            包含 knowledge_bases 列表的响应
        """
        return {"knowledge_bases": service.get_system().list_knowledge_bases()}

    @app.post("/knowledge-bases", status_code=status.HTTP_201_CREATED)
    def create_knowledge_base(payload: CreateKnowledgeBaseRequest, request: Request):
        """
        创建一个知识库
        Args:
            payload: 创建请求
            request: 当前 HTTP 请求
        Returns:
            新知识库公开记录
        """
        request_id = _request_id(request)
        try:
            return service.get_system().create_knowledge_base(
                payload.knowledge_base_id,
            )
        except DuplicateKnowledgeBaseError:
            return _build_error_response(
                code="duplicate_knowledge_base",
                message=f"知识库已存在: {payload.knowledge_base_id}",
                request_id=request_id,
                status_code=409,
            )
        except ValueError as exc:
            return _build_error_response(
                code="invalid_knowledge_base",
                message=str(exc),
                request_id=request_id,
                status_code=400,
            )

    @app.post("/knowledge-bases/{knowledge_base_id}/archive")
    def archive_knowledge_base(knowledge_base_id: str, request: Request):
        """
        归档一个知识库
        Args:
            knowledge_base_id: 知识库 ID
            request: 当前 HTTP 请求
        Returns:
            归档结果
        """
        request_id = _request_id(request)
        try:
            resolved_knowledge_base_id = _resolve_knowledge_base_id(knowledge_base_id)
        except ValueError as exc:
            # 处理无效的知识库 ID
            return _build_error_response(
                code="invalid_knowledge_base_id",
                message=str(exc),
                request_id=request_id,
                status_code=422,
            )
        try:
            # 归档知识库
            return service.get_system().archive_knowledge_base(resolved_knowledge_base_id)
        except ProtectedKnowledgeBaseError:
            # 处理默认知识库保护
            return _protected(resolved_knowledge_base_id, request_id)
        except KnowledgeBaseNotFoundError:
            # 处理知识库不存在
            return _not_found(resolved_knowledge_base_id, request_id)

    @app.post("/knowledge-bases/{knowledge_base_id}/restore")
    def restore_knowledge_base(knowledge_base_id: str, request: Request):
        """
        恢复一个归档的知识库
        Args:
            knowledge_base_id: 知识库 ID
            request: 当前 HTTP 请求
        Returns:
            恢复结果
        """
        request_id = _request_id(request)
        try:
            # 解析知识库 ID
            resolved_knowledge_base_id = _resolve_knowledge_base_id(knowledge_base_id)
        except ValueError as exc:
            # 处理无效的知识库 ID
            return _build_error_response(
                code="invalid_knowledge_base_id",
                message=str(exc),
                request_id=request_id,
                status_code=422,
            )
        try:
            # 恢复知识库
            return service.get_system().restore_knowledge_base(resolved_knowledge_base_id)
        except KnowledgeBaseNotFoundError:
            # 处理知识库不存在
            return _not_found(resolved_knowledge_base_id, request_id)

    @app.delete("/knowledge-bases/{knowledge_base_id}")
    def delete_knowledge_base(knowledge_base_id: str, request: Request):
        """
        删除一个知识库
        Args:
            knowledge_base_id: 知识库 ID
            request: 当前 HTTP 请求
        Returns:
            删除结果
        """
        request_id = _request_id(request)
        try:
            # 解析知识库 ID
            resolved_knowledge_base_id = _resolve_knowledge_base_id(knowledge_base_id)
        except ValueError as exc:
            # 处理无效的知识库 ID
            return _build_error_response(
                code="invalid_knowledge_base_id",
                message=str(exc),
                request_id=request_id,
                status_code=422,
            )
        try:
            # 删除知识库
            return service.get_system().delete_knowledge_base(resolved_knowledge_base_id)
        except ProtectedKnowledgeBaseError:
            # 处理默认知识库保护
            return _protected(resolved_knowledge_base_id, request_id)
        except KnowledgeBaseNotFoundError:
             # 处理知识库不存在
            return _not_found(resolved_knowledge_base_id, request_id)
