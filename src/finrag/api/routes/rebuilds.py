"""知识库全量重建任务路由"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, Request, status

from finrag.api.errors import _build_error_response
from finrag.api.rag_service import RAGService
from finrag.core.config import validate_knowledge_base_id
from finrag.storage.knowledge_base_registry import KnowledgeBaseArchivedError


def register_rebuild_routes(app: FastAPI, service: RAGService) -> None:
    """
    注册知识库全量重建任务路由
    Args:
        app: 需要注册路由的 FastAPI 应用实例
        service: 访问 RAG 系统的服务适配器
    """

    def _resolve_knowledge_base_id(value: str) -> str:
        """
        验证并返回知识库 ID
        Args:
            value: 输入的知识库 ID
        Returns:
            验证后的知识库 ID
        """
        return validate_knowledge_base_id(value)

    def _request_id(request: Request) -> str:
        """
        从请求状态中获取或生成请求 ID
        Args:
            request: FastAPI 请求实例
        Returns:
            请求 ID
        """
        return getattr(request.state, "request_id", None) or uuid.uuid4().hex

    @app.post("/knowledge-bases/{knowledge_base_id}/rebuilds", status_code=status.HTTP_202_ACCEPTED)
    def start_knowledge_base_rebuild(knowledge_base_id: str, request: Request):
        """
        创建指定知识库的全量重建任务
        Args:
            knowledge_base_id: 目标知识库 ID
            request: FastAPI 请求实例
        Returns:
            任务 ID、任务状态、开始时间、完成时间、错误信息和结果
        """
        request_id = _request_id(request)
        try:
            resolved_knowledge_base_id = _resolve_knowledge_base_id(knowledge_base_id)
        except ValueError as exc:
            return _build_error_response(
                code="invalid_knowledge_base_id",
                message=str(exc),
                request_id=request_id,
                status_code=422,
            )
        try:
            # 开始重建任务
            return service.start_rebuild(resolved_knowledge_base_id)
        except KnowledgeBaseArchivedError:
            # 处理知识库已归档
            return _build_error_response(
                code="knowledge_base_archived",
                message=f"知识库 {resolved_knowledge_base_id!r} 已归档",
                request_id=request_id,
                status_code=409,
            )

    @app.get("/knowledge-bases/{knowledge_base_id}/rebuilds/{job_id}")
    def get_knowledge_base_rebuild(knowledge_base_id: str, job_id: str, request: Request):
        """
        查询指定知识库的全量重建任务状态
        Args:
            knowledge_base_id: 目标知识库 ID
            job_id: 任务 ID
            request: FastAPI 请求实例
        Returns:
            任务状态、开始时间、完成时间、错误信息和结果
        """
        request_id = _request_id(request)
        try:
            resolved_knowledge_base_id = _resolve_knowledge_base_id(knowledge_base_id)
        except ValueError as exc:
            return _build_error_response(
                code="invalid_knowledge_base_id",
                message=str(exc),
                request_id=request_id,
                status_code=422,
            )
        try:
            return service.get_rebuild_job(resolved_knowledge_base_id, job_id)
        except KeyError:
            return _build_error_response(
                code="rebuild_job_not_found",
                message=f"重建任务 {job_id!r} 不存在",
                request_id=request_id,
                status_code=404,
            )
