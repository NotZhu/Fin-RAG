"""知识库管理路由"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, Request, status

from finrag.api.errors import _build_error_response
from finrag.api.rag_service import RAGService
from finrag.api.schemas import CreateKnowledgeBaseRequest
from finrag.storage.knowledge_base_registry import DuplicateKnowledgeBaseError


def register_knowledge_base_routes(app: FastAPI, service: RAGService) -> None:
    """
    注册知识库创建和列表路由
    Args:
        app: FastAPI 应用
        service: RAG 服务适配器
    """

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
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
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
