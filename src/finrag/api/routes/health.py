"""健康检查、就绪状态和预热路由"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, Request

from finrag.api.errors import _build_error_response
from finrag.api.rag_service import RAGService
from finrag.core.config import validate_knowledge_base_id


def register_health_routes(app: FastAPI, service: RAGService) -> None:
    """
    注册运行时状态路由
    Args:
        app: 需要注册路由的 FastAPI 应用实例
        service: 访问 RAG 系统的服务适配器
    """

    @app.get("/health")
    def health() -> dict:
        """
        返回服务基础存活状态
        Returns:
            包含 status=ok 的响应字典
        """
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        """
        返回 RAG 系统全局就绪状态
        Returns:
            服务层 ready 状态字典
        """
        return service.ready()

    @app.get("/knowledge-bases/{knowledge_base_id}/ready")
    def ready_knowledge_base(knowledge_base_id: str, request: Request):
        """
        返回指定知识库的 RAG 运行时就绪状态
        Args:
            knowledge_base_id: URL 路径中的知识库 ID
            request: 当前 HTTP 请求
        Returns:
            服务层 ready 状态字典
        """
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
        try:
            resolved_knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        except ValueError as exc:
            return _build_error_response(
                code="invalid_knowledge_base_id",
                message=str(exc),
                request_id=request_id,
                status_code=422,
            )
        return service.ready(resolved_knowledge_base_id)

    @app.post("/knowledge-bases/{knowledge_base_id}/warmup")
    def warmup_knowledge_base(knowledge_base_id: str, request: Request):
        """
        主动触发指定知识库的 RAG 运行时初始化
        Args:
            knowledge_base_id: URL 路径中的知识库 ID
            request: 当前 HTTP 请求
        Returns:
            初始化后的 ready 状态字典
        """
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
        try:
            resolved_knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        except ValueError as exc:
            return _build_error_response(
                code="invalid_knowledge_base_id",
                message=str(exc),
                request_id=request_id,
                status_code=422,
            )
        return service.warmup(resolved_knowledge_base_id)
