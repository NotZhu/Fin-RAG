"""问答路由"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from finrag.api.errors import _build_error_response
from finrag.api.schemas import AskRequest
from finrag.api.rag_service import RAGService
from finrag.core.config import validate_knowledge_base_id
from finrag.storage.knowledge_base_registry import KnowledgeBaseArchivedError


def register_qa_routes(app: FastAPI, service: RAGService) -> None:
    """
    注册流式问答路由
    Args:
        app: 需要注册路由的 FastAPI 应用实例
        service: 访问 RAG 系统的服务适配器
    """

    @app.post("/knowledge-bases/{knowledge_base_id}/ask")
    async def ask_knowledge_base_question(
        knowledge_base_id: str,
        payload: AskRequest,
        request: Request,
    ):
        """
        以 SSE 形式返回问答过程和最终结果
        Args:
            knowledge_base_id: URL 路径中的知识库 ID
            payload: 已校验的问答请求体
            request: 当前 HTTP 请求，用于检测客户端断连
        Returns:
            text/event-stream 流式响应
        """
        request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
        try:
            # 验证并解析知识库 ID
            resolved_knowledge_base_id = validate_knowledge_base_id(knowledge_base_id)
        except ValueError as exc:
            # 处理无效的知识库 ID
            return _build_error_response(
                code="invalid_knowledge_base_id",
                message=str(exc),
                request_id=request_id,
                status_code=422,
            )
        try:
            # 确保知识库已激活
            service.ensure_knowledge_base_active(resolved_knowledge_base_id)
        except KnowledgeBaseArchivedError:
            # 处理知识库已归档
            return _build_error_response(
                code="knowledge_base_archived",
                message=f"知识库 {resolved_knowledge_base_id!r} 已归档",
                request_id=request_id,
                status_code=409,
            )

        # 问答流
        return StreamingResponse(
            # 异步迭代器，不断 yield SSE 事件
            service.ask_stream(
                payload,
                resolved_knowledge_base_id,
                is_disconnected=request.is_disconnected,
            ),
            media_type="text/event-stream", # SSE 响应
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}, # 禁用缓存和缓冲
        )
