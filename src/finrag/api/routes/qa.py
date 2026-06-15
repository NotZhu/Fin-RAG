"""问答路由"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from finrag.api.schemas import AskRequest
from finrag.api.rag_service import RAGService


def register_qa_routes(app: FastAPI, service: RAGService) -> None:
    """
    注册流式问答路由
    Args:
        app: 需要注册路由的 FastAPI 应用实例
        service: 访问 RAG 系统的服务适配器
    """

    @app.post("/ask")
    async def ask(payload: AskRequest, request: Request) -> StreamingResponse:
        """
        以 SSE 形式返回问答过程和最终结果
        Args:
            payload: 已校验的问答请求体
            request: 当前 HTTP 请求，用于检测客户端断连
        Returns:
            text/event-stream 流式响应
        """
        # 问答流
        return StreamingResponse(
            # 异步迭代器，不断 yield SSE 事件
            service.ask_stream(payload, is_disconnected=request.is_disconnected),
            media_type="text/event-stream", # SSE 响应
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}, # 禁用缓存和缓冲
        )
