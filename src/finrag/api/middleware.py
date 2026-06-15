"""FastAPI 请求上下文与统一错误信封中间件"""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request

from finrag.api.errors import _build_error_response, _build_upstream_model_error_response, _is_upstream_model_error
from finrag.api.rag_service import RAGService

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID" # 请求链路追踪 ID 响应头
PROCESS_TIME_HEADER = "X-Process-Time-MS" # 请求处理耗时响应头


def install_request_context_middleware(app: FastAPI, service: RAGService) -> None:
    """
    注册请求 ID、耗时响应头和统一错误信封中间件
    """

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        """
        为单次请求补充上下文、捕获异常并写入响应头
        Args:
            request: 当前 HTTP 请求
            call_next: FastAPI 提供的后续处理链
        Returns:
            带请求 ID 和耗时响应头的 HTTP 响应
        """
        # 优先使用请求头中的 REQUEST_ID_HEADER，其次生成一个新的 UUID 作为 request_id
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        # 将 request_id 存储到请求上下文 state 中
        request.state.request_id = request_id

        start_time = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            ready_state = service.ready()
            # 如果 RAG 服务未准备好且状态为 error，返回 503 错误响应
            if ready_state.get("status") == "error":
                response = _build_error_response(
                    code="rag_initialization_failed",
                    message="RAG 服务初始化失败",
                    request_id=request_id,
                    status_code=503,
                )
            # 如果异常来自上游模型 SDK，返回 502 错误响应
            elif _is_upstream_model_error(exc):
                response = _build_upstream_model_error_response(exc, request_id)
            # 其他异常，返回 500 错误响应
            else:
                response = _build_error_response(
                    code="internal_error",
                    message="服务内部错误",
                    request_id=request_id,
                    status_code=500,
                )
            status_code = response.status_code
            logger.exception(
                "请求失败 method=%s path=%s request_id=%s",
                request.method,
                request.url.path,
                request_id,
            )
        duration_ms = (time.perf_counter() - start_time) * 1000

        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[PROCESS_TIME_HEADER] = f"{duration_ms:.2f}"
        logger.info(
            "请求完成 method=%s path=%s status_code=%s duration_ms=%.2f request_id=%s",
            request.method,
            request.url.path,
            status_code,
            duration_ms,
            request_id,
        )
        return response
