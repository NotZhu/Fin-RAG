"""FastAPI 异常处理器"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from finrag.api.errors import _build_error_response, _validation_error_message
from finrag.api.middleware import REQUEST_ID_HEADER


def install_exception_handlers(app: FastAPI) -> None:
    """
    注册 API 异常处理器
    Args:
        app: 需要安装处理器的 FastAPI 应用实例
    """

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        """
        将请求校验异常转换为统一错误响应
        Args:
            request: 当前 HTTP 请求
            exc: FastAPI 请求校验异常
        Returns:
            包含错误信封的 JSON 响应
        """
        # 优先使用请求上下文中的 request_id，其次使用请求头中的 REQUEST_ID_HEADER，最后生成一个新的 UUID 作为 request_id
        request_id = getattr(request.state, "request_id", None) or request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        # 构建并返回统一的错误响应
        return _build_error_response(
            code="validation_error",
            message=_validation_error_message(exc),
            request_id=request_id,
            status_code=422,
        )
