"""FinRAG HTTP API 的错误响应辅助函数"""

from __future__ import annotations

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def _build_error_response(*, code: str, message: str, request_id: str, status_code: int) -> JSONResponse:
    """
    构造统一错误响应结构
    Args:
        code: 业务错误码
        message: 面向调用方的错误信息
        request_id: 当前请求 ID
        status_code: HTTP 状态码
    Returns:
        FastAPI JSONResponse 错误响应
    """
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message, "request_id": request_id}})


def _is_upstream_model_error(exc: Exception) -> bool:
    """
    判断异常是否来自上游模型 SDK
    Args:
        exc: 请求处理中捕获的异常
    Returns:
        如果异常来自 openai 兼容 SDK 则返回 True
    """
    module_name = exc.__class__.__module__ # 异常对象所属类的模块名
    return module_name == "openai" or module_name.startswith("openai.")


def _build_upstream_model_error_response(exc: Exception, request_id: str) -> JSONResponse:
    """
    构造上游模型服务失败时的错误响应
    Args:
        exc: 上游模型调用异常
        request_id: 当前请求 ID
    Returns:
        HTTP 502 JSONResponse
    """
    return _build_error_response(
        code="upstream_model_error",
        message=f"上游模型服务请求失败: {exc}",
        request_id=request_id,
        status_code=502,
    )


def _build_document_not_found_response(document_id: str, request_id: str) -> JSONResponse:
    """
    构造文档不存在时的错误响应
    Args:
        document_id: 请求操作的文档 ID
        request_id: 当前请求 ID
    Returns:
        HTTP 404 JSONResponse
    """
    return _build_error_response(
        code="document_not_found",
        message=f"文档不存在: {document_id}",
        request_id=request_id,
        status_code=404,
    )


def _validation_error_message(exc: RequestValidationError) -> str:
    """
    从 FastAPI/Pydantic 校验异常中提取首个可读错误信息
    Args:
        exc: 请求参数校验异常
    Returns:
        可展示给调用方的错误文本
    """
    errors = exc.errors()
    if not errors:
        return "请求参数校验失败"
    return str(errors[0].get("msg") or "请求参数校验失败")
