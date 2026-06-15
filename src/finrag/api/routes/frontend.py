"""前端静态文件路由"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse

from finrag.api.errors import _build_error_response
from finrag.core.config import PROJECT_ROOT

FRONTEND_DIST_DIR = PROJECT_ROOT / "apps" / "web" / "dist" # 前端构建产物目录
FRONTEND_INDEX = FRONTEND_DIST_DIR / "index.html" # 前端单页应用入口文件


def register_frontend_routes(app: FastAPI) -> None:
    """
    注册 Web 工作台入口路由
    Args:
        app: 需要注册路由的 FastAPI 应用实例
    """

    @app.get("/", include_in_schema=False) # 不在文档中显示该路由
    def frontend(request: Request):
        """
        返回已构建的前端入口文件
        Args:
            request: 当前 HTTP 请求
        Returns:
            index.html 文件响应；未构建时返回统一错误响应
        """
        if not FRONTEND_INDEX.exists():
            request_id = getattr(request.state, "request_id", None) or uuid.uuid4().hex
            return _build_error_response(
                code="frontend_not_built",
                message="React 前端尚未构建，请在 apps/web/ 下运行 `npm install` 和 `npm run build`",
                request_id=request_id,
                status_code=503,
            )
        return FileResponse(FRONTEND_INDEX)
