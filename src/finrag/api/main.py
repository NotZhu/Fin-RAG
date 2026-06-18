"""FinRAG 的 FastAPI 应用工厂"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from finrag.api.handlers import install_exception_handlers
from finrag.api.middleware import install_request_context_middleware
from finrag.api.routes.documents import register_document_routes
from finrag.api.routes.frontend import FRONTEND_DIST_DIR, register_frontend_routes
from finrag.api.routes.health import register_health_routes
from finrag.api.routes.knowledge_bases import register_knowledge_base_routes
from finrag.api.routes.qa import register_qa_routes
from finrag.api.rag_service import RAGService
from finrag.application.system import FinRAGSystem


def create_app(
    system_factory: Callable[[], FinRAGSystem] | None = None,
    upload_dir: str | Path | None = None,
) -> FastAPI:
    """
    创建并配置 FinRAG FastAPI 应用
    Args:
        system_factory: 可选的 FinRAGSystem 工厂函数，测试中可替换
        upload_dir: 上传文件的临时落盘目录
    Returns:
        已注册中间件和路由的 FastAPI 应用实例
    """
    service = RAGService(system_factory or FinRAGSystem)
    app = FastAPI(title="FinRAG 金融知识库 API")

    # 挂载前端静态资源目录
    frontend_assets_dir = FRONTEND_DIST_DIR / "assets"
    if frontend_assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=frontend_assets_dir), name="frontend-assets")

    # 安装全局异常处理器
    install_exception_handlers(app)
    # 安装请求上下文中间件
    install_request_context_middleware(app, service)
    # 注册前端入口路由
    register_frontend_routes(app)
    # 注册健康检查路由
    register_health_routes(app, service)
    # 注册知识库管理路由
    register_knowledge_base_routes(app, service)
    # 注册文档路由
    register_document_routes(app, service, upload_dir)
    # 注册问答路由
    register_qa_routes(app, service)
    return app

#  创建全局应用实例
app = create_app()
