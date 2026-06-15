"""健康检查、就绪状态和预热路由"""

from __future__ import annotations

from fastapi import FastAPI

from finrag.api.rag_service import RAGService


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
        返回 RAG 系统就绪状态
        Returns:
            服务层 ready 状态字典
        """
        return service.ready()

    @app.post("/warmup")
    def warmup() -> dict:
        """
        主动触发 RAG 系统初始化
        Returns:
            初始化后的 ready 状态字典
        """
        return service.warmup()
