"""FinRAG HTTP API 使用的服务适配层"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from finrag.api.schemas import AskRequest
from finrag.core.response_schema import FinRAGResponse
from finrag.application.system import FinRAGSystem


class RAGService:
    """封装 FastAPI 路由访问 FinRAGSystem 的服务入口"""

    def __init__(self, system_factory: Callable[[], FinRAGSystem]):
        """
        初始化 RAG 服务适配器
        Args:
            system_factory: 创建 FinRAGSystem 实例的工厂函数
        """
        # 初始化系统工厂
        self._system_factory = system_factory
        # FinRAGSystem 实例，初始为 None，使用时懒加载
        self._system: Optional[FinRAGSystem] = None
        # 初始化锁，确保并发请求只初始化一次
        self._init_lock = Lock()
        # 初始化错误信息
        self._last_error: Optional[str] = None
        # 重建任务状态，键为知识库 ID，值为任务状态字典
        self._rebuild_jobs: dict[str, dict[str, Any]] = {}
        # 重建任务状态锁，确保并发访问安全
        self._rebuild_jobs_lock = Lock()
        # 重建任务执行器，确保串行执行
        self._rebuild_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="finrag-rebuild")

    def get_system(self) -> FinRAGSystem:
        """
        懒加载 FinRAGSystem，保证并发请求只创建一次实例
        Returns:
            FinRAGSystem 实例；知识库构建由 warmup 或问答请求按需触发
        """
        if self._system is None:
            with self._init_lock:
                if self._system is None:
                    try:
                        system = self._system_factory()
                    except Exception as exc:
                        self._last_error = f"{exc.__class__.__name__}: {exc}"
                        raise
                    self._system = system
                    self._last_error = None
        return self._system

    def ensure_knowledge_base_ready(self, knowledge_base_id: str) -> FinRAGSystem:
        """
        按需初始化检索链路并构建知识库
        Args:
            knowledge_base_id: 目标知识库 ID
        Returns:
            已完成检索链路准备的 FinRAGSystem 实例
        """
        system = self.get_system()
        with self._init_lock:
            try:
                # 确保系统已初始化
                system.ensure_knowledge_base_ready(knowledge_base_id)
            except Exception as exc:
                self._last_error = f"{exc.__class__.__name__}: {exc}"
                raise
            self._last_error = None
            return system

    def ready(self, knowledge_base_id: str | None = None) -> dict:
        """
        返回 RAG 服务就绪状态，未初始化或初始化失败时给出错误信息
        Args:
            knowledge_base_id: 可选的目标知识库 ID，提供时返回该知识库的文档统计
        Returns:
            包含 ready、status、文档统计和 last_error 的状态字典
        """
        # 如果之前初始化失败，优先返回错误状态和信息
        if self._last_error:
            return {
                "ready": False,
                "status": "error",
                "total_documents": 0,
                "total_chunks": 0,
                "last_error": self._last_error,
            }
        # 全局 ready 不主动创建系统；scoped ready 需要读取对应知识库文档列表。
        if self._system is None and knowledge_base_id is None:
            return {
                "ready": False,
                "status": "not_ready",
                "total_documents": 0,
                "total_chunks": 0,
                "last_error": None,
            }
        system = self.get_system()
        if knowledge_base_id is None:
            return system.ready()
        return system.ready(knowledge_base_id)

    def warmup(self, knowledge_base_id: str) -> dict:
        """
        主动触发系统初始化，用于服务启动后的预热
        Args:
            knowledge_base_id: 目标知识库 ID
        Returns:
            初始化后的 ready 状态
        """
        # 确保系统已初始化
        self.ensure_knowledge_base_ready(knowledge_base_id)
        return self.ready(knowledge_base_id)

    def start_rebuild(self, knowledge_base_id: str) -> dict[str, Any]:
        """
        创建或复用指定知识库的进程内全量重建任务
        Args:
            knowledge_base_id: 目标知识库 ID
        Returns:
            可用于轮询的任务状态字典
        """
        with self._rebuild_jobs_lock:
            for job in self._rebuild_jobs.values():
                # 检查是否有正在运行或已排队的任务
                if job["knowledge_base_id"] == knowledge_base_id and job["status"] in {"queued", "running"}:
                    return dict(job)
            # 如果没有任务，创建新任务
            job_id = uuid.uuid4().hex
            job = {
                "job_id": job_id,
                "knowledge_base_id": knowledge_base_id,
                "status": "queued",
                "created_at": self._utc_now(),
                "started_at": None,
                "completed_at": None,
                "error": None,
                "result": None,
            }
            self._rebuild_jobs[job_id] = job
        # 提交任务到线程池
        self._rebuild_executor.submit(self._run_rebuild_job, job_id, knowledge_base_id)
        return self.get_rebuild_job(knowledge_base_id, job_id)

    def get_rebuild_job(self, knowledge_base_id: str, job_id: str) -> dict[str, Any]:
        """
        查询指定知识库的进程内全量重建任务
        Args:
            knowledge_base_id: 目标知识库 ID
            job_id: 任务 ID
        Returns:
            任务状态字典
        Raises:
            KeyError: 任务不存在或不属于该知识库
        """
        with self._rebuild_jobs_lock:
            job = self._rebuild_jobs.get(job_id)
            if job is None or job["knowledge_base_id"] != knowledge_base_id:
                raise KeyError(job_id)
            return dict(job)

    def _run_rebuild_job(self, job_id: str, knowledge_base_id: str) -> None:
        """
        在线程池中执行全量重建，并更新进程内任务状态
        Args:
            job_id: 任务 ID
            knowledge_base_id: 目标知识库 ID
        """
        self._update_rebuild_job(job_id, status="running", started_at=self._utc_now())
        try:
            system = self.get_system()
            result = system.rebuild_from_sources(knowledge_base_id)
        except Exception as exc:
            error = f"{exc.__class__.__name__}: {exc}"
            self._last_error = error
            self._update_rebuild_job(
                job_id,
                status="failed",
                completed_at=self._utc_now(),
                error=error,
            )
            return
        self._last_error = None
        self._update_rebuild_job(
            job_id,
            status="succeeded",
            completed_at=self._utc_now(),
            result=result,
        )

    def _update_rebuild_job(self, job_id: str, **changes: Any) -> None:
        """
        原子更新重建任务状态；任务不存在时静默忽略
        Args:
            job_id: 任务 ID
            changes: 要更新的字段值对
        """
        with self._rebuild_jobs_lock:
            job = self._rebuild_jobs.get(job_id)
            if job is None:
                return
            job.update(changes)

    @staticmethod
    def _utc_now() -> str:
        """返回 ISO-8601 UTC 时间戳"""
        return datetime.now(timezone.utc).isoformat()

    async def ask_stream(
        self,
        request: AskRequest,
        knowledge_base_id: str,
        is_disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
    ) -> AsyncIterator[str]:
        """
        以 Server-Sent Events 形式流式执行问答
        Args:
            request: 已通过 Pydantic 校验的问答请求
            knowledge_base_id: URL 路径中的知识库 ID
            is_disconnected: 可选的客户端断连检测函数
        Yields:
            SSE 文本块
        """
        # 获取当前 FastAPI 请求所在的 asyncio 事件循环
        loop = asyncio.get_running_loop()
        # 初始化异步 SSE 队列，用于存储事件
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        # 记录已经发送过哪些事件类型
        emitted_event_types: set[str] = set()
        # 创建一个线程安全的取消信号
        cancel_event = threading.Event()

        def emit(event: dict[str, Any]) -> None:
            """
            将系统事件写入 SSE 队列
            Args:
                event: 问答流程产生的事件载荷
            """
            # 从事件中提取事件类型
            event_type = str(event.get("type") or event.get("event") or "message")
            # 如果是 source 事件且请求中没有请求 sources，跳过
            if event_type == "source" and not request.return_sources:
                emitted_event_types.add(event_type)
                return
            payload = dict(event)
            # 从有效负载中移除事件类型和事件字段
            payload.pop("type", None)
            payload.pop("event", None)
            # 记录事件类型
            emitted_event_types.add(event_type)
            # 往队列中写入事件，格式化为 SSE 文本块
            loop.call_soon_threadsafe(queue.put_nowait, self._format_sse(event_type, payload))

        async def run_query() -> None:
            """
            在线程中执行问答，并把结果事件转发给异步响应流
            """
            try:
                # 执行问答流程
                response = await asyncio.to_thread(
                    self._ask_with_event_sink,
                    request,
                    knowledge_base_id,
                    emit,
                    cancel_event,
                ) # 在线程池中运行的函数，传递的参数
                # 生成完成事件
                for event in self._completion_events(response, emitted_event_types):
                    emit(event)
            except Exception as exc:
                emit({"type": "error", "message": f"{exc.__class__.__name__}: {exc}"})
            finally:
                # 往队列中写入 None，表示问答流程结束
                loop.call_soon_threadsafe(queue.put_nowait, None)

        # 创建一个异步任务，用于执行问答流程
        task = asyncio.create_task(run_query())
        try:
            while True:
                # 检查客户端是否断开连接
                if is_disconnected is not None and await is_disconnected():
                    cancel_event.set()
                    break
                try:
                    # 从队列中获取事件，超时 0.1 秒
                    item = await asyncio.wait_for(queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                if item is None:
                    break
                # 发送事件给前端
                yield item
        finally:
            # 取消问答任务
            cancel_event.set()
            # 如果问答任务没有完成，等待 1 秒
            if not task.done():
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(task, timeout=1.0)
            # 如果 1 秒内没有完成，则强制取消
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    def _ask_with_event_sink(
        self,
        request: AskRequest,
        knowledge_base_id: str,
        event_sink: Callable[[dict[str, Any]], None],
        cancel_event: threading.Event,
    ) -> FinRAGResponse:
        """
        执行带事件回调的问答请求
        Args:
            request: 已通过 Pydantic 校验的问答请求
            knowledge_base_id: URL 路径中的知识库 ID
            event_sink: 接收问答流程事件的回调函数
            cancel_event: 客户端断连时触发的取消信号
        Returns:
            完整的 FinRAGResponse 问答结果
        """
        system = self.get_system()
        try:
            system.ensure_knowledge_base_ready(knowledge_base_id)
        except Exception as exc:
            self._last_error = f"{exc.__class__.__name__}: {exc}"
            raise
        self._last_error = None
        return system.ask_question(
            request.question, # 问答请求
            return_sources=request.return_sources, # 是否返回来源文档
            return_trace=request.return_trace, # 是否返回详细跟踪信息
            knowledge_base_id=knowledge_base_id, # 关联的知识库ID
            event_sink=event_sink, # 事件回调函数
            cancel_event=cancel_event, # 客户端断连时触发的取消信号
        )

    @staticmethod
    def _format_sse(event_type: str, payload: dict[str, Any]) -> str:
        """
        将事件类型和载荷格式化为 SSE 文本块
        Args:
            event_type: SSE 事件类型
            payload: 需要序列化到 data 字段的事件载荷
        Returns:
            可直接写入响应流的 SSE 文本
        """
        data = json.dumps(payload, ensure_ascii=False, default=str)
        return f"event: {event_type}\ndata: {data}\n\n"

    @staticmethod
    def _completion_events(response: FinRAGResponse, emitted_event_types: set[str]) -> list[dict[str, Any]]:
        """
        为流式问答补齐最终完成事件
        Args:
            response: 已完成的问答响应
            emitted_event_types: 已经发送过的事件类型集合
        Returns:
            需要补发的最终事件列表
        """
        if "done" in emitted_event_types:
            return []
        # 从响应中提取跟踪信息
        payload = response.to_dict()
        trace = payload.get("trace") or {}
        event = {"type": "done", "response": payload}
        if trace.get("final_decision"):
            event["final_decision"] = trace["final_decision"]
        return [event]
