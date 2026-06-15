"""问答管道服务类，负责执行问答流程"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager

from finrag.core.node_schema import TextNode
from finrag.core.response_schema import FinRAGResponse, RAGTrace, RetrievedSource
from finrag.retrieval.llamaindex_trace import FinRAGTraceHandler

_SOURCE_LIST_LINE = re.compile(r"^\s*\[\d+\]\s*(?:来源|Source|source)\s*[:：].*$")


class QAPipelineService:
    """问答管道服务类，负责执行问答流程"""

    def __init__(self, system: Any):
        self.system = system

    def ask_question(
        self,
        question: str,
        return_sources: bool = False,
        return_trace: bool = False,
        knowledge_base_id: Optional[str] = None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_event: Any = None,
    ):
        """
        执行问答流程：路由判断、检索、生成回答
        Args:
            question: 问题
            return_sources: 是否返回来源列表
            return_trace: 是否返回轨迹
            knowledge_base_id: 知识库ID
            event_sink: 事件接收器
            cancel_event: 取消事件
        Returns:
            FinRAGResponse: 问答响应
        """

        def emit(event_type: str, **payload: Any) -> None:
            if event_sink is not None:
                event_sink({"type": event_type, **payload})

        def check_cancelled() -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("ask stream 已取消")

        system = self.system
        total_start = time.perf_counter()
        # trace 事件列表
        events: List[Dict[str, Any]] = []

        # 检索策略
        strategy = getattr(system.config, "retrieval_strategy", "llamaindex_router")
        analysis_start = time.perf_counter()

        emit("analysis", route_type="pending", query=question, retrieval_strategy=strategy)
        check_cancelled()

        # 检查路由引擎是否初始化
        if system.router_engine is None:
            analysis_ms = self.elapsed_ms(analysis_start)
            return self._knowledge_unavailable(
                question=question,
                strategy=strategy,
                route_type="knowledge",
                filters={"knowledge_base_id": knowledge_base_id or system.config.knowledge_base_id},
                error=RuntimeError("router_engine 未初始化"),
                events=events,
                timings_ms={"analysis": round(analysis_ms, 2), "total": round(self.elapsed_ms(total_start), 2)},
                return_trace=return_trace,
                emit=emit,
            )

        # 使用路由引擎
        return self._ask_via_router(
            question=question,
            strategy=strategy,
            analysis_start=analysis_start,
            total_start=total_start,
            return_sources=return_sources, return_trace=return_trace,
            knowledge_base_id=knowledge_base_id,
            events=events,
            emit=emit, check_cancelled=check_cancelled,
        )

    # ------------------------------------------------------------------
    # Router path
    # ------------------------------------------------------------------

    def _ask_via_router(
        self,
        *,
        question: str,
        strategy: str,
        analysis_start: float,
        total_start: float,
        return_sources: bool,
        return_trace: bool,
        knowledge_base_id: Optional[str],
        events: List[Dict[str, Any]],
        emit: Callable[..., None],
        check_cancelled: Callable[[], None],
    ) -> FinRAGResponse:
        """
        执行问答流程：路由判断、检索、生成回答
        Args:
            question: 问题
            strategy: 检索策略
            analysis_start: 分析开始时间
            total_start: 总开始时间
            return_sources: 是否返回来源列表
            return_trace: 是否返回轨迹
            knowledge_base_id: 知识库ID
            events: trace 事件列表
            emit: 事件接收器
            check_cancelled: 取消事件检查函数
        Returns:
            FinRAGResponse: 问答响应
        """
        system = self.system
        analysis_ms = self.elapsed_ms(analysis_start)
        # 初始化轨迹处理程序
        trace_handler = FinRAGTraceHandler()
        # 保存旧的回调管理器
        old_callback_manager = Settings.callback_manager
        # 设置回调管理器为轨迹处理程序
        Settings.callback_manager = CallbackManager([trace_handler])

        try:
            try:
                # 执行路由查询
                response_obj = system.router_engine.query(question)
            except Exception as exc:
                # 处理路由查询异常
                return self._knowledge_unavailable(
                    question=question, strategy=strategy, route_type="knowledge",
                    filters={"knowledge_base_id": knowledge_base_id or system.config.knowledge_base_id},
                    error=exc, events=events,
                    timings_ms={"analysis": round(analysis_ms, 2), "total": round(self.elapsed_ms(total_start), 2)},
                    return_trace=return_trace, emit=emit,
                )

            # 从路由查询结果中提取检索节点
            retrieved = list(getattr(response_obj, "source_nodes", []) or [])
            # 从路由查询结果中提取证据节点
            evidence_nodes: List[TextNode] = [item.node for item in retrieved]
            # 从路由查询结果中提取回答生成器
            response_gen = getattr(response_obj, "response_gen", None)
            # 从路由查询结果中提取回答流
            answer_stream = response_gen if response_gen is not None else [str(response_obj)]
            # 从回答流中提取回答
            answer = self.strip_generated_source_list(self.emit_answer_stream(answer_stream, emit, check_cancelled))

            # 确定路由类型
            route_type = "knowledge" if evidence_nodes else "general"
            # 确定查询引擎
            selected_engine = "knowledge_router" if evidence_nodes else "general_engine"
            emit("route", route_type=route_type, selected_query_engine=selected_engine)
            events.append({"stage": "route", "route_type": route_type, "selected_query_engine": selected_engine})
            check_cancelled()

            # 去重后的证据节点
            deduped_evidence_nodes = self.dedupe_evidence_nodes(evidence_nodes)
            # 构建检索源列表
            sources = self.build_sources(deduped_evidence_nodes, retrieved)
            if return_sources:
                for source in sources:
                    emit("source", source=source.to_dict())

            # 确定最终决策
            final_decision = "generate" if (evidence_nodes or route_type == "general") else "insufficient_evidence"

            # 构建轨迹
            trace = RAGTrace(
                retrieval_strategy=strategy, 
                route_type=route_type,
                timings_ms={"analysis": round(analysis_ms, 2), "total": round(self.elapsed_ms(total_start), 2)},
                retrieved_nodes=[self.node_trace(rank, item.node, item.score) for rank, item in enumerate(retrieved, 1)],
                evidence_nodes=[self.node_trace(rank, node, None) for rank, node in enumerate(deduped_evidence_nodes, 1)],
                events=events, 
                source_count=len(sources),
                reranker={"provider": getattr(system.reranker, "provider", "none") if getattr(system, "reranker", None) is not None else "none"},
                auto_merge={"simple_ratio_thresh": system.config.auto_merge_ratio_threshold},
                final_decision=final_decision,
                llamaindex_events=trace_handler.events,
            )
            # 构建问答响应
            response = FinRAGResponse(
                question=question,
                answer=answer,
                sources=sources if return_sources else [],
                trace=trace if return_trace else None,
                retrieval_strategy=strategy,
                route_type=route_type,
            )
            emit("done", response=response.to_dict(), final_decision=final_decision)
            return response
        finally:
            # 恢复旧的回调管理器
            Settings.callback_manager = old_callback_manager

    def _knowledge_unavailable(
        self,
        *,
        question: str,
        strategy: str,
        route_type: str,
        filters: Dict[str, Any],
        error: Exception,
        events: List[Dict[str, Any]],
        timings_ms: Dict[str, float],
        return_trace: bool,
        emit: Callable[..., None],
    ) -> FinRAGResponse:
        """处理知识库不可用错误"""
        code = str(getattr(error, "code", "") or self._error_code(error))
        message = f"{error.__class__.__name__}: {error}"
        payload = {"code": code, "message": message, "retryable": self._is_retryable(code)}
        emit("error", **payload)
        events.append({"stage": "error", **payload})

        trace = RAGTrace(
            retrieval_strategy=strategy,
            route_type=route_type,
            filters=filters,
            timings_ms=timings_ms,
            events=events,
            final_decision="knowledge_unavailable",
        )
        response = FinRAGResponse(
            question=question,
            answer="知识库当前不可用，请稍后重试或联系运维检查检索服务配置",
            sources=[],
            trace=trace if return_trace else None,
            retrieval_strategy=strategy,
            route_type=route_type,
        )
        emit("done", response=response.to_dict(), final_decision="knowledge_unavailable")
        return response

    @staticmethod
    def _error_code(error: Exception) -> str:
        """根据异常类型生成错误码"""
        text = f"{error.__class__.__name__}: {error}".lower()
        if "milvus" in text:
            return "milvus_unavailable"
        if "embedding" in text or "dashscope_api_key" in text:
            return "embedding_unavailable"
        if "summary" in text:
            return "summary_llm_unavailable"
        return "knowledge_unavailable"

    @staticmethod
    def _is_retryable(code: str) -> bool:
        """判断错误码是否可重试"""
        return code in {"milvus_unavailable", "embedding_unavailable", "knowledge_unavailable"}


    def build_sources(self, evidence_nodes: List[TextNode], retrieved: List[Any]) -> List[RetrievedSource]:
        """
        从证据节点和检索结果中构建检索源
        Args:
            evidence_nodes: 证据节点列表
            retrieved: 检索结果列表
        Returns:
            检索源列表
        """
        # 建立节点ID到分数的映射表
        score_by_node_id: Dict[str, float] = {}
        for result in retrieved:
            metadata = result.node.metadata or {}
            # 将同一个检索分数挂到所有相关的节点ID
            for key in (result.node.node_id, metadata.get("chunk_id"), metadata.get("parent_chunk_id"), metadata.get("root_chunk_id")):
                if key:
                    score_by_node_id[str(key)] = max(self.safe_score(result.score), score_by_node_id.get(str(key), 0.0))
        # 构建检索源列表
        sources: List[RetrievedSource] = []
        seen: set[str] = set()

        for node in evidence_nodes:
            metadata = node.metadata or {}
            node_id = node.node_id
            source_key = str(metadata.get("chunk_id") or node_id)
            if source_key in seen:
                continue
            seen.add(source_key)
            source_id = len(sources) + 1
            sources.append(RetrievedSource(
                source_id=source_id,
                filename=metadata.get("filename", ""),
                file_type=metadata.get("file_type", ""),
                page_number=self.safe_int(metadata.get("page_number"), None),
                chunk_id=metadata.get("chunk_id", node_id),
                parent_chunk_id=metadata.get("parent_chunk_id", ""),
                root_chunk_id=metadata.get("root_chunk_id", ""),
                chunk_level=int(metadata.get("chunk_level", 0) or 0),
                chunk_idx=int(metadata.get("chunk_idx", -1) or -1),
                score=score_by_node_id.get(node_id) or score_by_node_id.get(metadata.get("chunk_id", "")),
                snippet=self.build_snippet(node.text),
            ))
        return sources

    @staticmethod
    def dedupe_evidence_nodes(evidence_nodes: List[TextNode]) -> List[TextNode]:
        """
        从证据节点中去重
        Args:
            evidence_nodes: 证据节点列表
        Returns:
            去重后的证据节点列表
        """
        deduped: List[TextNode] = []
        seen: set[str] = set()
        for node in evidence_nodes:
            metadata = node.metadata or {}
            source_key = str(metadata.get("chunk_id") or node.node_id)
            if source_key in seen:
                continue
            seen.add(source_key)
            deduped.append(node)
        return deduped

    @staticmethod
    def node_trace(rank: int, node: TextNode, score: Optional[float]) -> Dict[str, Any]:
        """
        从文本节点中提取轨迹信息
        Args:
            rank: 节点排名
            node: 文本节点
            score: 节点分数
        Returns:
            轨迹字典
        """
        metadata = node.metadata or {}
        return {
            "rank": rank,
            "node_id": node.node_id,
            "filename": metadata.get("filename", ""),
            "chunk_id": metadata.get("chunk_id", node.node_id),
            "parent_chunk_id": metadata.get("parent_chunk_id", ""),
            "root_chunk_id": metadata.get("root_chunk_id", ""),
            "chunk_level": metadata.get("chunk_level"),
            "chunk_idx": metadata.get("chunk_idx"),
            "score": score,
        }

    @staticmethod
    def build_snippet(content: str, max_length: int = 200) -> str:
        """
        从内容中提取摘要片段
        Args:
            content: 内容字符串
            max_length: 最大片段长度
        Returns:
            片段字符串
        """
        snippet = " ".join((content or "").split())
        return snippet if len(snippet) <= max_length else snippet[:max_length - 3].rstrip() + "..."

    @staticmethod
    def emit_answer_stream(chunks: Iterable[Any], emit: Callable[..., None], check_cancelled: Callable[[], None]) -> str:
        """
        从回答流中提取回答并发送
        Args:
            chunks: 回答流
            emit: 发送函数
            check_cancelled: 检查取消函数
        Returns:
            回答字符串
        """
        parts: List[str] = []
        for chunk in chunks:
            check_cancelled()
            if chunk is None:
                continue
            text = str(chunk)
            if not text:
                continue
            parts.append(text)
            emit("token", text=text)
        return "".join(parts)

    @staticmethod
    def strip_generated_source_list(answer: str) -> str:
        """
        移除模型在答案末尾自行追加的来源列表，保留正文中的引用编号。
        """
        lines = answer.rstrip().splitlines()
        end = len(lines)
        while end > 0 and _SOURCE_LIST_LINE.match(lines[end - 1]):
            end -= 1
        if end == len(lines):
            return answer.rstrip()
        while end > 0 and not lines[end - 1].strip():
            end -= 1
        return "\n".join(lines[:end]).rstrip()

    @staticmethod
    def safe_int(value: Any, default: Optional[int]) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def safe_score(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def elapsed_ms(start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000
