"""问答管道服务类，负责执行问答流程"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable, List, Optional

from finrag.core.node_schema import TextNode
from finrag.core.response_schema import FinRAGResponse, RAGTrace, RetrievedSource


class QAPipelineService:
    """问答管道服务类，负责执行问答流程"""

    def __init__(self, system: Any):
        self.system = system

    def ask_question(
        self,
        question: str,
        knowledge_base_id: str,
        return_sources: bool = False,
        return_trace: bool = False,
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
        filters = self.knowledge_base_filters(system, knowledge_base_id)
        router_start = time.perf_counter()

        emit("analysis", route_type="pending", query=question, retrieval_strategy=strategy)
        check_cancelled()

        # 检查路由引擎是否初始化
        if system.router_engine is None:
            pipeline_steps = [
                self.trace_step(
                    "query_router",
                    1,
                    "error",
                    self.elapsed_ms(router_start),
                    {"route_type": "knowledge", "selected_query_engine": "knowledge_router"},
                    "router_engine 未初始化",
                )
            ]
            return self._knowledge_unavailable(
                question=question,
                strategy=strategy,
                route_type="knowledge",
                filters=filters,
                error=RuntimeError("router_engine 未初始化"),
                events=events,
                timings_ms={"analysis": round(self.elapsed_ms(router_start), 2), "total": round(self.elapsed_ms(total_start), 2)},
                pipeline_steps=pipeline_steps,
                return_trace=return_trace,
                emit=emit,
            )

        # 使用路由引擎
        return self._ask_via_router(
            question=question,
            strategy=strategy,
            total_start=total_start,
            return_sources=return_sources, return_trace=return_trace,
            filters=filters,
            events=events,
            emit=emit, check_cancelled=check_cancelled,
        )

    def _ask_via_router(
        self,
        *,
        question: str,
        strategy: str,
        total_start: float,
        return_sources: bool,
        return_trace: bool,
        filters: Dict[str, Any],
        events: List[Dict[str, Any]],
        emit: Callable[..., None],
        check_cancelled: Callable[[], None],
    ) -> FinRAGResponse:
        """
        执行问答流程：路由判断、检索、生成回答
        Args:
            question: 问题
            strategy: 检索策略
            total_start: 总开始时间
            return_sources: 是否返回来源列表
            return_trace: 是否返回轨迹
            events: trace 事件列表
            emit: 事件接收器
            check_cancelled: 取消事件检查函数
        Returns:
            FinRAGResponse: 问答响应
        """
        system = self.system
        query_start = time.perf_counter()
        try:
            # 交给 LlamaIndex router 完成路由、检索和生成
            response_obj = system.router_engine.query(question)
        except Exception as exc:
            query_ms = self.elapsed_ms(query_start)
            pipeline_steps = [
                self.trace_step(
                    "query_router",
                    1,
                    "error",
                    query_ms,
                    {"error": f"{exc.__class__.__name__}: {exc}", "selected_query_engine": "knowledge_router"},
                    "查询路由失败",
                )
            ]
            # 处理路由查询异常
            return self._knowledge_unavailable(
                question=question, strategy=strategy, route_type="knowledge",
                filters=filters,
                error=exc, events=events,
                timings_ms={"analysis": round(query_ms, 2), "total": round(self.elapsed_ms(total_start), 2)},
                pipeline_steps=pipeline_steps,
                return_trace=return_trace, emit=emit,
            )
        query_ms = self.elapsed_ms(query_start)

        # 从路由查询结果中提取检索节点
        retrieved = list(getattr(response_obj, "source_nodes", []) or [])
        # 从路由查询结果中提取证据节点
        evidence_nodes: List[TextNode] = [item.node for item in retrieved]

        # 从路由查询结果中提取选中的路由引擎
        selected_engine = self.selected_route_engine(response_obj)
        if not selected_engine:
            selected_engine = "knowledge_router" if evidence_nodes else "general_router"
        route_type = self.route_type_from_engine(selected_engine)
        emit("route", route_type=route_type, selected_query_engine=selected_engine)
        events.append({"stage": "route", "route_type": route_type, "selected_query_engine": selected_engine})
        check_cancelled()

        # 去重后的证据节点
        evidence_start = time.perf_counter()
        deduped_evidence_nodes = self.dedupe_evidence_nodes(evidence_nodes)
        # 构建检索源列表
        sources = self.build_sources(deduped_evidence_nodes, retrieved)
        evidence_ms = self.elapsed_ms(evidence_start)
        if return_sources:
            for source in sources:
                emit("source", source=source.to_dict())

        # 从路由查询结果中提取回答生成器
        response_gen = getattr(response_obj, "response_gen", None)
        # 从路由查询结果中提取回答流
        answer_stream = response_gen if response_gen is not None else [str(response_obj)]
        # 从回答流中提取回答
        generation_start = time.perf_counter()
        answer = self.emit_answer_stream(answer_stream, emit, check_cancelled)
        generation_ms = self.elapsed_ms(generation_start)
        hybrid_trace = dict(getattr(getattr(system, "hybrid_retriever", None), "last_hybrid_trace", {}) or {})
        pipeline_steps = self.build_pipeline_steps(
            system=system,
            route_type=route_type,
            selected_engine=selected_engine,
            selected_knowledge_engine=self.selected_inner_engine(response_obj) or "auto_merge",
            has_evidence=bool(evidence_nodes),
            query_ms=query_ms,
            retrieve_ms=self.safe_optional_float(hybrid_trace.get("elapsed_ms")),
            ranking_ms=self.ranking_postprocess_duration_ms(system),
            evidence_ms=evidence_ms,
            generation_ms=generation_ms,
            answer_chars=len(answer),
            source_count=len(sources),
            evidence_count=len(deduped_evidence_nodes),
            hybrid_trace=hybrid_trace,
        )

        # 确定最终决策
        final_decision = "generate" if (evidence_nodes or route_type == "general") else "insufficient_evidence"

        # 构建轨迹
        trace = RAGTrace(
            retrieval_strategy=strategy,
            route_type=route_type,
            filters=filters,
            timings_ms={"analysis": round(query_ms, 2), "total": round(self.elapsed_ms(total_start), 2)},
            pipeline_steps=pipeline_steps,
            retrieved_nodes=[self.node_trace(rank, item.node, item.score) for rank, item in enumerate(retrieved, 1)],
            evidence_nodes=[self.node_trace(rank, node, None) for rank, node in enumerate(deduped_evidence_nodes, 1)],
            events=events,
            source_count=len(sources),
            reranker={"provider": getattr(system.reranker, "provider", "none") if getattr(system, "reranker", None) is not None else "none"},
            auto_merge={"simple_ratio_thresh": system.config.auto_merge_ratio_threshold},
            final_decision=final_decision,
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

    @staticmethod
    def trace_step(
        step_id: str,
        order: int,
        status: str,
        duration_ms: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
        detail: str = "",
    ) -> Dict[str, Any]:
        """
        构建最终 trace 中的链路步骤
        Args:
            step_id: 步骤 ID
            order: 展示顺序
            status: 步骤状态
            duration_ms: 步骤耗时
            meta: 步骤元数据
            detail: 错误等必要详情
        Returns:
            trace step 字典
        """
        return {
            "id": step_id,
            "order": order,
            "label": "",
            "detail": detail,
            "status": status,
            "duration_ms": round(float(duration_ms), 2) if duration_ms is not None else None,
            "meta": dict(meta or {}),
        }

    def build_pipeline_steps(
        self,
        *,
        system: Any,
        route_type: str,
        selected_engine: str,
        selected_knowledge_engine: str,
        has_evidence: bool,
        query_ms: float,
        retrieve_ms: Optional[float],
        ranking_ms: Optional[float],
        evidence_ms: float,
        generation_ms: float,
        answer_chars: int,
        source_count: int,
        evidence_count: int,
        hybrid_trace: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        根据最终查询结果构建完整链路快照
        Args:
            system: 系统对象
            route_type: 路由类型
            selected_engine: 顶层路由选择
            selected_knowledge_engine: 知识库内查询引擎
            has_evidence: 是否有知识库证据
            query_ms: 路由查询总耗时
            retrieve_ms: LlamaIndex 检索耗时
            ranking_ms: 精排和后处理耗时
            evidence_ms: 证据整理耗时
            generation_ms: 回答输出耗时
            answer_chars: 回答字符数
            source_count: 来源数量
            evidence_count: 证据片段数量
            hybrid_trace: 混合召回元数据
        Returns:
            完整 pipeline step 列表
        """
        knowledge_status = "complete" if route_type == "knowledge" else "skipped"
        evidence_status = "complete" if has_evidence else "skipped"
        return [
            self.trace_step(
                "query_router",
                1,
                "complete",
                query_ms,
                {"route_type": route_type, "selected_query_engine": selected_engine},
            ),
            self.trace_step(
                "knowledge_engine",
                2,
                knowledge_status,
                None,
                {"route_type": route_type, "selected_knowledge_engine": selected_knowledge_engine},
            ),
            self.trace_step(
                "hybrid_search",
                3,
                evidence_status,
                retrieve_ms if has_evidence else None,
                self.hybrid_search_meta(system, hybrid_trace if has_evidence else {}),
            ),
            self.trace_step(
                "ranking_postprocess",
                4,
                evidence_status,
                ranking_ms if has_evidence else None,
                self.ranking_postprocess_meta(system),
            ),
            self.trace_step(
                "context_expansion",
                5,
                evidence_status,
                None,
                self.context_expansion_meta(system),
            ),
            self.trace_step(
                "evidence_window",
                6,
                evidence_status,
                evidence_ms if has_evidence else None,
                {"source_count": source_count, "evidence_count": evidence_count},
            ),
            self.trace_step(
                "streaming_answer",
                7,
                "complete",
                generation_ms,
                {"answer_chars": answer_chars},
            ),
        ]

    @staticmethod
    def selected_inner_engine(response_obj: Any) -> str:
        """
        提取选中的知识引擎
        Args:
            response_obj: 问答响应对象
        Returns:
            选中的知识引擎字符串
        """
        metadata = getattr(response_obj, "metadata", {}) or {}
        for key in ("selected_knowledge_engine", "selected_engine", "selected_tool", "tool_name"):
            value = metadata.get(key)
            if value:
                return str(value)
        return ""

    @staticmethod
    def selected_route_engine(response_obj: Any) -> str:
        """
        提取顶层路由选中的查询引擎
        """
        metadata = getattr(response_obj, "metadata", {}) or {}
        for key in ("selected_query_engine", "selected_route_engine", "route_engine"):
            value = metadata.get(key)
            if value:
                return str(value)
        route_type = str(metadata.get("route_type") or "")
        if route_type == "knowledge":
            return "knowledge_router"
        if route_type == "general":
            return "general_router"
        return ""

    @staticmethod
    def route_type_from_engine(selected_engine: str) -> str:
        """
        从选中的查询引擎中提取路由类型
        Args:
            selected_engine: 选中的查询引擎字符串
        Returns:
            路由类型字符串
        """
        if str(selected_engine) in {"general", "general_router"}:
            return "general"
        return "knowledge"

    @staticmethod
    def hybrid_search_meta(system: Any, hybrid_trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        提取混合召回元数据
        Args:
            system: 系统对象
            hybrid_trace: 混合搜索轨迹字典
        Returns:
            混合召回元数据字典
        """
        config = getattr(system, "config", None)
        return {
            "hybrid_provider": hybrid_trace.get("hybrid_provider", "milvus"),
            "hybrid_mode": hybrid_trace.get("hybrid_mode", "native_dense_sparse"),
            "hybrid_ranker": hybrid_trace.get("hybrid_ranker", "RRFRanker"),
            "candidate_k": hybrid_trace.get("candidate_k", getattr(config, "retrieval_candidate_k", None)),
            "top_k": getattr(config, "top_k", None),
            "rrf_k": hybrid_trace.get("rrf_k", getattr(config, "rrf_k", None)),
        }

    @staticmethod
    def ranking_postprocess_meta(system: Any) -> Dict[str, Any]:
        """
        提取排名后处理元数据
        Args:
            system: 系统对象
        Returns:
            排名后处理元数据字典
        """
        config = getattr(system, "config", None)
        reranker = getattr(system, "reranker", None)
        reranker_provider = getattr(reranker, "provider", "none") if reranker is not None else "none"
        return {
            "score_threshold": getattr(config, "score_threshold", None),
            "reranker_provider": reranker_provider,
            "reranker_top_n": getattr(config, "reranker_top_n", None),
            "context_token_budget": getattr(config, "context_token_budget", None),
            "prev_next": getattr(config, "neighbor_window", 1),
        }

    @staticmethod
    def ranking_postprocess_duration_ms(system: Any) -> Optional[float]:
        """
        获取精排与后处理耗时；Jina 兼容 reranker 使用自身记录的真实 HTTP 调用耗时
        Args:
            system: 系统对象
        Returns:
            耗时毫秒数；无法获取时返回 None
        """
        reranker = getattr(system, "reranker", None)
        reranker_provider = getattr(reranker, "provider", "") if reranker is not None else ""
        if reranker is not None and reranker_provider and reranker_provider != "none":
            elapsed_ms = getattr(reranker, "last_elapsed_ms", None)
            if elapsed_ms is not None:
                return QAPipelineService.safe_score(elapsed_ms)
        return None

    @staticmethod
    def context_expansion_meta(system: Any) -> Dict[str, Any]:
        """
        提取上下文扩展元数据
        Args:
            system: 系统对象
        Returns:
            上下文扩展元数据字典
        """
        config = getattr(system, "config", None)
        return {"simple_ratio_thresh": getattr(config, "auto_merge_ratio_threshold", None)}

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
        pipeline_steps: List[Dict[str, Any]],
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
            pipeline_steps=pipeline_steps,
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
    def knowledge_base_filters(system: Any, knowledge_base_id: str) -> Dict[str, Any]:
        """
        构建当前知识库 trace 过滤信息
        Args:
            system: FinRAGSystem 或测试替身
            knowledge_base_id: 知识库 ID
        Returns:
            包含知识库和 Milvus collection 的过滤信息
        """
        filters = {"knowledge_base_id": knowledge_base_id}
        scope_builder = getattr(system, "knowledge_base_scope", None)
        if callable(scope_builder):
            scope = scope_builder(knowledge_base_id)
            filters["collection"] = str(getattr(scope, "collection_name", "") or "")
        return filters

    @staticmethod
    def _error_code(error: Exception) -> str:
        """根据异常类型生成错误码"""
        text = f"{error.__class__.__name__}: {error}".lower()
        if "milvus" in text:
            return "milvus_unavailable"
        if "embedding" in text or "dashscope_api_key" in text:
            return "embedding_unavailable"
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
    def safe_optional_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def elapsed_ms(start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000
