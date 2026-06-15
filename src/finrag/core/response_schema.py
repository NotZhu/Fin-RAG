"""FinRAG API 响应结构"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RetrievedSource:
    """与答案引用编号对齐的证据来源"""

    source_id: int # 前端展示用的来源编号
    filename: str # 来源文档文件名
    file_type: str # 来源文档类型
    page_number: Optional[int] # PDF 页码，非 PDF 文档为空
    chunk_id: str # 命中的叶子节点 ID
    parent_chunk_id: str # 叶子节点所属父级节点 ID
    root_chunk_id: str # 叶子节点所属根级节点 ID
    chunk_level: int # 当前节点层级
    chunk_idx: int # 当前节点在文档内的序号
    score: Optional[float] # 检索或重排后的相关性分数
    snippet: str # 对外展示的证据片段

    def to_dict(self) -> Dict[str, Any]:
        """
        将来源对象转换为 API 可序列化字典
        Returns:
            包含用户可读来源、页码、片段和相关度的字典
        """
        return {
            "source_id": self.source_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "score": self.score,
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class RAGTrace:
    """FinRAG 查询调试追踪"""

    retrieval_strategy: str # 当前使用的检索策略
    route_type: str = "" # 问题路由类型
    filters: Dict[str, Any] = field(default_factory=dict) # 检索过滤条件
    timings_ms: Dict[str, float] = field(default_factory=dict) # 各阶段耗时毫秒数
    retrieval_params: Dict[str, Any] = field(default_factory=dict) # 检索参数快照
    retrieved_nodes: List[Dict[str, Any]] = field(default_factory=list) # 初始召回节点摘要
    evidence_nodes: List[Dict[str, Any]] = field(default_factory=list) # 生成阶段证据节点摘要
    events: List[Dict[str, Any]] = field(default_factory=list) # 查询过程事件列表
    llamaindex_events: List[Dict[str, Any]] = field(default_factory=list) # LlamaIndex callback 事件摘要
    fusion: Dict[str, Any] = field(default_factory=dict) # 混合召回融合信息
    hybrid_provider: str = "" # 混合检索提供者
    hybrid_mode: str = "" # 混合检索模式
    hybrid_ranker: str = "" # 混合检索排序器
    rrf_k: int = 0 # RRF 参数
    candidate_k: int = 0 # 召回候选数量
    source_count: int = 0 # 最终来源数量
    reranker: Dict[str, Any] = field(default_factory=dict) # reranker 执行信息
    auto_merge: Dict[str, Any] = field(default_factory=dict) # 自动合并证据信息
    final_decision: str = "generate" # 最终处理决策

    def to_dict(self) -> Dict[str, Any]:
        """
        将查询 trace 转换为 API 可序列化字典
        Returns:
            包含检索、融合、精排、置信度和耗时信息的字典
        """
        return asdict(self)


@dataclass(frozen=True)
class FinRAGResponse:
    """API 返回的结构化答案"""

    question: str # 用户原始问题
    answer: str # 最终生成答案
    sources: List[RetrievedSource] # 与答案引用编号对齐的来源列表
    trace: Optional[Dict[str, Any] | RAGTrace] = None # 可选调试追踪信息
    retrieval_strategy: str = "llamaindex_router" # 对外展示的固定检索策略
    route_type: str = "general" # 问题路由类型

    def to_dict(self) -> Dict[str, Any]:
        """
        将 FinRAG 响应转换为 HTTP API 返回结构
        Returns:
            包含问题、答案、来源、路由类型和可选 trace 的字典
        """
        payload = {
            "question": self.question,
            "route_type": self.route_type,
            "retrieval_strategy": self.retrieval_strategy,
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
        }
        if self.trace is not None:
            payload["trace"] = self.trace.to_dict() if hasattr(self.trace, "to_dict") else self.trace
        return payload
