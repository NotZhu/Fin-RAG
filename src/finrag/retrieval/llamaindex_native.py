"""LlamaIndex-native retrieval helpers for FinRAG."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.utils import get_tokenizer
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from finrag.retrieval.search import build_metadata_filters, _matches_filters


class HybridRetrieverUnavailable(RuntimeError):
    """Raised when Milvus native dense+sparse hybrid retrieval is not usable."""

    def __init__(self, reason: str):
        self.code = "hybrid_retriever_unavailable"
        super().__init__(f"{self.code}: {reason}")


class MilvusNativeHybridRetriever(BaseRetriever):
    """
    基于 Milvus 原生密疏混合的 LlamaIndex BaseRetriever 适配器
    """

    def __init__(
        self,
        *,
        vector_index: Any,
        candidate_k: int = 10,
        top_k: int = 3,
        rrf_k: int = 60,
        filters: Optional[Dict[str, Any]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """
        初始化基于 Milvus 原生密疏混合的 LlamaIndex BaseRetriever 适配器
        Args:
            vector_index: 向量索引对象
            candidate_k: 候选数量
            top_k: 返回数量
            rrf_k: RRF 算法参数
            filters: 元数据筛选条件
            callback_manager: 回调管理器
        """
        self.vector_index = vector_index # 向量索引对象
        self.candidate_k = max(int(candidate_k), 1) # 候选数量
        self.top_k = max(int(top_k), 1) # 返回数量
        self.rrf_k = int(rrf_k) # RRF 算法参数
        self.filters = dict(filters or {}) # 元数据筛选条件
        self.last_hybrid_trace: Dict[str, Any] = {} # 最后一次混合检索的跟踪信息
        self._ensure_sparse_vector_store() # 确保向量存储支持稀疏向量
        super().__init__(callback_manager=callback_manager) # 初始化父类，设置回调管理器

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        vector_store = self._ensure_sparse_vector_store()
        metadata_filters = build_metadata_filters(self.filters)
        kwargs: Dict[str, Any] = {
            "vector_store_query_mode": VectorStoreQueryMode.HYBRID,
            "similarity_top_k": self.candidate_k,
            "sparse_top_k": self.candidate_k,
            "hybrid_top_k": self.candidate_k,
        }
        if metadata_filters is not None:
            kwargs["filters"] = metadata_filters
        retriever = self.vector_index.as_retriever(**kwargs)
        results = retriever.retrieve(query_bundle)
        filtered = [item for item in results if _matches_filters(item.node, self.filters)]
        filtered.sort(key=lambda item: float(item.score or 0.0), reverse=True)
        self.last_hybrid_trace = {
            "hybrid_provider": "milvus",
            "hybrid_mode": "native_dense_sparse",
            "hybrid_ranker": getattr(vector_store, "hybrid_ranker", "RRFRanker"),
            "rrf_k": int((getattr(vector_store, "hybrid_ranker_params", {}) or {}).get("k", self.rrf_k)),
            "candidate_k": self.candidate_k,
        }
        return filtered[: self.top_k]

    def _ensure_sparse_vector_store(self) -> Any:
        """
        确保向量存储支持稀疏向量
        Returns:
            向量存储对象
        """
        vector_store = getattr(self.vector_index, "vector_store", None) or getattr(self.vector_index, "_vector_store", None)
        if vector_store is None:
            raise HybridRetrieverUnavailable("Milvus vector store 未初始化")
        if not bool(getattr(vector_store, "enable_sparse", False)):
            raise HybridRetrieverUnavailable("Milvus collection 缺少 sparse vector schema")
        return vector_store


class SentenceAwareTokenBudgetPostprocessor(BaseNodePostprocessor):
    """
    基于句子边界硬截断的节点后处理器
    """
    
    # 最大 token 数量
    token_budget: int = Field(default=2400)

    @classmethod
    def class_name(cls) -> str:
        return "SentenceAwareTokenBudgetPostprocessor"

    def __init__(self, token_budget: int = 2400, **kwargs: Any) -> None:
        """
        初始化基于句子边界硬截断的节点后处理器
        Args:
            token_budget: 最大 token 数量
        """
        super().__init__(token_budget=max(int(token_budget), 1), **kwargs)

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """
        对节点列表进行硬截断处理，确保总 token 数量不超过指定预算
        Args:
            nodes: 输入节点列表
        Returns:
            处理后的节点列表
        """
        # 初始化结果列表
        selected: List[NodeWithScore] = []
        # 初始化总 token 数量
        total_tokens = 0
        # 初始化已处理节点 ID 集合
        seen: set[str] = set()
        for item in nodes:
            node_id = item.node.node_id
            if node_id in seen:
                continue
            # 计算节点 token 数量
            node_tokens = self._estimate_tokens(item.node.get_content())
            # 如果添加节点后不超过预算，直接添加
            if total_tokens + node_tokens <= self.token_budget:
                selected.append(item)
                seen.add(node_id)
                total_tokens += node_tokens
                continue
            # 计算剩余预算
            remaining = self.token_budget - total_tokens
            if remaining <= 0:
                break
            # 如果结果列表为空，说明当前节点是第一个节点，直接截断并添加
            if not selected:
                selected.append(self._truncate_item(item, self.token_budget))
            break
        return selected

    def _truncate_item(self, item: NodeWithScore, token_budget: int) -> NodeWithScore:
        """
        对单个节点进行硬截断处理，确保总 token 数量不超过指定预算
        Args:
            item: 输入节点
            token_budget: 最大 token 数量
        Returns:
            戄断后的节点
        """
        node = item.node
        if not isinstance(node, TextNode):
            return item
        text = node.get_content()
        # 截断后文本
        truncated_text = self._truncate_to_sentence_boundary(text, token_budget)
        # 创建新节点
        truncated_node = node.model_copy(deep=True)
        # 更新节点文本
        truncated_node.text = truncated_text
        metadata = dict(truncated_node.metadata or {})
        # 标记节点为截断
        metadata["truncated"] = True
        # 更新节点元数据
        truncated_node.metadata = metadata
        # 返回截断后的节点
        return NodeWithScore(node=truncated_node, score=item.score)

    def _truncate_to_sentence_boundary(self, text: str, token_budget: int) -> str:
        """
        截断文本到句子边界，确保总 token 数量不超过指定预算
        Args:
            text: 输入文本
            token_budget: 最大 token 数量
        Returns:
            截断后的文本
        """
        # 分割文本为句子列表
        sentences = self._split_sentences(text)
        # 初始化结果列表
        selected: List[str] = []
        # 初始化总 token 数量
        total = 0
        for sentence in sentences:
            # 计算句子 token 数量
            sentence_tokens = self._estimate_tokens(sentence)
            # 如果添加句子后不超过预算，直接添加
            if total + sentence_tokens > token_budget:
                break
            selected.append(sentence)
            total += sentence_tokens
        if selected:
            # 合并选中的句子
            return "".join(selected).strip()
        # 如果没有选中的句子，截断文本到 token 数量
        return self._truncate_by_token(text, token_budget).strip()

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        将文本按句子边界分割为句子列表
        Args:
            text: 输入文本
        Returns:
            句子列表
        """
        parts = re.findall(r"[^。！？!?；;\n]+[。！？!?；;]|\S+[^\S\n]*", text or "")
        return [part for part in parts if part.strip()]

    def _truncate_by_token(self, text: str, token_budget: int) -> str:
        """
        截断文本到指定 token 数量
        Args:
            text: 输入文本
            token_budget: 最大 token 数量
        Returns:
            截断后的文本
        """
        if token_budget <= 0:
            return ""
        # 计算截断位置，截断位置不能小于 1，不能大于文本长度，必须是整数倍
        cap = max(1, min(len(text or ""), int(token_budget)))
        return (text or "")[:cap]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        估计文本中的 token 数量
        Args:
            text: 输入文本
        Returns:
            估计的 token 数量
        """
        try:
            return len(get_tokenizer()(text or ""))
        except Exception:
            return len(text or "")
