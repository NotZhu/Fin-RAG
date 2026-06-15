"""Jina 兼容 HTTP 重排器工厂"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any, Optional

from llama_index.core.schema import NodeWithScore, QueryBundle

logger = logging.getLogger(__name__)


class JinaCompatibleReranker:
    """对 Jina 风格 rerank endpoint 的轻量 LlamaIndex 兼容封装"""

    provider = "jina" # reranker provider 标识

    def __init__(self, *, model: str, endpoint: str, api_key: str = "", top_n: int = 3, timeout: float = 10.0):
        # 初始化模型名称
        self.model = model or "jina-reranker-v2-base-multilingual"
        # 初始化 endpoint
        self.endpoint = (endpoint or "").strip()
        # 初始化 API 密钥
        self.api_key = (api_key or "").strip()
        # 初始化返回的相关节点数量
        self.top_n = max(1, int(top_n or 1))
        # 初始化超时时间
        self.timeout = float(timeout)
        # 初始化最后调用时间
        self.last_elapsed_ms = 0.0

    def postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
        query_str: str | None = None,
    ) -> list[NodeWithScore]:
        """
        调用远端 rerank 服务并按相关性重排节点
        Args:
            nodes: 待重排的候选节点列表
            query_bundle: 可选 LlamaIndex 查询对象
            query_str: 可选查询字符串，优先级高于 query_bundle
        Returns:
            重排并截断后的节点列表
        """
        query = query_str or (query_bundle.query_str if query_bundle is not None else "")
        # 提取候选节点的文本内容
        documents = [item.node.get_content() for item in nodes]
        # 构建请求体
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(self.top_n, len(nodes)),
        }

        started = time.perf_counter()
        # 发送 JSON 请求
        data = self._post_json(payload)
        self.last_elapsed_ms = (time.perf_counter() - started) * 1000

        # 重排节点
        reranked: list[NodeWithScore] = []
        for result in data.get("results", []):
            # 对应候选节点的索引
            index = int(result.get("index"))
            if index < 0 or index >= len(nodes):
                continue
            item = nodes[index]
            # 更新节点的相关性分数
            item.score = float(result.get("relevance_score", result.get("score", item.score or 0.0)))
            reranked.append(item)
        return reranked[: self.top_n]

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        向 Jina 兼容 endpoint 发送 JSON 请求
        Args:
            payload: rerank 请求体
        Returns:
            解析后的 JSON 响应字典
        """
        if not self.endpoint:
            raise RuntimeError("使用 jina reranker 时必须配置 RAG_RERANKER_ENDPOINT")
        # 构建请求体
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        # 构建请求头
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # 发送 POST 请求
        request = urllib.request.Request(self.endpoint, data=body, headers=headers, method="POST")
        # 发送请求并读取响应
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            # 返回解析后的 JSON 响应字典
            return json.loads(response.read().decode("utf-8"))


def build_reranker(provider: str, model_name: str, endpoint: str = "", api_key: str = "", top_n: int = 3) -> Optional[Any]:
    """
    构建已配置的 reranker，仅支持 none 和 Jina 兼容 HTTP
    Args:
        provider: reranker provider 标识，支持 "none" 或 "jina"
        model_name: Jina 兼容 reranker 的模型名称
        endpoint: Jina 兼容 reranker 的 HTTP endpoint URL
        api_key: 可选的 API 密钥，用于授权访问 Jina 兼容 reranker
        top_n: Jina 兼容 reranker 返回的相关节点数量
    Returns:
        配置的 reranker 实例，或在配置无效时返回 None
    """
    provider = (provider or "none").strip().lower()
    if provider in {"", "none", "disabled"}:
        return None
    if provider != "jina":
        logger.warning("不支持的 reranker provider %s，rerank 将降级", provider)
        return None
    if not endpoint:
        logger.warning("使用 jina reranker 时必须配置 RAG_RERANKER_ENDPOINT，rerank 将降级")
        return None
    # 构建 Jina 兼容 reranker 实例
    return JinaCompatibleReranker(
        model=model_name or "jina-reranker-v2-base-multilingual",
        endpoint=endpoint,
        api_key=api_key,
        top_n=top_n,
    )
