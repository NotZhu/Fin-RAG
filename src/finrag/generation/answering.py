"""FinRAG 有据回答生成"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from llama_index.core.llms import CustomLLM

logger = logging.getLogger(__name__)

DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DashScopeCompatibleLLM(CustomLLM):
    """通过 DashScope OpenAI 兼容接口接入 LlamaIndex 的轻量 LLM"""

    api_key: str
    model_name: str = "qwen3.7-max"
    temperature: float = 0.1
    max_tokens: int = 2048
    api_base: str = DASHSCOPE_COMPATIBLE_BASE_URL
    request_timeout: int = 60

    @property
    def metadata(self) -> Any:
        """返回 LlamaIndex 路由和合成器需要的模型元数据"""
        from llama_index.core.llms import LLMMetadata

        return LLMMetadata(
            model_name=self.model_name,
            num_output=self.max_tokens,
            is_chat_model=True,
        )

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        """
        调用 DashScope OpenAI 兼容 chat/completions，并返回普通 completion 文本
        LlamaIndex 的 Router 会把 complete() 输出继续解析成 JSON，因此这里必须只返回
        choices[0].message.content，不能把 reasoning_content 或完整响应拼进去
        """
        from llama_index.core.llms import CompletionResponse

        raw = self._chat_completion(prompt, **kwargs)
        content = _extract_message_content(raw)
        return CompletionResponse(text=content, raw=raw)

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
        """兼容 LlamaIndex streaming 调用；兼容接口先按非流式返回一个完整块"""
        from llama_index.core.llms import CompletionResponse

        response = self.complete(prompt, formatted=formatted, **kwargs)
        yield CompletionResponse(text=response.text, delta=response.text, raw=response.raw)

    def _chat_completion(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """发送兼容模式 chat/completions 请求"""
        temperature = float(kwargs.get("temperature", self.temperature))
        max_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DashScope compatible-mode 请求失败: HTTP {exc.code} {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"DashScope compatible-mode 请求失败: {exc}") from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("DashScope compatible-mode 返回了非法 JSON") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("DashScope compatible-mode 响应不是 JSON 对象")
        return parsed


def _extract_message_content(response: dict[str, Any]) -> str:
    """从 OpenAI 兼容响应中提取 assistant content"""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("DashScope compatible-mode 响应缺少 choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("DashScope compatible-mode choices[0] 不是 JSON 对象")
    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("DashScope compatible-mode 响应缺少 message")
    content = message.get("content")
    if not str(content or "").strip():
        raise RuntimeError("DashScope compatible-mode 响应缺少 message.content")
    return str(content)


class GenerationIntegrationModule:
    """负责将生成模块与 DashScope OpenAI 兼容 LLM 集成，生成基于证据的回答"""

    def __init__(self, model_name: str = "qwen3.7-max", temperature: float = 0.1, max_tokens: int = 2048):
        """
        初始化生成集成模块
        Args:
            model_name: DashScope LLM 模型名称
            temperature: 生成温度
            max_tokens: 单次回答的最大 token 数量
        """
        self.model_name = model_name # LLM 模型名称
        self.temperature = float(temperature) # 温度参数
        self.max_tokens = int(max_tokens) # 最大令牌数
        self.llm = None # LLM 实例实例
        self.setup_llm()

    def setup_llm(self) -> None:
        """
        初始化 DashScope OpenAI 兼容 LLM；未配置 API key 时保留 llm 为空
        Returns:
            无返回值，初始化结果写入 self.llm
        """
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.info("DASHSCOPE_API_KEY 未设置，LLM 生成功能不可用")
            return
        # 直接使用 OpenAI 兼容 endpoint，避免 llama-index-llms-dashscope 原生接口与新模型不兼容。
        self.llm = DashScopeCompatibleLLM(
            model_name=self.model_name,
            api_key=api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
