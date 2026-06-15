"""FinRAG 有据回答生成"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class GenerationIntegrationModule:
    """负责将生成模块与 DashScope LLM 集成，生成基于证据的回答"""

    def __init__(self, model_name: str = "qwen-max", temperature: float = 0.1, max_tokens: int = 2048):
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
        初始化 DashScope LLM；未配置 API key 时保留 llm 为空
        Returns:
            无返回值，初始化结果写入 self.llm
        """
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            logger.info("DASHSCOPE_API_KEY 未设置，LLM 生成功能不可用")
            return
        try:
            from llama_index.llms.dashscope import DashScope
            # 初始化 DashScope LLM 实例
            self.llm = DashScope(
                model_name=self.model_name,
                api_key=api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            logger.warning("DashScope LlamaIndex LLM 不可用，LLM 生成功能不可用: %s", exc)
