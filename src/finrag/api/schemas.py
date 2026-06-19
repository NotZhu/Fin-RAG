"""FinRAG HTTP API 的 Pydantic 模型"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from finrag.core.config import validate_knowledge_base_id


class AskRequest(BaseModel):
    """问答接口请求体模型"""

    question: str = Field(..., min_length=1) # 用户问题文本
    return_sources: bool = True # 是否返回来源证据
    return_trace: bool = False # 是否返回调试 trace

    @model_validator(mode="before")
    @classmethod
    def knowledge_base_id_must_not_be_in_body(cls, data: Any) -> Any:
        """
        拒绝在问答请求体中传入 knowledge_base_id，知识库作用域只能来自 URL。
        Args:
            data: 原始请求体
        Returns:
            未修改的请求体
        """
        if isinstance(data, dict) and "knowledge_base_id" in data:
            raise ValueError("knowledge_base_id 必须通过 URL 路径提供")
        return data

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        """
        校验问题内容不能只包含空白字符
        Args:
            value: 请求中的 question 字段
        Returns:
            去除首尾空白后的问题文本
        """
        question = value.strip()
        if not question:
            raise ValueError("question 不能为空")
        return question

class CreateKnowledgeBaseRequest(BaseModel):
    """创建知识库请求体模型"""

    # 知识库 ID
    knowledge_base_id: str = Field(..., min_length=1)

    @field_validator("knowledge_base_id") # 校验知识库 ID 是否为空，只包含安全字符
    @classmethod
    def knowledge_base_id_must_be_safe(cls, value: str) -> str:
        """
        校验知识库 ID 只包含安全字符，避免路径或过滤条件异常
        Args:
            value: 请求中的 knowledge_base_id 字段
        Returns:
            通过校验的知识库 ID
        """
        return validate_knowledge_base_id(value)
