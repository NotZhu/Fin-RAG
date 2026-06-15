"""FinRAG HTTP API 的 Pydantic 模型"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from finrag.core.config import validate_knowledge_base_id


class AskRequest(BaseModel):
    """问答接口请求体模型"""

    question: str = Field(..., min_length=1) # 用户问题文本
    knowledge_base_id: str | None = None # 可选资料库 ID，为空时使用系统默认值
    return_sources: bool = True # 是否返回来源证据
    return_trace: bool = False # 是否返回调试 trace

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

    @field_validator("knowledge_base_id")
    @classmethod
    def knowledge_base_id_must_be_safe(cls, value: str | None) -> str | None:
        """
        校验资料库 ID 只包含安全字符，避免路径或过滤条件异常
        Args:
            value: 请求中的 knowledge_base_id 字段
        Returns:
            通过校验的资料库 ID
        """
        if value in (None, ""):
            return None
        return validate_knowledge_base_id(value)
