"""知识库运行时模块"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from finrag.application.knowledge_base_scope import KnowledgeBaseScope


@dataclass
class KnowledgeBaseRuntime:
    """知识库运行时模块"""

    scope: KnowledgeBaseScope # 知识库作用域
    data_module: Any # 数据准备模块
    index_module: Any # 索引构造模块
    generation_module: Any # 生成模块
    knowledge_query_engine: Optional[Any] = None # 知识库查询引擎
    auto_merge_retriever: Optional[Any] = None # 自动合并检索器
    hybrid_retriever: Optional[Any] = None # 混合检索器
    router_engine: Optional[Any] = None # 路由引擎
