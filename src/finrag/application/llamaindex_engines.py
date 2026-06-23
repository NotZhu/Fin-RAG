"""
LlamaIndex 查询引擎组装器
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import asyncio as _asyncio

from llama_index.core import PromptTemplate
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.postprocessor import SimilarityPostprocessor, PrevNextNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine, RouterQueryEngine, TransformQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import QueryBundle
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.storage.storage_context import StorageContext

from finrag.retrieval.llamaindex_native import MilvusNativeHybridRetriever, SentenceAwareTokenBudgetPostprocessor

logger = logging.getLogger(__name__)


class KnowledgeBaseUnavailableError(RuntimeError):
    """知识库未初始化错误"""

    def __init__(self, message: str, *, code: str = "knowledge_unavailable"):
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class LlamaIndexKnowledgeEngines:
    """
    LlamaIndex 知识引擎组装器
    """

    # 混合检索器
    hybrid_retriever: MilvusNativeHybridRetriever
    # 自动合并检索器
    auto_merge_retriever: AutoMergingRetriever
    # 自动合并查询引擎
    auto_merge_engine: RetrieverQueryEngine
    # 混合查询引擎
    hyde_engine: Any = None
    # 回退查询引擎
    step_back_engine: Any = None
    # 知识查询引擎
    knowledge_query_engine: Any = None



class LazyKnowledgeQueryEngine(BaseQueryEngine):
    """
    懒加载知识查询引擎
    """

    def __init__(
        self,
        system: Any,
        knowledge_base_id: str | None = None,
        callback_manager: CallbackManager | None = None,
    ) -> None:
        """
        初始化懒加载知识查询引擎
        Args:
            system: 系统对象
            knowledge_base_id: 可选知识库 ID，用于限定懒加载目标
            callback_manager: 回调管理器
        """
        self._system = system
        self._knowledge_base_id = knowledge_base_id
        # 初始化回调管理器
        super().__init__(callback_manager=callback_manager)

    def _query(self, query_bundle: QueryBundle) -> Response:
        """
        查询懒加载知识查询引擎
        Args:
            query_bundle: 查询绑定包
        Returns:
            Response: 懒加载知识查询引擎回答
        """
        # 确保知识库已初始化
        if self._knowledge_base_id is None:
            self._system.ensure_knowledge_base_ready()
        else:
            self._system.ensure_knowledge_base_ready(self._knowledge_base_id)
        # 确保知识查询引擎已初始化
        engine = self._system.knowledge_query_engine
        if engine is None:
            raise KnowledgeBaseUnavailableError(
                "knowledge_query_engine 未初始化",
                code="knowledge_unavailable",
            )
        response = engine.query(query_bundle)
        # 补充路由元数据
        _tag_response_metadata(response, selected_query_engine="knowledge_router", route_type="knowledge")
        return response

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        """
        异步查询懒加载知识查询引擎
        Args:
            query_bundle: 查询绑定包
        Returns:
            Response: 懒加载知识查询引擎回答
        """
        return await _asyncio.to_thread(self._query, query_bundle)

    def _get_prompt_modules(self) -> dict:
        """
        获取懒加载知识查询引擎的提示模块
        Returns:
            dict: 提示模块字典
        """
        return {}


class GeneralChatEngine(BaseQueryEngine):
    """
    通用问答引擎
    """

    def __init__(
        self,
        llm: Any,
        callback_manager: CallbackManager | None = None,
    ) -> None:
        """
        初始化通用问答引擎
        Args:
            llm: DashScope LLM实例
            callback_manager: 回调管理器
        """
        super().__init__(callback_manager=callback_manager)
        self.llm = llm

    def _query(self, query_bundle: QueryBundle) -> Response:
        prompt = (
            "你是通用问答助手，请直接回答用户问题。"
            "如果问题需要金融资料库证据，请说明需要切换到资料库问答。\n\n"
            f"用户问题: {query_bundle.query_str}\n回答:"
        )
        result = self.llm.complete(prompt)
        return Response(
            response=str(result),
            metadata={"selected_query_engine": "general_router", "route_type": "general"},
        )

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        """
        异步查询通用问答引擎
        Args:
            query_bundle: 查询绑定包
        Returns:
            Response: 通用问答引擎回答
        """
        return await _asyncio.to_thread(self._query, query_bundle)

    def _get_prompt_modules(self) -> dict:
        """
        获取通用问答引擎的提示模块
        Returns:
            dict: 提示模块字典
        """
        return {}


def build_top_router(*, system: Any, llm: Any | None = None, knowledge_base_id: str | None = None) -> Any:
    """
    构建顶部路由查询引擎
    Args:
        system: 系统对象
        llm: DashScope LLM实例
        knowledge_base_id: 知识库 ID
    Returns:
        RouterQueryEngine: 顶部路由查询引擎
    """
    from llama_index.core.tools import QueryEngineTool

    # 初始化工具列表
    tools: list[QueryEngineTool] = []
    # 知识查询引擎
    knowledge_engine: Any = LazyKnowledgeQueryEngine(system, knowledge_base_id=knowledge_base_id)
    tools.append(
        QueryEngineTool.from_defaults(
            query_engine=knowledge_engine,
            name="knowledge",
            description=(
                "已上传知识库资料问答。问题涉及资料库文档、公司经营复盘、收入质量、"
                "财务指标、经营计划、收入确认政策、行业投研、公告索引、金融制度、"
                "合规流程、产品规则、风险控制、反洗钱、账户开立、授信尽调时选择。"
            ),
        )
    )

    if llm is not None:
        # 通用问答引擎
        general_router_engine: Any = GeneralChatEngine(llm=llm)
        tools.append(
            QueryEngineTool.from_defaults(
                query_engine=general_router_engine,
                name="general",
                description=(
                    "日常对话、闲聊、写作、翻译、代码、常识问答，且不需要查询当前知识库资料时选择。"
                ),
            )
        )
    else:
        return knowledge_engine

    # 构建路由选择器
    selector = LLMSingleSelector.from_defaults(
        llm=llm,
        prompt_template_str=(
            "你是知识库问答的总路由器。根据用户问题判断应该查询当前内部资料库还是直接闲聊回答。\n"
            "只要问题询问已上传文档、公司、经营、财务、收入质量、财务信号、经营复盘、"
            "收入确认、投研、公告、制度或合规要求，优先选择 knowledge。\n"
            "只有明显不依赖知识库资料的闲聊、通用写作、翻译、代码和常识问题才选择 general。\n"
            "可用工具如下，编号范围为 1 到 {num_choices}:\n"
            "{context_list}\n"
            "用户问题: {query_str}\n"
            "请选择最合适的一个工具，并按要求输出 JSON。"
        )
    )
    # 返回顶部路由查询引擎
    return RouterQueryEngine(selector=selector, query_engine_tools=tools, llm=llm)


def _tag_response_metadata(response: Any, **metadata: Any) -> None:
    """
    给 LlamaIndex response 补充路由元数据
    Args:
        response: LlamaIndex response 对象
        metadata: 路由元数据字典
    """
    existing = getattr(response, "metadata", None) or {}
    existing.update(metadata)
    response.metadata = existing




def build_knowledge_engines(
    *,
    vector_index: Any,
    storage_context: StorageContext,
    config: Any,
    reranker: Optional[Any] = None,
    llm: Optional[Any] = None,
) -> LlamaIndexKnowledgeEngines:
    """
    构建 LlamaIndex 知识引擎栈
    Args:
        vector_index: 向量索引对象
        storage_context: 存储上下文
        config: 系统配置
        reranker: 重排序模型
        llm: LLM 模型
    Returns:
        知识引擎栈对象
       """

    # 构建 Milvus 原生混合检索器
    hybrid_retriever = MilvusNativeHybridRetriever(
        vector_index=vector_index, # 向量索引对象
        candidate_k=config.retrieval_candidate_k, # 候选数量
        top_k=config.top_k, # 返回数量
        rrf_k=config.rrf_k, # RRF 算法参数
    )

    # 构建自动合并检索器
    auto_merge_retriever = AutoMergingRetriever(
        vector_retriever=hybrid_retriever, # 底层混合检索器
        storage_context=storage_context, # 存储上下文
        simple_ratio_thresh=config.auto_merge_ratio_threshold, # 自动合并阈值
    )

    # 构建文档存储
    docstore = getattr(storage_context, "docstore", None)
    # 构建节点后处理器
    postprocessors: list[Any] = build_node_postprocessors(config=config, reranker=reranker, docstore=docstore)
    # 构建 LLM 实例
    engine_llm = llm
    
    # 构建自动合并查询引擎
    auto_merge_engine = RetrieverQueryEngine.from_args(
        retriever=auto_merge_retriever, # 自动合并检索器
        llm=engine_llm, # LLM 实例
        node_postprocessors=postprocessors, # 节点后处理器
        response_mode="compact", # 响应模式，压缩输出：只包含必要信息
        text_qa_template=grounded_answer_template(),
        streaming=llm is not None, # 是否开启流式输出
    )

    # HYDE 查询引擎
    hyde_engine: Any = None
    # 退步查询引擎
    step_back_engine: Any = None
    # 知识查询引擎
    knowledge_query_engine: Any = auto_merge_engine # 默认使用自动合并查询引擎

    if llm is not None:
        # 构建HYDE查询引擎
        hyde_engine = _build_hyde_engine(auto_merge_engine, llm)
        # 构建退步查询引擎
        step_back_engine = _build_step_back_engine(auto_merge_engine, llm)

    if llm is not None:
        # 构建知识查询引擎，包含所有子引擎，用于路由用户问题
        knowledge_query_engine = _build_knowledge_router(
            auto_merge_engine=auto_merge_engine,
            hyde_engine=hyde_engine,
            step_back_engine=step_back_engine,
            llm=llm,
        )

    return LlamaIndexKnowledgeEngines(
        hybrid_retriever=hybrid_retriever,
        auto_merge_retriever=auto_merge_retriever,
        auto_merge_engine=auto_merge_engine,
        hyde_engine=hyde_engine,
        step_back_engine=step_back_engine,
        knowledge_query_engine=knowledge_query_engine,
    )




def _build_hyde_engine(
    auto_merge_engine: Any,
    llm: Any,
) -> Optional[Any]:
    """
    构建HYDE查询引擎
    Args:
        auto_merge_engine: 自动合并查询引擎
        llm: LLM 模型
    """
    try:
        from llama_index.core.indices.query.query_transform import HyDEQueryTransform

        # 构建HYDE查询转换器
        hyde = HyDEQueryTransform(llm=llm, include_original=True) # 检索 embedding 时不只用 LLM 生成的假想答案，也保留原始问题
        # 构建HYDE查询引擎
        return TransformQueryEngine(query_engine=auto_merge_engine, query_transform=hyde)
    except Exception as exc:
        logger.warning("HyDE engine 构建失败，跳过: %s", exc)
        return None


def _build_step_back_engine(
    auto_merge_engine: Any,
    llm: Any,
) -> Optional[Any]:
    """
    构建退步查询引擎
    Args:
        auto_merge_engine: 自动合并查询引擎
        llm: LLM 模型
    """
    try:
        from llama_index.core.indices.query.query_transform import StepDecomposeQueryTransform
        # 构建退步查询转换器
        step_back = StepDecomposeQueryTransform(llm=llm)
        # 构建退步查询引擎
        return TransformQueryEngine(query_engine=auto_merge_engine, query_transform=step_back)
    except Exception as exc:
        logger.warning("Step-Back engine 构建失败，跳过: %s", exc)
        return None


def _build_knowledge_router(
    *,
    auto_merge_engine: Any,
    hyde_engine: Optional[Any] = None,
    step_back_engine: Optional[Any] = None,
    llm: Any,
) -> Any:
    """
    构建知识查询引擎，包含所有子引擎，用于路由用户问题
    Args:
        auto_merge_engine: 自动合并查询引擎
        hyde_engine: Hyde 查询引擎
        step_back_engine: StepBack 查询引擎
        llm: LLM 模型
    Returns:
        知识查询引擎实例
    """
    try:
        from llama_index.core.tools import QueryEngineTool
        # 构建知识查询引擎的子工具
        router_tools: list[QueryEngineTool] = []
        if hyde_engine is not None:
            router_tools.append(
                QueryEngineTool.from_defaults(
                    query_engine=hyde_engine,
                    name="hyde",
                    description="精确事实、数值、条款、日期查询。涉及具体金额、百分比、时间、条款编号时选择。",
                )
            )
        if step_back_engine is not None:
            router_tools.append(
                QueryEngineTool.from_defaults(
                    query_engine=step_back_engine,
                    name="step_back",
                    description="原因、背景、制度目的类问题。涉及'为什么'、'立法背景'、'监管目的'时选择。",
                )
            )
        if auto_merge_engine is not None:
            router_tools.append(
                QueryEngineTool.from_defaults(
                    query_engine=auto_merge_engine,
                    name="auto_merge",
                    description=(
                        "金融制度与合规问答。问题涉及具体制度条款、业务流程、合规要求、"
                        "风险控制措施、反洗钱规定、账户开立流程、授信尽调要求时选择。"
                    ),
                )
            )

        # 构建知识查询引擎的路由选择器
        selector = LLMSingleSelector.from_defaults(
            llm=llm,
            prompt_template_str=(
                "你是一个金融资料库路由选择器。根据用户问题选择最合适的查询工具。\n"
                "可用工具如下，编号范围为 1 到 {num_choices}:\n"
                "{context_list}\n"
                "用户问题: {query_str}\n"
                "请选择最合适的一个工具，并按要求输出 JSON。"
            )
        )
        # 构建知识查询引擎的路由引擎
        router = RouterQueryEngine(
            selector=selector,
            query_engine_tools=router_tools,
            llm=llm,
        )
        logger.info("Knowledge RouterQueryEngine 构建成功，注册 %d 个工具", len(router_tools))
        return router
    except Exception as exc:
        logger.warning("Knowledge Router 构建失败，退回 auto_merge_engine: %s", exc)
        return auto_merge_engine


def build_node_postprocessors(*, config: Any, reranker: Optional[Any] = None, docstore: Any = None) -> List[Any]:
    """
    构建节点后处理器
    Args:
        config: 配置对象
        reranker: 二阶段精排模型
        docstore: 文档存储对象
    Returns:
        节点后处理器列表
    """
    processors: List[Any] = []
    # 丢弃不相关 chunk；0.0 表示不启用相似度阈值过滤
    cutoff = float(getattr(config, "score_threshold", 0.0) or 0.0)
    if cutoff > 0.0:
        processors.append(SimilarityPostprocessor(similarity_cutoff=cutoff))

    # 二阶段精排
    if reranker is not None:
        processors.append(reranker)

    # 给精排后的 chunk 加前后相邻节点
    neighbor_window = max(int(getattr(config, "neighbor_window", 1) or 0), 0)
    if docstore is not None and neighbor_window > 0:
        processors.append(PrevNextNodePostprocessor(docstore=docstore, num_nodes=neighbor_window, mode="both"))

    # 硬截断兜底
    processors.append(SentenceAwareTokenBudgetPostprocessor(token_budget=config.context_token_budget))
    return processors


def grounded_answer_template() -> PromptTemplate:
    """
    构建基于证据回答的提示模板
    """
    return PromptTemplate(
        "你是金融机构内部资料库助手请根据金融资料库证据回答用户问题\n\n"
        "用户问题: {query_str}\n\n"
        "金融资料库证据:\n{context_str}\n\n"
        "回答规则：\n"
        "1. 只能基于证据回答，不要使用未检索到的外部资料\n"
        "2. 证据不足时明确说明'当前资料不足'\n"
        "3. 不编造监管条款、日期、收益、流程或来源\n"
        "4. 不要在回答末尾输出'[1] 来源：...'这类来源清单，来源由系统单独展示\n"
        "5. 不输出 document_id、chunk_id、node_id 等内部字段\n\n"
        "回答:"
    )
