"""
LlamaIndex 查询引擎组装器
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, TYPE_CHECKING

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

if TYPE_CHECKING:
    from llama_index.core import Document
    from llama_index.core.base.embeddings.base import BaseEmbedding

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
    # 摘要索引
    summary_index: Any = None
    # 摘要查询引擎
    summary_engine: Any = None
    # 子问题查询引擎
    sub_question_engine: Any = None
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
        return engine.query(query_bundle)

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
        return Response(response=str(result))

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
                "金融内部资料库问答。问题涉及金融制度、合规流程、产品规则、"
                "风险控制、反洗钱、账户开立、授信尽调、已上传的内部资料文档时选择。"
            ),
        )
    )

    if llm is not None:
        # 通用问答引擎
        general_engine: Any = GeneralChatEngine(llm=llm)
        tools.append(
            QueryEngineTool.from_defaults(
                query_engine=general_engine,
                name="general",
                description=(
                    "日常对话、闲聊、写作、翻译、代码、常识问答、非金融业务问题时选择。"
                ),
            )
        )
    else:
        return knowledge_engine

    # 构建路由选择器
    selector = LLMSingleSelector.from_defaults(
        llm=llm,
        prompt_template_str=(
            "你是金融资料库的总路由器。根据用户问题判断应该查询内部资料库还是直接闲聊回答。\n"
            "可用工具如下，编号范围为 1 到 {num_choices}:\n"
            "{context_list}\n"
            "用户问题: {query_str}\n"
            "请选择最合适的一个工具，并按要求输出 JSON。"
        )
    )
    # 返回顶部路由查询引擎
    return RouterQueryEngine(selector=selector, query_engine_tools=tools, llm=llm)




def build_knowledge_engines(
    *,
    vector_index: Any,
    storage_context: StorageContext,
    config: Any,
    reranker: Optional[Any] = None,
    llm: Optional[Any] = None,
    documents: Optional[Sequence["Document"]] = None,
    embed_model: Optional["BaseEmbedding"] = None,
) -> LlamaIndexKnowledgeEngines:
    """
    构建 LlamaIndex 知识引擎栈
    Args:
        vector_index: 向量索引对象
        storage_context: 存储上下文
        config: 系统配置
        reranker: 重排序模型
        llm: LLM 模型
        documents: 文档列表
        embed_model: 嵌入模型
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

    # 摘要索引
    summary_index: Any = None
    # 摘要查询引擎
    summary_engine: Any = None
    # HYDE 查询引擎
    hyde_engine: Any = None
    # 退步查询引擎
    step_back_engine: Any = None
    # 子问题查询引擎
    sub_question_engine: Any = None
    # 知识查询引擎
    knowledge_query_engine: Any = auto_merge_engine # 默认使用自动合并查询引擎

    if llm is not None and documents:
        # 构建摘要索引和摘要查询引擎
        summary_index, summary_engine = _build_summary_engine(
            documents=documents, embed_model=embed_model, llm=llm, config=config
        )
    
    if llm is not None:
        # 构建HYDE查询引擎
        hyde_engine = _build_hyde_engine(auto_merge_engine, llm)
        # 构建退步查询引擎
        step_back_engine = _build_step_back_engine(auto_merge_engine, llm)
        # 构建子问题查询引擎
        sub_question_engine = _build_sub_question_engine(
            summary_engine=summary_engine,
            auto_merge_engine=auto_merge_engine,
            llm=llm,
        )

    if llm is not None:
        # 构建知识查询引擎，包含所有子引擎，用于路由用户问题
        knowledge_query_engine = _build_knowledge_router(
            summary_engine=summary_engine,
            auto_merge_engine=auto_merge_engine,
            hyde_engine=hyde_engine,
            step_back_engine=step_back_engine,
            sub_question_engine=sub_question_engine,
            llm=llm,
        )

    return LlamaIndexKnowledgeEngines(
        hybrid_retriever=hybrid_retriever,
        auto_merge_retriever=auto_merge_retriever,
        auto_merge_engine=auto_merge_engine,
        summary_index=summary_index,
        summary_engine=summary_engine,
        sub_question_engine=sub_question_engine,
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


def _build_summary_engine(
    *,
    documents: Sequence[Any],
    embed_model: Optional[Any],
    llm: Any,
    config: Any,
) -> tuple[Any, Any]:
    """
    构建文档总结索引和查询引擎
    Args:
        documents: 输入的金融文档序列，每个文档是一个字符串
        embed_model: 嵌入模型，用于将文档转换为向量表示
        llm: 用于生成总结的 LLM 模型
        config: 用于配置索引构建的参数字典
    Returns:
        包含文档总结索引和查询引擎的元组
    """
    try:
        from llama_index.core.indices.document_summary import DocumentSummaryIndex

        # 构建文档总结索引
        # from_documents 会先用默认 transformations 将每个 Document 切成多个 TextNode
        # 默认 splitter 是 SentenceSplitter(chunk_size=1024, chunk_overlap=200)，会尽量按句子/段落边界切分
        # 然后按 ref_doc_id 将这些 nodes 分组，对每个原始文档的全部 nodes 调用 LLM 生成一条摘要 TextNode
        # 原文 nodes 和 summary node 会写入 summary_index 自己的 docstore
        # 默认 embed_summaries=True，会把摘要 TextNode 向量化后写入该 index 自己的 vector_store
        # index_store 中保存 doc_id、summary_node_id、原文 node_id 之间的映射关系
        summary_index = DocumentSummaryIndex.from_documents(
            list(documents),
            embed_model=embed_model, # 嵌入模型
            llm=llm, # 用于生成总结的 LLM 模型
            show_progress=True, # 显示构建进度
            # 自定义总结查询模板
            summary_query=(
                "用中文总结以下金融文档的核心内容、适用范围与关键条款，保留制度名称、文号和重要数字: "
                "{context_str}"
            ),
        )
        
        # 将文档总结索引构建为文档总结查询引擎
        summary_engine = summary_index.as_query_engine(
            response_mode="tree_summarize", # 分批总结检索内容，再逐层汇总成最终答
            similarity_top_k=3, # 在摘要向量中找最相关的 3 个文档摘要
            llm=llm,
            streaming=True,
        )
        logger.info("文档总结索引构建完成，包含 %d 个文档", len(list(documents)))
        return summary_index, summary_engine
    except Exception as exc:
        logger.warning("文档总结索引构建失败，跳过: %s", exc)
        return None, None


def _build_sub_question_engine(
    *,
    summary_engine: Optional[Any],
    auto_merge_engine: Any,
    llm: Any,
) -> Optional[Any]:
    """
    构建子问题查询引擎
    Args:
        summary_engine: 文档总结查询引擎
        auto_merge_engine: 自动合并查询引擎
        llm: LLM 模型
    Returns:
        子问题查询引擎实例
    """
    try:
        from llama_index.core.query_engine import SubQuestionQueryEngine
        from llama_index.core.tools import QueryEngineTool, ToolMetadata
        
        # 构建子问题查询引擎的子工具
        tools: list[QueryEngineTool] = []

        if summary_engine is not None:
            tools.append(
                # 将文档总结查询引擎添加为子工具
                QueryEngineTool(
                    query_engine=summary_engine,
                    # 子工具元数据
                    metadata=ToolMetadata(
                        name="summary",
                        description="文档总结、资料库概览，用于问某份文档或主题的主要内容",
                    ),
                )
            )
        tools.append(
            QueryEngineTool(
                # 将自动合并查询引擎添加为子工具
                query_engine=auto_merge_engine,
                # 子工具元数据
                metadata=ToolMetadata(
                    name="auto_merge",
                    description="金融制度、合规流程、产品规则、风险控制、授信尽调、反洗钱等具体条款",
                ),
            )
        )
        # 构建子问题查询引擎
        engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=tools,
            llm=llm,
            verbose=False # 不显示详细日志
        )
        logger.info("SubQuestionQueryEngine 构建成功，子工具数量=%d", len(tools))
        return engine
    except Exception as exc:
        logger.warning("SubQuestionQueryEngine 构建失败，跳过: %s", exc)
        return None


def _build_knowledge_router(
    *,
    summary_engine: Optional[Any],
    auto_merge_engine: Any,
    sub_question_engine: Optional[Any],
    hyde_engine: Optional[Any] = None,
    step_back_engine: Optional[Any] = None,
    llm: Any,
) -> Any:
    """
    构建知识查询引擎，包含所有子引擎，用于路由用户问题
    Args:
        summary_engine: 文档总结查询引擎
        auto_merge_engine: 自动合并查询引擎
        sub_question_engine: 子问题查询引擎
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
        if summary_engine is not None:
            router_tools.append(
                QueryEngineTool.from_defaults(
                    query_engine=summary_engine,
                    name="summary",
                    description="文档总结与资料概览。问题涉及某份文档主旨、主题概述、资料库整体情况时选择。",
                )
            )
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
        if sub_question_engine is not None:
            router_tools.append(
                QueryEngineTool.from_defaults(
                    query_engine=sub_question_engine,
                    name="sub_question",
                    description=(
                        "复杂金融问题。问题包含对比、并列、多个条件、"
                        "不同制度间关系分析时选择。例如'对比A和B'、'A与B的关系'。"
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

    # 给精排后的 chunk 加前后各 1 个相邻节点
    if docstore is not None:
        processors.append(PrevNextNodePostprocessor(docstore=docstore, num_nodes=1, mode="both"))

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
