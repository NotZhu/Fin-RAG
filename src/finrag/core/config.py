"""FinRAG 包配置"""

import os
import re
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict

PACKAGE_DIR = Path(__file__).resolve().parents[1] # 当前文件的绝对路径的上上级目录
SRC_DIR = PACKAGE_DIR.parent # 包目录的上一级目录，即项目根目录的 src 目录
PROJECT_ROOT = SRC_DIR.parent # 项目根目录，即项目根目录的上一级目录，即项目根目录
KNOWLEDGE_BASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$") # 资料库 ID 安全格式


def resolve_project_path(path_value: str) -> str:
    """
    将配置路径解析为项目内的绝对路径
    Args:
        path_value: 原始路径配置，可为相对路径、绝对路径或带 ~ 的用户路径
    Returns:
        解析后的绝对路径字符串
    """
    path = Path(path_value).expanduser() # 将路径中的用户~替换为用户主目录
    if not path.is_absolute(): # 如果路径不是绝对路径
        path = PROJECT_ROOT / path # 则将路径拼接到项目根目录
    return str(path.resolve())


def validate_knowledge_base_id(value: str) -> str:
    """
    校验资料库 ID 格式，确保只包含字母、数字、下划线和连字符
    Args:
        value: 待校验的资料库 ID
    Returns:
        清理空白后的资料库 ID
    """
    knowledge_base_id = (value or "").strip()
    if not KNOWLEDGE_BASE_ID_PATTERN.fullmatch(knowledge_base_id):
        raise ValueError("knowledge_base_id 只能包含字母、数字、连字符或下划线")
    return knowledge_base_id


@dataclass(init=False)
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置
    data_path: str = field(default_factory=lambda: str(PROJECT_ROOT / "data" / "documents")) # 默认数据目录
    upload_dir: str = field(default_factory=lambda: str(PROJECT_ROOT / "storage" / "uploads")) # 上传临时目录
    knowledge_base_id: str = "finance" # 默认资料库 ID
    database_url: str = "postgresql://finrag:finrag@localhost:5432/finrag" # PostgreSQL 连接地址
    redis_url: str = "redis://localhost:6379/0" # Redis 缓存连接地址
    milvus_host: str = "localhost" # Milvus 服务地址
    milvus_port: int = 19530 # Milvus 服务端口
    milvus_collection: str = "finrag_leaf_nodes" # Milvus 叶子节点 collection 名称

    # 模型配置
    embedding_model: str = "text-embedding-v4" # Embedding 模型名称
    llm_model: str = "qwen-max" # LLM 生成模型名称

    # 检索配置
    top_k: int = 3 # 最终返回给 LLM 的证据块数量
    retrieval_candidate_k: int = 10 # 检索候选数量
    rrf_k: int = 60 # 倒数排名融合参数
    retrieval_strategy: str = "llamaindex_router" # 查询编排策略
    llamaindex_index_store_dir: str = field(default_factory=lambda: str(PROJECT_ROOT / "storage" / "llamaindex")) # LlamaIndex index metadata 本地目录
    score_threshold: float = 0.0 # 检索候选分数阈值，低于该分数的候选将被过滤
    reranker_provider: str = "none" # 是否启用重排序模型
    reranker_model: str = "jina-reranker-v2-base-multilingual" # 重排序模型名称
    reranker_endpoint: str = "" # Jina 兼容 rerank endpoint
    reranker_api_key: str = "" # Jina 兼容 rerank API key
    reranker_top_n: int = 3 # 重排序后最终返回给 LLM 的证据块数量
    auto_merge_ratio_threshold: float = 0.5 # 自动合并检索器阈值
    context_token_budget: int = 2400 # 上下文 token 硬预算
    neighbor_window: int = 1 # 前后相邻节点扩展数量
    max_upload_bytes: int = 20 * 1024 * 1024 # 单个上传文件最大字节数

    # 生成配置
    temperature: float = 0.1 # LLM 生成温度，控制生成文本的随机性
    max_tokens: int = 2048 # LLM 生成的最大 token 数量，超过该数量将被截断

    def __init__(self, **kwargs: Any):
        """
        初始化配置并拒绝已移除或未知的配置项
        Args:
            **kwargs: 覆盖默认值的配置字段
        Raises:
            TypeError: 缺少必需字段或传入未知字段
        """
        config_fields = {item.name: item for item in fields(type(self))}
        values: Dict[str, Any] = {}
        for name, item in config_fields.items():
            if name in kwargs:
                values[name] = kwargs.pop(name)
            elif item.default_factory is not MISSING:
                values[name] = item.default_factory()  # type: ignore[misc]
            elif item.default is not MISSING:
                values[name] = item.default
            else:
                raise TypeError(f"缺少必需的 RAGConfig 字段: {name}")
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"未知 RAGConfig 字段: {unknown}")
        for name, value in values.items():
            setattr(self, name, value)
        self.__post_init__()

    # 初始化后时自动调用
    def __post_init__(self):
        """初始化后的处理"""
        # 路径统一，将所有路径转换为绝对路径
        self.data_path = resolve_project_path(self.data_path)
        self.upload_dir = resolve_project_path(self.upload_dir)
        self.knowledge_base_id = validate_knowledge_base_id(self.knowledge_base_id)
        self.database_url = str(self.database_url).strip()
        self.redis_url = str(self.redis_url).strip()
        self.milvus_host = str(self.milvus_host).strip()
        self.milvus_port = int(self.milvus_port)
        self.milvus_collection = str(self.milvus_collection).strip()
        # 类型转换，将从环境变量中获取的字符串转换为整数或浮点数
        self.top_k = int(self.top_k)
        self.retrieval_candidate_k = int(self.retrieval_candidate_k)
        self.rrf_k = int(self.rrf_k)
        self.retrieval_strategy = str(self.retrieval_strategy).strip() or "llamaindex_router"
        self.llamaindex_index_store_dir = resolve_project_path(self.llamaindex_index_store_dir)
        self.score_threshold = float(self.score_threshold)
        self.reranker_provider = str(self.reranker_provider).strip().lower()
        self.reranker_model = str(self.reranker_model).strip()
        self.reranker_endpoint = str(self.reranker_endpoint).strip()
        self.reranker_api_key = str(self.reranker_api_key).strip()
        self.reranker_top_n = int(self.reranker_top_n)
        self.auto_merge_ratio_threshold = float(self.auto_merge_ratio_threshold)
        self.context_token_budget = int(self.context_token_budget)
        self.neighbor_window = max(int(self.neighbor_window), 0)
        self.max_upload_bytes = int(self.max_upload_bytes)
        self.temperature = float(self.temperature)
        self.max_tokens = int(self.max_tokens)

    @classmethod
    def from_env(cls) -> 'RAGConfig': # 当前类  
        """
        从 RAG_* 环境变量创建配置对象
        Returns:
            已完成路径解析和类型转换的 RAGConfig
        """
        # 环境变量映射表，将配置类的属性名映射到环境变量名
        env_mapping = {
            "data_path": "RAG_DATA_PATH", # 数据路径
            "upload_dir": "RAG_UPLOAD_DIR", # 上传目录
            "knowledge_base_id": "RAG_KNOWLEDGE_BASE_ID", # 知识库ID
            "database_url": "RAG_DATABASE_URL", # 数据库URL
            "redis_url": "RAG_REDIS_URL", # Redis URL
            "milvus_host": "RAG_MILVUS_HOST", # Milvus 主机
            "milvus_port": "RAG_MILVUS_PORT", # Milvus 端口
            "milvus_collection": "RAG_MILVUS_COLLECTION", # Milvus 集合
            "embedding_model": "RAG_EMBEDDING_MODEL", # 嵌入模型
            "llm_model": "RAG_LLM_MODEL", # LLM 模型
            "top_k": "RAG_TOP_K", # 检索结果数量
            "retrieval_candidate_k": "RAG_RETRIEVAL_CANDIDATE_K", # 检索候选数量
            "rrf_k": "RAG_RRF_K", # RRF 策略参数
            "retrieval_strategy": "RAG_RETRIEVAL_STRATEGY", # 检索策略
            "llamaindex_index_store_dir": "RAG_LLAMAINDEX_INDEX_STORE_DIR", # LlamaIndex 索引存储目录
            "score_threshold": "RAG_SCORE_THRESHOLD", # 分数阈值
            "reranker_provider": "RAG_RERANKER_PROVIDER", # 重排序器提供程序
            "reranker_model": "RAG_RERANKER_MODEL", # 重排序器模型
            "reranker_endpoint": "RAG_RERANKER_ENDPOINT", # 重排序器端点
            "reranker_api_key": "RAG_RERANKER_API_KEY", # 重排序器 API密钥
            "reranker_top_n": "RAG_RERANKER_TOP_N", # 重排序器返回结果数量
            "auto_merge_ratio_threshold": "RAG_AUTO_MERGE_RATIO_THRESHOLD", # 自动合并阈值
            "context_token_budget": "RAG_CONTEXT_TOKEN_BUDGET", # 上下文令牌预算
            "neighbor_window": "RAG_NEIGHBOR_WINDOW", # 前后相邻节点扩展数量
            "max_upload_bytes": "RAG_MAX_UPLOAD_BYTES",  # 最大上传文件大小（字节）
            "temperature": "RAG_TEMPERATURE", # 温度参数
            "max_tokens": "RAG_MAX_TOKENS", # 最大令牌数
        }
        # 字典推导式，将环境变量中的值赋值给配置类的属性
        config_dict = {
            field_name: os.environ[env_name]
            for field_name, env_name in env_mapping.items()
            if env_name in os.environ
        }
        return cls(**config_dict) # 字典解包
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将配置对象转换为普通字典
        Returns:
            包含路径、模型、检索和生成配置的字典
        """
        return {
            'data_path': self.data_path,
            'upload_dir': self.upload_dir,
            'knowledge_base_id': self.knowledge_base_id,
            'database_url': self.database_url,
            'redis_url': self.redis_url,
            'milvus_host': self.milvus_host,
            'milvus_port': self.milvus_port,
            'milvus_collection': self.milvus_collection,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'retrieval_candidate_k': self.retrieval_candidate_k,
            'rrf_k': self.rrf_k,
            'retrieval_strategy': self.retrieval_strategy,
            'llamaindex_index_store_dir': self.llamaindex_index_store_dir,
            'score_threshold': self.score_threshold,
            'reranker_provider': self.reranker_provider,
            'reranker_model': self.reranker_model,
            'reranker_endpoint': self.reranker_endpoint,
            'reranker_api_key': self.reranker_api_key,
            'reranker_top_n': self.reranker_top_n,
            'auto_merge_ratio_threshold': self.auto_merge_ratio_threshold,
            'context_token_budget': self.context_token_budget,
            'neighbor_window': self.neighbor_window,
            'max_upload_bytes': self.max_upload_bytes,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
