"""FinRAG 持久化存储适配器"""

from .document_registry import PostgreSQLDocumentRegistry
from .knowledge_base_registry import (
    DuplicateKnowledgeBaseError,
    KnowledgeBaseArchivedError,
    KnowledgeBaseNotFoundError,
    KnowledgeBaseRecord,
    PostgreSQLKnowledgeBaseRegistry,
    ProtectedKnowledgeBaseError,
)
from .llama_docstore import PostgreSQLLlamaIndexDocumentStore
from .manifest_store import PostgreSQLIndexManifestStore

__all__ = [
    "DuplicateKnowledgeBaseError",
    "KnowledgeBaseArchivedError",
    "KnowledgeBaseNotFoundError",
    "KnowledgeBaseRecord",
    "PostgreSQLDocumentRegistry",
    "PostgreSQLIndexManifestStore",
    "PostgreSQLKnowledgeBaseRegistry",
    "PostgreSQLLlamaIndexDocumentStore",
    "ProtectedKnowledgeBaseError",
]
