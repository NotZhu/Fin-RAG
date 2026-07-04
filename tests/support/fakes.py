from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from finrag.core.node_schema import TextNode
from finrag.ingestion import DocumentRecord
from finrag.storage.knowledge_base_registry import (
    DuplicateKnowledgeBaseError,
    KnowledgeBaseNotFoundError,
    KnowledgeBaseRecord,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryDocumentRegistry:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url
        self.records: Dict[str, DocumentRecord] = {}

    def save(self) -> None:
        pass

    def list(self) -> List[dict]:
        return [record.to_dict() for record in sorted(self.records.values(), key=lambda item: item.upload_time)]

    def list_public(self, knowledge_base_id: str | None = None) -> List[dict]:
        public_fields = {
            "document_id",
            "filename",
            "file_type",
            "knowledge_base_id",
            "status",
            "chunk_count",
            "upload_time",
            "last_error",
        }
        return [
            {key: value for key, value in record.to_dict().items() if key in public_fields}
            for record in sorted(self.records.values(), key=lambda item: item.upload_time)
            if record.status != "deleted"
            and (knowledge_base_id is None or record.knowledge_base_id == knowledge_base_id)
        ]

    def get(self, document_id: str) -> DocumentRecord:
        return self.records[document_id]

    def find_by_hash(self, content_hash: str, knowledge_base_id: str) -> Optional[DocumentRecord]:
        for record in self.records.values():
            if record.content_hash == content_hash and record.knowledge_base_id == knowledge_base_id and record.status != "deleted":
                return record
        return None

    def upsert_uploaded(
        self,
        *,
        source_path: Path,
        filename: str,
        file_type: str,
        content_hash: str,
        knowledge_base_id: str,
    ) -> DocumentRecord:
        existing = self.find_by_hash(content_hash, knowledge_base_id)
        if existing is not None:
            return existing
        record = DocumentRecord(
            document_id=uuid.uuid4().hex,
            source_path=str(source_path),
            filename=filename,
            file_type=file_type,
            content_hash=content_hash,
            knowledge_base_id=knowledge_base_id,
            status="uploaded",
            upload_time=_utc_now_iso(),
        )
        self.records[record.document_id] = record
        return record

    def update_source_path(self, document_id: str, source_path: str) -> None:
        self.records[document_id].source_path = source_path

    def mark_parsing(self, document_id: str) -> None:
        self.records[document_id].status = "parsing"
        self.records[document_id].last_error = None

    def mark_indexed(self, document_id: str, chunk_count: int) -> None:
        self.records[document_id].status = "indexed"
        self.records[document_id].chunk_count = int(chunk_count)
        self.records[document_id].last_error = None

    def mark_failed(self, document_id: str, error: str) -> None:
        self.records[document_id].status = "failed"
        self.records[document_id].last_error = error

    def mark_deleted(self, document_id: str) -> dict:
        self.records[document_id].status = "deleted"
        return self.records[document_id].to_dict()


class MemoryNodeStore:
    def __init__(self, database_url: str | None = None, node_cache: Any = None):
        self.database_url = database_url
        self.node_cache = node_cache
        self.nodes: Dict[str, TextNode] = {}

    def replace_document_nodes(self, document_id: str, nodes: List[TextNode], knowledge_base_id: str) -> None:
        self.delete_document(document_id, knowledge_base_id)
        for node in nodes:
            self.nodes[node.node_id] = node
            if self.node_cache is not None:
                self.node_cache.set_node(node)

    def load_all_nodes(self, knowledge_base_id: str) -> List[TextNode]:
        return sorted(
            [
                node
                for node in self.nodes.values()
                if str((node.metadata or {}).get("knowledge_base_id") or "") == knowledge_base_id
            ],
            key=lambda node: (
                str((node.metadata or {}).get("document_id") or ""),
                int((node.metadata or {}).get("chunk_level", 0) or 0),
                int((node.metadata or {}).get("chunk_idx", 0) or 0),
                node.node_id,
            ),
        )

    def load_leaf_nodes(self, knowledge_base_id: str) -> List[TextNode]:
        return [node for node in self.load_all_nodes(knowledge_base_id) if int((node.metadata or {}).get("chunk_level", 0) or 0) == 3]

    def get_node(self, node_id: str) -> Optional[TextNode]:
        if self.node_cache is not None:
            cached = self.node_cache.get_node(node_id)
            if cached is not None:
                return cached
        node = self.nodes.get(node_id)
        if node is not None and self.node_cache is not None:
            self.node_cache.set_node(node)
        return node

    def delete_document(self, document_id: str, knowledge_base_id: str) -> None:
        node_ids = [
            node_id
            for node_id, node in self.nodes.items()
            if str((node.metadata or {}).get("document_id") or "") == document_id
            and str((node.metadata or {}).get("knowledge_base_id") or "") == knowledge_base_id
        ]
        for node_id in node_ids:
            self.nodes.pop(node_id, None)
        if self.node_cache is not None:
            self.node_cache.delete_document(document_id, node_ids)

    def clear(self) -> None:
        self.nodes.clear()

    def delete_knowledge_base(self, knowledge_base_id: str) -> None:
        for node_id in [
            node_id
            for node_id, node in self.nodes.items()
            if str((node.metadata or {}).get("knowledge_base_id") or "") == knowledge_base_id
        ]:
            self.nodes.pop(node_id, None)

    def count_leaf_nodes(self, document_id: str) -> int:
        return sum(
            1
            for node in self.nodes.values()
            if str((node.metadata or {}).get("document_id") or "") == document_id
            and int((node.metadata or {}).get("chunk_level", 0) or 0) == 3
        )


class MemoryIndexManifestStore:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url
        self.manifests: Dict[str, Dict[str, Any]] = {}

    def save_manifest(self, manifest: Dict[str, Any], knowledge_base_id: str) -> None:
        self.manifests[knowledge_base_id] = dict(manifest)

    def load_manifest(self, knowledge_base_id: str) -> Optional[Dict[str, Any]]:
        manifest = self.manifests.get(knowledge_base_id)
        return dict(manifest) if manifest is not None else None

    def delete_manifest(self, knowledge_base_id: str) -> None:
        self.manifests.pop(knowledge_base_id, None)


class MemoryKnowledgeBaseRegistry:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url
        self.records: Dict[str, KnowledgeBaseRecord] = {}

    def ensure_default(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        if knowledge_base_id in self.records:
            record = self.records[knowledge_base_id]
            if record.status == "deleted":
                restored = KnowledgeBaseRecord(
                    knowledge_base_id=record.knowledge_base_id,
                    created_at=record.created_at,
                    updated_at=_utc_now_iso(),
                    status="active",
                    archived_at=None,
                    deleted_at=None,
                )
                self.records[knowledge_base_id] = restored
                return restored
            return record
        now = _utc_now_iso()
        record = KnowledgeBaseRecord(
            knowledge_base_id=knowledge_base_id,
            created_at=now,
            updated_at=now,
        )
        self.records[knowledge_base_id] = record
        return record

    def create(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        if knowledge_base_id in self.records:
            raise DuplicateKnowledgeBaseError(knowledge_base_id)
        now = _utc_now_iso()
        record = KnowledgeBaseRecord(
            knowledge_base_id=knowledge_base_id,
            created_at=now,
            updated_at=now,
        )
        self.records[knowledge_base_id] = record
        return record

    def list(self, *, include_deleted: bool = False) -> List[KnowledgeBaseRecord]:
        records = self.records.values() if include_deleted else [
            record for record in self.records.values() if record.status != "deleted"
        ]
        return sorted(records, key=lambda item: (item.created_at, item.knowledge_base_id))

    def get(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        record = self.get_optional(knowledge_base_id)
        if record is None:
            raise KnowledgeBaseNotFoundError(knowledge_base_id)
        return record

    def get_optional(self, knowledge_base_id: str, *, include_deleted: bool = False) -> Optional[KnowledgeBaseRecord]:
        record = self.records.get(knowledge_base_id)
        if record is not None and record.status == "deleted" and not include_deleted:
            return None
        return record

    def archive(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        record = self.get(knowledge_base_id)
        now = _utc_now_iso()
        archived = KnowledgeBaseRecord(
            knowledge_base_id=record.knowledge_base_id,
            created_at=record.created_at,
            updated_at=now,
            status="archived",
            archived_at=now,
            deleted_at=None,
        )
        self.records[knowledge_base_id] = archived
        return archived

    def restore(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        record = self.get_optional(knowledge_base_id, include_deleted=True)
        if record is None or record.status == "deleted":
            raise KnowledgeBaseNotFoundError(knowledge_base_id)
        restored = KnowledgeBaseRecord(
            knowledge_base_id=record.knowledge_base_id,
            created_at=record.created_at,
            updated_at=_utc_now_iso(),
            status="active",
            archived_at=None,
            deleted_at=None,
        )
        self.records[knowledge_base_id] = restored
        return restored

    def touch(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        record = self.get(knowledge_base_id)
        touched = KnowledgeBaseRecord(
            knowledge_base_id=record.knowledge_base_id,
            created_at=record.created_at,
            updated_at=_utc_now_iso(),
            status=record.status,
            archived_at=record.archived_at,
            deleted_at=record.deleted_at,
        )
        self.records[knowledge_base_id] = touched
        return touched

    def mark_deleted(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        record = self.get_optional(knowledge_base_id, include_deleted=True)
        if record is None:
            raise KnowledgeBaseNotFoundError(knowledge_base_id)
        now = _utc_now_iso()
        deleted = KnowledgeBaseRecord(
            knowledge_base_id=record.knowledge_base_id,
            created_at=record.created_at,
            updated_at=now,
            status="deleted",
            archived_at=record.archived_at,
            deleted_at=now,
        )
        self.records[knowledge_base_id] = deleted
        return deleted
