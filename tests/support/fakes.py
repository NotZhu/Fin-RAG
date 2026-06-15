from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from finrag.core.node_schema import TextNode
from finrag.ingestion import DocumentRecord
from finrag.storage.protocols import SparseVector


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

    def list_public(self) -> List[dict]:
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

    def replace_document_nodes(self, document_id: str, nodes: List[TextNode]) -> None:
        self.delete_document(document_id)
        for node in nodes:
            self.nodes[node.node_id] = node
            if self.node_cache is not None:
                self.node_cache.set_node(node)

    def load_all_nodes(self) -> List[TextNode]:
        return sorted(
            self.nodes.values(),
            key=lambda node: (
                str((node.metadata or {}).get("document_id") or ""),
                int((node.metadata or {}).get("chunk_level", 0) or 0),
                int((node.metadata or {}).get("chunk_idx", 0) or 0),
                node.node_id,
            ),
        )

    def load_leaf_nodes(self) -> List[TextNode]:
        return [node for node in self.load_all_nodes() if int((node.metadata or {}).get("chunk_level", 0) or 0) == 3]

    def get_node(self, node_id: str) -> Optional[TextNode]:
        if self.node_cache is not None:
            cached = self.node_cache.get_node(node_id)
            if cached is not None:
                return cached
        node = self.nodes.get(node_id)
        if node is not None and self.node_cache is not None:
            self.node_cache.set_node(node)
        return node

    def delete_document(self, document_id: str) -> None:
        node_ids = [
            node_id
            for node_id, node in self.nodes.items()
            if str((node.metadata or {}).get("document_id") or "") == document_id
        ]
        for node_id in node_ids:
            self.nodes.pop(node_id, None)
        if self.node_cache is not None:
            self.node_cache.delete_document(document_id, node_ids)

    def clear(self) -> None:
        self.nodes.clear()

    def count_leaf_nodes(self, document_id: str) -> int:
        return sum(
            1
            for node in self.nodes.values()
            if str((node.metadata or {}).get("document_id") or "") == document_id
            and int((node.metadata or {}).get("chunk_level", 0) or 0) == 3
        )


class MemoryBM25StateStore:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url
        self.term_ids: Dict[str, int] = {}
        self.documents: Dict[str, tuple[str, int]] = {}
        self.term_counts: Dict[str, Dict[str, int]] = {}
        self.dfs: Dict[str, int] = {}

    def replace_document_chunks(self, document_id: str, chunk_token_counts: Dict[str, Dict[str, int]]) -> None:
        self.delete_document(document_id)
        for chunk_id, token_counts in chunk_token_counts.items():
            clean_counts = {str(term).strip().lower(): int(count) for term, count in token_counts.items() if term and int(count) > 0}
            self.documents[chunk_id] = (document_id, sum(clean_counts.values()))
            self.term_counts[chunk_id] = clean_counts
            for term in clean_counts:
                self._ensure_term(term)
        self._refresh_dfs()

    def delete_document(self, document_id: str) -> None:
        chunk_ids = [chunk_id for chunk_id, (doc_id, _) in self.documents.items() if doc_id == document_id]
        for chunk_id in chunk_ids:
            self.documents.pop(chunk_id, None)
            self.term_counts.pop(chunk_id, None)
        self._refresh_dfs()

    def clear(self) -> None:
        self.documents.clear()
        self.term_counts.clear()
        self.dfs = {term: 0 for term in self.term_ids}

    def build_query_sparse_vector(self, tokens: Iterable[str]) -> SparseVector:
        token_counts = self._token_counts(tokens)
        indices = [self.term_ids[term] for term in token_counts if term in self.term_ids]
        return SparseVector(indices=indices, values=[1.0 for _ in indices], token_count=sum(token_counts.values()))

    def build_document_sparse_vector(self, tokens: Iterable[str]) -> SparseVector:
        token_counts = self._token_counts(tokens)
        token_count = sum(token_counts.values())
        if not token_counts:
            return SparseVector(indices=[], values=[], token_count=0)
        total_documents = len(self.documents)
        avgdl = sum(length for _, length in self.documents.values()) / total_documents if total_documents else token_count
        indices: List[int] = []
        values: List[float] = []
        for term, term_frequency in token_counts.items():
            if term not in self.term_ids:
                continue
            document_frequency = self.dfs.get(term, 0)
            idf = math.log(1.0 + ((total_documents - document_frequency + 0.5) / (document_frequency + 0.5))) if total_documents else 0.0
            denominator = term_frequency + 1.5 * (1.0 - 0.75 + 0.75 * (token_count / (avgdl or token_count or 1)))
            weight = idf * ((term_frequency * 2.5) / denominator) if denominator else 0.0
            if weight:
                indices.append(self.term_ids[term])
                values.append(float(weight))
        return SparseVector(indices=indices, values=values, token_count=token_count)

    def _ensure_term(self, term: str) -> None:
        if term not in self.term_ids:
            self.term_ids[term] = len(self.term_ids) + 1

    def _refresh_dfs(self) -> None:
        self.dfs = {term: 0 for term in self.term_ids}
        for term in self.term_ids:
            self.dfs[term] = sum(1 for counts in self.term_counts.values() if term in counts)

    @staticmethod
    def _token_counts(tokens: Iterable[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for token in tokens:
            token = str(token).strip().lower()
            if token:
                counts[token] = counts.get(token, 0) + 1
        return counts


class MemoryIndexManifestStore:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url
        self.manifest: Optional[Dict[str, Any]] = None

    def save_manifest(self, manifest: Dict[str, Any]) -> None:
        self.manifest = dict(manifest)

    def load_manifest(self) -> Optional[Dict[str, Any]]:
        return dict(self.manifest) if self.manifest is not None else None
