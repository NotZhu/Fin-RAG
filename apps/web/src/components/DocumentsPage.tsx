import { useEffect, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  Clock3,
  FileSearch,
  FileText,
  Plus,
  RefreshCw,
  Trash2,
} from "lucide-react";

import type { DocumentRecord } from "../api/client";
import { ConfirmDialog, type ConfirmAction } from "./ConfirmDialog";
import { CreateKnowledgeBaseDialog } from "./CreateKnowledgeBaseDialog";
import { displayName, formatUploadTime } from "./utils";

type DocumentsPageProps = {
  documents: DocumentRecord[];
  knowledgeBaseId: string;
  knowledgeBaseIsAvailable: boolean;
  knowledgeBaseLoadState: "loading" | "ready" | "error";
  knowledgeBaseUpdatedAt: string;
  onCreateKnowledgeBase: (knowledgeBaseId: string) => Promise<void> | void;
  onReindexDocument: (documentId: string) => void;
  onDeleteDocument: (documentId: string) => void;
  onWarmupKnowledgeBase: () => void;
  reindexingDocId: string | null;
  warmupBusy: boolean;
  totalDocuments: number;
  totalChunks: number;
};

const PAGE_SIZE = 10;

export function DocumentsPage({
  documents,
  knowledgeBaseId,
  knowledgeBaseIsAvailable,
  knowledgeBaseLoadState,
  knowledgeBaseUpdatedAt,
  onCreateKnowledgeBase,
  onReindexDocument,
  onDeleteDocument,
  onWarmupKnowledgeBase,
  reindexingDocId,
  warmupBusy,
  totalDocuments,
  totalChunks,
}: DocumentsPageProps) {
  const [page, setPage] = useState(1);
  const [pendingAction, setPendingAction] = useState<ConfirmAction | null>(
    null,
  );
  const [isCreateDialogOpen, setIsCreateDialogOpen] = useState(false);
  const totalPages = Math.max(1, Math.ceil(documents.length / PAGE_SIZE));

  useEffect(() => {
    setPage(1);
  }, [documents.length]);

  const safePage = Math.min(page, totalPages);
  const start = (safePage - 1) * PAGE_SIZE;
  const pageDocs = documents.slice(start, start + PAGE_SIZE);
  const knowledgeBaseName = knowledgeBaseIsAvailable
    ? displayName(knowledgeBaseId)
    : knowledgeBaseLoadState === "loading"
      ? "加载中"
      : knowledgeBaseLoadState === "error"
        ? "加载失败"
        : "暂无知识库";
  const updatedAtText = formatUploadTime(knowledgeBaseUpdatedAt);

  return (
    <div className="docs-page">
      <div className="docs-toolbar">
        <div className="kb-stats">
          <span className="kb-stat-kb-name">
            <span className="kb-stat-label">当前知识库：</span>
            <span className="kb-stat-value">{knowledgeBaseName}</span>
          </span>
          <span className="kb-stat-divider">|</span>
          <div className="kb-stat-item">
            <FileText size={16} />
            <span className="kb-stat-label">文档</span>
            <span className="kb-stat-value">{totalDocuments}</span>
            <span className="kb-stat-label">份</span>
          </div>
          <span className="kb-stat-divider">|</span>
          <div className="kb-stat-item">
            <FileSearch size={16} />
            <span className="kb-stat-label">总分块</span>
            <span className="kb-stat-value">{totalChunks}</span>
          </div>
          <span className="kb-stat-divider">|</span>
          <div className="kb-stat-item">
            <Clock3 size={16} />
            <span className="kb-stat-label">更新</span>
            {updatedAtText ? (
              <span className="kb-stat-value">{updatedAtText}</span>
            ) : null}
          </div>
          <button
            aria-label="刷新知识库"
            className="kb-warmup-button"
            disabled={
              warmupBusy ||
              !knowledgeBaseIsAvailable ||
              knowledgeBaseLoadState !== "ready"
            }
            title="刷新知识库"
            type="button"
            onClick={onWarmupKnowledgeBase}
          >
            <RefreshCw className={warmupBusy ? "spin" : ""} size={16} />
          </button>
        </div>
        <button
          className="docs-create-button"
          title="新建知识库"
          type="button"
          onClick={() => setIsCreateDialogOpen(true)}
        >
          <Plus size={16} />
          <span>新建知识库</span>
        </button>
      </div>
      {documents.length ? (
        <>
          <div className="docs-grid">
            {pageDocs.map((doc) => (
              <div className="doc-card" key={doc.document_id}>
                <div className="doc-icon">
                  <FileText size={20} />
                </div>
                <div className="doc-info">
                  <strong className="doc-name">{doc.filename}</strong>
                  <span className="doc-meta">
                    {doc.status === "indexed"
                      ? `${doc.chunk_count} 个分块`
                      : doc.status === "parsing"
                        ? "解析中..."
                        : doc.status === "failed"
                          ? "索引失败"
                          : "已上传"}
                    {doc.upload_time
                      ? ` · ${formatUploadTime(doc.upload_time)}`
                      : ""}
                    {doc.status === "failed" && doc.last_error ? " · " : ""}
                    {doc.status === "failed" && doc.last_error ? (
                      <span className="doc-meta-error">{doc.last_error}</span>
                    ) : null}
                  </span>
                </div>
                <div className="doc-actions">
                  <button
                    className="doc-action-btn"
                    disabled={reindexingDocId === doc.document_id}
                    title="重新索引"
                    type="button"
                    onClick={() =>
                      setPendingAction({
                        type: "reindex",
                        documentId: doc.document_id,
                        filename: doc.filename,
                      })
                    }
                  >
                    <RefreshCw
                      className={
                        reindexingDocId === doc.document_id ? "spin" : ""
                      }
                      size={14}
                    />
                  </button>
                  <button
                    className="doc-action-btn delete"
                    title="删除"
                    type="button"
                    onClick={() =>
                      setPendingAction({
                        type: "delete",
                        documentId: doc.document_id,
                        filename: doc.filename,
                      })
                    }
                  >
                    <Trash2 size={14} />
                  </button>
                  <span className={`doc-badge status-${doc.status}`}>
                    {doc.status === "indexed"
                      ? "已索引"
                      : doc.status === "parsing"
                        ? "解析中"
                        : doc.status === "failed"
                          ? "失败"
                          : "已上传"}
                  </span>
                </div>
              </div>
            ))}
          </div>
          <div className="pagination">
            <span className="pagination-info">
              共 {documents.length} 条，第 {safePage}/{totalPages} 页
            </span>
            <div className="pagination-buttons">
              <button
                className="pagination-btn"
                disabled={safePage <= 1}
                type="button"
                onClick={() => setPage(safePage - 1)}
              >
                <ChevronLeft size={16} />
              </button>
              {Array.from({ length: totalPages }, (_, i) => i + 1).map((p) => (
                <button
                  className={`pagination-btn${p === safePage ? " is-active" : ""}`}
                  key={p}
                  type="button"
                  onClick={() => setPage(p)}
                >
                  {p}
                </button>
              ))}
              <button
                className="pagination-btn"
                disabled={safePage >= totalPages}
                type="button"
                onClick={() => setPage(safePage + 1)}
              >
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        </>
      ) : (
        <div className="docs-empty">
          <FileSearch size={32} />
          <strong>暂无文档</strong>
          <span>上传文档后将在此处显示</span>
        </div>
      )}
      {pendingAction ? (
        <ConfirmDialog
          action={pendingAction}
          onConfirm={() => {
            if (pendingAction.type === "reindex") {
              onReindexDocument(pendingAction.documentId);
            } else {
              onDeleteDocument(pendingAction.documentId);
            }
            setPendingAction(null);
          }}
          onCancel={() => setPendingAction(null)}
        />
      ) : null}
      {isCreateDialogOpen ? (
        <CreateKnowledgeBaseDialog
          onCancel={() => setIsCreateDialogOpen(false)}
          onCreate={onCreateKnowledgeBase}
        />
      ) : null}
    </div>
  );
}
