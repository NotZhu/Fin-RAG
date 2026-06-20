import type { FormEvent } from "react";
import { FileText, MessageSquareText, PenLine, Trash2 } from "lucide-react";
import ReactMarkdown from "react-markdown";

import type { AskResponse, DocumentRecord } from "../api/client";
import { DocumentsPage } from "./DocumentsPage";
import { ErrorPanel } from "./ErrorPanel";
import { QuestionComposer } from "./QuestionComposer";
import { SourcesPane } from "./SourcesPane";

export type ActiveTab = "qa" | "docs";

type ChatPanelProps = {
  activeTab: ActiveTab;
  answerBusy: boolean;
  documents: DocumentRecord[];
  error: string;
  hasConversation: boolean;
  knowledgeBaseId: string;
  knowledgeBaseIsAvailable: boolean;
  knowledgeBaseLoadState: "loading" | "ready" | "error";
  knowledgeBaseUpdatedAt: string;
  question: string;
  response: AskResponse | null;
  streamedAnswer: string;
  submittedQuestion: string;
  submitDisabled: boolean;
  totalDocuments: number;
  totalChunks: number;
  rebuildBusy: boolean;
  warmupBusy: boolean;
  onAbort: () => void;
  onClearConversation: () => void;
  onNewChat: () => void;
  onQuestionChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onTabChange: (tab: ActiveTab) => void;
  onReindexDocument: (documentId: string) => void;
  onDeleteDocument: (documentId: string) => void;
  onCreateKnowledgeBase: (knowledgeBaseId: string) => void;
  onRebuildKnowledgeBase: () => void;
  onWarmupKnowledgeBase: () => void;
  reindexingDocId: string | null;
};

export function ChatPanel({
  activeTab,
  answerBusy,
  documents,
  error,
  hasConversation,
  knowledgeBaseId,
  knowledgeBaseIsAvailable,
  knowledgeBaseLoadState,
  knowledgeBaseUpdatedAt,
  question,
  response,
  streamedAnswer,
  submittedQuestion,
  submitDisabled,
  totalDocuments,
  totalChunks,
  rebuildBusy,
  warmupBusy,
  onAbort,
  onClearConversation,
  onNewChat,
  onQuestionChange,
  onSubmit,
  onTabChange,
  onReindexDocument,
  onDeleteDocument,
  onCreateKnowledgeBase,
  onRebuildKnowledgeBase,
  onWarmupKnowledgeBase,
  reindexingDocId,
}: ChatPanelProps) {
  const answer = streamedAnswer || response?.answer || "";

  return (
    <section
      className={`chat-panel${hasConversation ? "" : " is-empty"}`}
      aria-label="知识库问答"
    >
      <header className="chat-header">
        <nav className="tab-bar">
          <button
            className={`tab-item${activeTab === "qa" ? " is-active" : ""}`}
            type="button"
            onClick={() => onTabChange("qa")}
          >
            <MessageSquareText size={16} />
            <span>问答</span>
          </button>
          <button
            className={`tab-item${activeTab === "docs" ? " is-active" : ""}`}
            type="button"
            onClick={() => onTabChange("docs")}
          >
            <FileText size={16} />
            <span>文档</span>
          </button>
        </nav>
        <div className="chat-header-actions">
          {activeTab === "qa" && hasConversation && (
            <button
              className="new-chat-button"
              title="清空对话"
              type="button"
              onClick={onClearConversation}
            >
              <Trash2 size={16} />
              <span>清空对话</span>
            </button>
          )}
          {activeTab === "qa" && (
            <button
              className="new-chat-button"
              title="新聊天"
              type="button"
              onClick={onNewChat}
            >
              <PenLine size={16} />
              <span>新聊天</span>
            </button>
          )}
        </div>
      </header>

      {activeTab === "qa" ? (
        hasConversation ? (
          <>
            <div className="chat-scroll">
              {submittedQuestion ? (
                <div className="message-row user-message">
                  <div className="chat-bubble user-bubble">
                    {submittedQuestion}
                  </div>
                </div>
              ) : null}

              {answer || answerBusy ? (
                <div className="message-row assistant-message">
                  <span className="assistant-mark">
                    <MessageSquareText size={18} />
                  </span>
                  <div className="assistant-content">
                    <div className="chat-bubble assistant-bubble">
                      {answer ? (
                        <ReactMarkdown>{answer}</ReactMarkdown>
                      ) : (
                        "正在检索资料库..."
                      )}
                    </div>
                    <SourcesPane response={response} />
                  </div>
                </div>
              ) : null}

              {error ? <ErrorPanel message={error} /> : null}
            </div>

            <QuestionComposer
              answerBusy={answerBusy}
              question={question}
              submitDisabled={submitDisabled}
              onAbort={onAbort}
              onQuestionChange={onQuestionChange}
              onSubmit={onSubmit}
              floating={false}
            />
          </>
        ) : (
          <div className="chat-scroll">
            <div className="chat-empty-stage">
              <div className="chat-empty">
                <strong>文档就绪，随时提问</strong>
              </div>
              <QuestionComposer
                answerBusy={answerBusy}
                question={question}
                submitDisabled={submitDisabled}
                onAbort={onAbort}
                onQuestionChange={onQuestionChange}
                onSubmit={onSubmit}
                floating
              />
              {error ? <ErrorPanel message={error} /> : null}
            </div>
          </div>
        )
      ) : (
        <div className="chat-scroll">
          <DocumentsPage
            documents={documents}
            knowledgeBaseId={knowledgeBaseId}
            knowledgeBaseIsAvailable={knowledgeBaseIsAvailable}
            knowledgeBaseLoadState={knowledgeBaseLoadState}
            knowledgeBaseUpdatedAt={knowledgeBaseUpdatedAt}
            onCreateKnowledgeBase={onCreateKnowledgeBase}
            onReindexDocument={onReindexDocument}
            onDeleteDocument={onDeleteDocument}
            onRebuildKnowledgeBase={onRebuildKnowledgeBase}
            onWarmupKnowledgeBase={onWarmupKnowledgeBase}
            reindexingDocId={reindexingDocId}
            rebuildBusy={rebuildBusy}
            warmupBusy={warmupBusy}
            totalDocuments={totalDocuments}
            totalChunks={totalChunks}
          />
        </div>
      )}
    </section>
  );
}
