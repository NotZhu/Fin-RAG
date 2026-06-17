import { FormEvent, useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  Check,
  ChevronLeft,
  ChevronRight,
  CloudUpload,
  Database,
  FileSearch,
  FileText,
  GitMerge,
  Loader2,
  MessageSquareText,
  PanelLeftClose,
  PanelLeftOpen,
  PenLine,
  RefreshCw,
  Route,
  Search,
  Send,
  Square,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import ReactMarkdown from "react-markdown";

import {
  ApiError,
  AskResponse,
  AskStreamEvent,
  DocumentRecord,
  PipelineStep,
  askQuestionStream,
  deleteDocument as deleteDocumentApi,
  getReady,
  listDocuments,
  reindexDocument as reindexDocumentApi,
  uploadDocument,
} from "../api/client";
import "../styles/app.css";

function errorMessage(error: unknown, fallback: string) {
  const apiError = error as ApiError;
  if (apiError?.message) {
    return apiError.request_id
      ? `${apiError.message} Request ID: ${apiError.request_id}`
      : apiError.message;
  }
  return fallback;
}

function formatRetrievalScore(score: number | null) {
  if (score === null || Number.isNaN(score)) {
    return "检索分数：-";
  }
  return `检索分数：${score.toFixed(2)}`;
}

function isPipelineStep(value: unknown): value is PipelineStep {
  return Boolean(
    value &&
    typeof value === "object" &&
    typeof (value as PipelineStep).id === "string" &&
    typeof (value as PipelineStep).label === "string" &&
    typeof (value as PipelineStep).order === "number",
  );
}

function isRetrievedSource(
  value: unknown,
): value is AskResponse["sources"][number] {
  return Boolean(
    value &&
    typeof value === "object" &&
    "source_id" in value &&
    "filename" in value,
  );
}

function isAskResponse(value: unknown): value is AskResponse {
  return Boolean(
    value &&
    typeof value === "object" &&
    "answer" in value &&
    "sources" in value,
  );
}

function upsertStep(steps: PipelineStep[], step: PipelineStep) {
  const next = steps.some((item) => item.id === step.id)
    ? steps.map((item) => (item.id === step.id ? step : item))
    : [...steps, step];
  return next.sort((a, b) => a.order - b.order);
}

function App() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [activeTab, setActiveTab] = useState<"qa" | "docs">("qa");
  const [question, setQuestion] = useState("");
  const [submittedQuestion, setSubmittedQuestion] = useState("");
  const [knowledgeBaseId, setKnowledgeBaseId] = useState("kb-finance");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [answerBusy, setAnswerBusy] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [reindexingDocId, setReindexingDocId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [streamedAnswer, setStreamedAnswer] = useState("");
  const [pipelineSteps, setPipelineSteps] = useState<PipelineStep[]>([]);
  const [timingsMs, setTimingsMs] = useState<Record<string, number>>({});
  const [totalDocuments, setTotalDocuments] = useState(0);
  const [totalChunks, setTotalChunks] = useState(0);
  const abortControllerRef = useRef<AbortController | null>(null);

  function handleClearConversation() {
    setSubmittedQuestion("");
    setResponse(null);
    setStreamedAnswer("");
    setPipelineSteps([]);
    setError("");
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setAnswerBusy(false);
  }

  async function reindexDocument(documentId: string) {
    setReindexingDocId(documentId);
    setError("");
    try {
      await reindexDocumentApi(documentId);
      await refreshDocuments();
    } catch (caught) {
      setError(errorMessage(caught, "重新索引失败"));
      await refreshDocuments();
    } finally {
      setReindexingDocId(null);
    }
  }

  async function deleteDocument(documentId: string) {
    setUploadBusy(true);
    setError("");
    try {
      await deleteDocumentApi(documentId);
      await refreshDocuments();
    } catch (caught) {
      setError(errorMessage(caught, "删除文档失败"));
    } finally {
      setUploadBusy(false);
    }
  }

  async function refreshDocuments() {
    const payload = await listDocuments();
    setDocuments(payload.documents.filter((doc) => doc.status !== "deleted"));
  }

  async function refreshReady() {
    setError("");
    try {
      const data = await getReady();
      setTotalDocuments(data.total_documents);
      setTotalChunks(data.total_chunks);
      await refreshDocuments();
    } catch (caught) {
      setError(errorMessage(caught, "无法连接后端服务。"));
    }
  }

  async function handleUpload() {
    if (!selectedFile) {
      setError("请选择要上传的金融文档。");
      return;
    }
    setUploadBusy(true);
    setError("");
    try {
      await uploadDocument(selectedFile, knowledgeBaseId);
      setSelectedFile(null);
      await refreshReady();
    } catch (caught) {
      setError(errorMessage(caught, "上传失败，请检查文件格式或后端服务。"));
      await refreshDocuments();
    } finally {
      setUploadBusy(false);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) {
      setError("请输入问题。");
      return;
    }

    setAnswerBusy(true);
    setError("");
    setResponse(null);
    setStreamedAnswer("");
    setPipelineSteps([]);
    setSubmittedQuestion(trimmedQuestion);
    setQuestion("");
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    try {
      const result = await askQuestionStream(
        trimmedQuestion,
        true,
        knowledgeBaseId,
        {
          signal: abortController.signal,
          onEvent(streamEvent: AskStreamEvent) {
            const eventData = streamEvent.data;
            if (
              streamEvent.type === "pipeline_step" &&
              isPipelineStep(eventData)
            ) {
              setPipelineSteps((current) => upsertStep(current, eventData));
            }
            if (streamEvent.type === "token") {
              setStreamedAnswer(
                (current) => current + String(streamEvent.data.text ?? ""),
              );
            }
            const source = "source" in eventData ? eventData.source : undefined;
            if (streamEvent.type === "source" && isRetrievedSource(source)) {
              setResponse((current) => {
                const base =
                  current ??
                  ({
                    question: trimmedQuestion,
                    route_type: "",
                    retrieval_strategy: "llamaindex_router",
                    answer: "",
                    sources: [],
                  } satisfies AskResponse);
                return { ...base, sources: [...base.sources, source] };
              });
            }
            if (
              streamEvent.type === "done" &&
              isAskResponse(streamEvent.data.response)
            ) {
              const doneResponse = streamEvent.data.response;
              setResponse(doneResponse);
              if (doneResponse.trace?.pipeline_steps?.length) {
                setPipelineSteps(doneResponse.trace.pipeline_steps);
              }
              if (doneResponse.trace?.timings_ms) {
                setTimingsMs(doneResponse.trace.timings_ms);
              }
            }
          },
        },
      );
      setResponse(result.payload);
      setStreamedAnswer(result.payload.answer);
      if (result.payload.trace?.pipeline_steps?.length) {
        setPipelineSteps(result.payload.trace.pipeline_steps);
      }
      if (result.payload.trace?.timings_ms) {
        setTimingsMs(result.payload.trace.timings_ms);
      }
    } catch (caught) {
      if ((caught as Error)?.name === "AbortError") {
        setError("已中断当前回答。");
      } else {
        setError(
          errorMessage(caught, "问答请求失败，请确认 API 服务仍在运行。"),
        );
      }
    } finally {
      setAnswerBusy(false);
      abortControllerRef.current = null;
    }
  }

  function handleAbortAnswer() {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setAnswerBusy(false);
  }

  useEffect(() => {
    void refreshReady();
  }, []);

  useEffect(() => {
    const hasParsing = documents.some((doc) => doc.status === "parsing");
    if (!hasParsing) return;
    const interval = setInterval(() => {
      void refreshDocuments();
    }, 3000);
    return () => clearInterval(interval);
  }, [documents]);

  return (
    <main className="app-shell">
      <div
        className={`app-layout${isSidebarCollapsed ? " sidebar-collapsed" : ""}`}
      >
        <aside
          className={`left-rail${isSidebarCollapsed ? " is-collapsed" : ""}`}
          aria-label="知识库工具栏"
        >
          <SidebarHeader
            collapsed={isSidebarCollapsed}
            onToggle={() => setIsSidebarCollapsed((current) => !current)}
          />
          <UploadPanel
            collapsed={isSidebarCollapsed}
            knowledgeBaseId={knowledgeBaseId}
            selectedFile={selectedFile}
            uploadBusy={uploadBusy}
            uploadDisabled={answerBusy}
            onKnowledgeBaseIdChange={setKnowledgeBaseId}
            onSelectedFileChange={setSelectedFile}
            onUpload={handleUpload}
          />
          <RetrievalTimeline
            collapsed={isSidebarCollapsed}
            steps={pipelineSteps}
            timingsMs={timingsMs}
          />
        </aside>
        <ChatPanel
          activeTab={activeTab}
          answerBusy={answerBusy}
          documents={documents}
          error={error}
          hasConversation={Boolean(
            submittedQuestion || streamedAnswer || response?.answer,
          )}
          question={question}
          response={response}
          streamedAnswer={streamedAnswer}
          submittedQuestion={submittedQuestion}
          submitDisabled={uploadBusy}
          totalDocuments={totalDocuments}
          totalChunks={totalChunks}
          onAbort={handleAbortAnswer}
          onClearConversation={handleClearConversation}
          onNewChat={handleClearConversation}
          onQuestionChange={setQuestion}
          onSubmit={handleSubmit}
          onTabChange={setActiveTab}
          onReindexDocument={reindexDocument}
          onDeleteDocument={deleteDocument}
          reindexingDocId={reindexingDocId}
        />
      </div>
    </main>
  );
}

function SidebarHeader({
  collapsed,
  onToggle,
}: {
  collapsed: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="rail-header">
      {!collapsed && <span className="rail-title">知识库问答</span>}
      <div className="rail-actions">
        <button
          aria-label={collapsed ? "展开侧栏" : "收起侧栏"}
          className="rail-toggle"
          title={collapsed ? "展开侧栏" : "收起侧栏"}
          type="button"
          onClick={onToggle}
        >
          {collapsed ? (
            <PanelLeftOpen size={20} />
          ) : (
            <PanelLeftClose size={20} />
          )}
        </button>
      </div>
    </div>
  );
}

function formatFileSize(bytes: number) {
  return `${(bytes / 1024).toFixed(1)} KB`;
}

function formatUploadTime(iso: string | undefined) {
  if (!iso) return "";
  const date = new Date(iso);
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  const h = String(date.getHours()).padStart(2, "0");
  const min = String(date.getMinutes()).padStart(2, "0");
  return `${y}-${m}-${d} ${h}:${min}`;
}

function splitFilename(filename: string) {
  const extensionStart = filename.lastIndexOf(".");
  if (extensionStart <= 0 || extensionStart === filename.length - 1) {
    return { stem: filename, extension: "" };
  }
  return {
    stem: filename.slice(0, extensionStart),
    extension: filename.slice(extensionStart),
  };
}

type UploadPanelProps = {
  collapsed: boolean;
  knowledgeBaseId: string;
  selectedFile: File | null;
  uploadBusy: boolean;
  uploadDisabled: boolean;
  onKnowledgeBaseIdChange: (value: string) => void;
  onSelectedFileChange: (value: File | null) => void;
  onUpload: () => void;
};

function UploadPanel({
  collapsed,
  knowledgeBaseId,
  selectedFile,
  uploadBusy,
  uploadDisabled,
  onKnowledgeBaseIdChange,
  onSelectedFileChange,
  onUpload,
}: UploadPanelProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const intakeDisabled = uploadBusy || uploadDisabled;
  const selectedFilename = selectedFile
    ? splitFilename(selectedFile.name)
    : null;

  function handleSelectedFile(file: File | null) {
    onSelectedFileChange(file);
    setIsConfirmOpen(Boolean(file));
  }

  function handleCancelUpload() {
    onSelectedFileChange(null);
    setIsConfirmOpen(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  function handleDragOver(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    if (intakeDisabled) {
      return;
    }
    setIsDragOver(true);
  }

  function handleDragLeave(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragOver(false);
  }

  function handleDrop(event: React.DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragOver(false);
    if (intakeDisabled) {
      return;
    }
    handleSelectedFile(event.dataTransfer.files?.[0] ?? null);
  }

  useEffect(() => {
    if (!selectedFile) {
      setIsConfirmOpen(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }, [selectedFile]);

  return (
    <section className="rail-card upload-panel" aria-label="文件上传">
      <div className="rail-item rail-label" title="文件上传">
        <CloudUpload size={20} />
        <span>文件上传</span>
      </div>
      <div
        className={`drop-zone${isDragOver ? " is-drag-over" : ""}${selectedFile ? " has-file" : ""}`}
        role="button"
        tabIndex={collapsed || intakeDisabled ? -1 : 0}
        aria-disabled={intakeDisabled}
        onClick={() => {
          if (!intakeDisabled) {
            fileInputRef.current?.click();
          }
        }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onKeyDown={(event) => {
          if (!intakeDisabled && (event.key === "Enter" || event.key === " ")) {
            fileInputRef.current?.click();
          }
        }}
      >
        <FileText size={18} />
        <strong>{selectedFile ? selectedFile.name : "点击或拖入文件"}</strong>
        {selectedFile ? <span>{formatFileSize(selectedFile.size)}</span> : null}
      </div>
      <input
        ref={fileInputRef}
        className="visually-hidden"
        type="file"
        accept=".pdf,.docx,.md,.txt"
        aria-label="上传文件"
        onChange={(event) =>
          handleSelectedFile(event.target.files?.[0] ?? null)
        }
      />
      {selectedFile && isConfirmOpen ? (
        <div className="upload-confirm-backdrop">
          <div
            aria-labelledby="upload-confirm-title"
            aria-modal="true"
            className="upload-confirm-dialog"
            role="dialog"
          >
            <button
              aria-label="取消"
              className="icon-button upload-close-button"
              title="取消"
              type="button"
              onClick={handleCancelUpload}
              disabled={uploadBusy}
            >
              <X size={16} />
            </button>
            <div className="upload-confirm-header">
              <FileText size={22} />
              <h3 id="upload-confirm-title">确认索引文档</h3>
            </div>
            <div className="upload-file-summary">
              <label className="upload-summary-row upload-knowledge-row">
                <span>知识库</span>
                <input
                  aria-label="知识库"
                  disabled={uploadBusy}
                  value={knowledgeBaseId}
                  onChange={(event) =>
                    onKnowledgeBaseIdChange(event.target.value)
                  }
                />
              </label>
              <div className="upload-summary-row">
                <span>文件</span>
                <strong
                  aria-label={selectedFile.name}
                  className="upload-filename"
                  title={selectedFile.name}
                >
                  <span className="upload-filename-stem">
                    {selectedFilename?.stem}
                  </span>
                  {selectedFilename?.extension ? (
                    <span className="upload-filename-extension">
                      {selectedFilename.extension}
                    </span>
                  ) : null}
                </strong>
              </div>
              <div className="upload-summary-row">
                <span>大小</span>
                <strong>{formatFileSize(selectedFile.size)}</strong>
              </div>
            </div>
            <div className="modal-actions">
              <button
                aria-label="索引"
                className="primary-button modal-action-button"
                title="索引"
                type="button"
                onClick={() => {
                  setIsConfirmOpen(false);
                  void onUpload();
                }}
                disabled={uploadBusy || uploadDisabled}
              >
                {uploadBusy ? (
                  <Loader2 className="spin" size={17} />
                ) : (
                  <Upload size={17} />
                )}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}

type RetrievalTimelineProps = {
  collapsed: boolean;
  steps: PipelineStep[];
  timingsMs: Record<string, number>;
};

const TIMINGS_LABELS: Record<string, string> = {
  analysis: "分析",
  total: "总计",
};

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms.toFixed(1)}ms`;
}

function RetrievalTimeline({
  collapsed,
  steps,
  timingsMs,
}: RetrievalTimelineProps) {
  const hasTimings = Object.keys(timingsMs).length > 0;
  return (
    <section className="rail-card timeline-panel" aria-label="实时检索链路">
      <div className="rail-item rail-label">
        <Activity size={20} />
        <span>实时检索链路</span>
      </div>
      {!collapsed && (
        <div className="timeline-inner">
          {steps.length ? (
            <>
              <ol className="timeline-list">
                {steps.map((step, index) => (
                  <li
                    className={`timeline-item step-${step.status}`}
                    key={step.id}
                  >
                    <div className="step-number">{index + 1}</div>
                    <div className="step-body">
                      <div className="step-title">
                        <PipelineIcon step={step} />
                        <strong>{step.label}</strong>
                      </div>
                      <span>{step.detail}</span>
                    </div>
                    <div className="step-status">
                      {step.status === "running" ? (
                        <Loader2 className="spin" size={16} />
                      ) : step.duration_ms != null ? (
                        <span className="step-duration">
                          {formatMs(step.duration_ms)}
                        </span>
                      ) : (
                        <Check size={15} />
                      )}
                    </div>
                  </li>
                ))}
              </ol>
              {hasTimings && (
                <div className="timings-section">
                  {Object.entries(timingsMs).map(([key, value]) => (
                    <div className="timings-row" key={key}>
                      <span className="timings-label">
                        {TIMINGS_LABELS[key] ?? key}
                      </span>
                      <span className="timings-value">{formatMs(value)}</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <div className="timeline-empty" aria-hidden="true">
              <Search size={24} />
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function PipelineIcon({ step }: { step: PipelineStep }) {
  if (step.id.includes("router")) return <Route size={16} />;
  if (step.id.includes("hybrid") || step.id.includes("search"))
    return <Database size={16} />;
  if (step.id.includes("merge")) return <GitMerge size={16} />;
  if (step.id.includes("evidence")) return <FileText size={16} />;
  if (step.id.includes("answer")) return <MessageSquareText size={16} />;
  return <Search size={16} />;
}

type ChatPanelProps = {
  activeTab: "qa" | "docs";
  answerBusy: boolean;
  documents: DocumentRecord[];
  error: string;
  hasConversation: boolean;
  question: string;
  response: AskResponse | null;
  streamedAnswer: string;
  submittedQuestion: string;
  submitDisabled: boolean;
  totalDocuments: number;
  totalChunks: number;
  onAbort: () => void;
  onClearConversation: () => void;
  onNewChat: () => void;
  onQuestionChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onTabChange: (tab: "qa" | "docs") => void;
  onReindexDocument: (documentId: string) => void;
  onDeleteDocument: (documentId: string) => void;
  reindexingDocId: string | null;
};

function ChatPanel({
  activeTab,
  answerBusy,
  documents,
  error,
  hasConversation,
  question,
  response,
  streamedAnswer,
  submittedQuestion,
  submitDisabled,
  totalDocuments,
  totalChunks,
  onAbort,
  onClearConversation,
  onNewChat,
  onQuestionChange,
  onSubmit,
  onTabChange,
  onReindexDocument,
  onDeleteDocument,
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
            onReindexDocument={onReindexDocument}
            onDeleteDocument={onDeleteDocument}
            reindexingDocId={reindexingDocId}
            totalDocuments={totalDocuments}
            totalChunks={totalChunks}
          />
        </div>
      )}
    </section>
  );
}

function QuestionComposer({
  answerBusy,
  floating,
  onAbort,
  onQuestionChange,
  onSubmit,
  question,
  submitDisabled,
}: {
  answerBusy: boolean;
  floating: boolean;
  onAbort: () => void;
  onQuestionChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  question: string;
  submitDisabled: boolean;
}) {
  return (
    <form
      className={`chat-composer-form${floating ? " is-floating" : " is-docked"}`}
      onSubmit={onSubmit}
    >
      <div className="chat-composer">
        <textarea
          aria-label="问题"
          className="chat-input"
          maxLength={100}
          rows={1}
          value={question}
          onChange={(event) => onQuestionChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              if (!answerBusy && !submitDisabled) {
                (event.target as HTMLTextAreaElement).form?.requestSubmit();
              }
            }
          }}
          placeholder="查询文档信息、定位条款、总结内容"
        />
        <button
          aria-label={answerBusy ? "中断回答" : "提交问题"}
          className={`primary-button answer-action-button${answerBusy ? " is-stop" : ""}`}
          disabled={!answerBusy && submitDisabled}
          title={answerBusy ? "中断回答" : "提交问题"}
          type={answerBusy ? "button" : "submit"}
          onClick={answerBusy ? onAbort : undefined}
        >
          {answerBusy ? <Square size={17} /> : <Send size={18} />}
        </button>
      </div>
    </form>
  );
}

function SourcesPane({ response }: { response: AskResponse | null }) {
  if (!response?.sources.length) {
    return null;
  }
  return (
    <div className="sources-list">
      {response.sources.map((source) => (
        <article
          className="source-card"
          key={`${source.source_id}-${source.filename}`}
        >
          <div className="source-title">
            <FileSearch size={16} />
            <span className="source-citation">证据块 [{source.source_id}]</span>
            <span>{source.filename}</span>
          </div>
          <div className="source-meta">
            <span>{formatRetrievalScore(source.score)}</span>
            {source.page_number === null ? null : (
              <span>第 {source.page_number} 页</span>
            )}
          </div>
          <p>{source.snippet}</p>
        </article>
      ))}
    </div>
  );
}

type ConfirmAction = {
  type: "reindex" | "delete";
  documentId: string;
  filename: string;
};

function ConfirmDialog({
  action,
  onConfirm,
  onCancel,
}: {
  action: ConfirmAction;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  const isDelete = action.type === "delete";
  return (
    <div className="confirm-overlay" onClick={onCancel}>
      <div className="confirm-dialog" onClick={(e) => e.stopPropagation()}>
        <p className="confirm-message">
          {isDelete
            ? `确定要删除「${action.filename}」吗？此操作不可撤销。`
            : `确定要重新索引「${action.filename}」吗？`}
        </p>
        <div className="confirm-actions">
          <button
            className="confirm-btn cancel"
            type="button"
            onClick={onCancel}
          >
            取消
          </button>
          <button
            className={`confirm-btn ${isDelete ? "danger" : "primary"}`}
            type="button"
            onClick={onConfirm}
          >
            {isDelete ? "删除" : "重新索引"}
          </button>
        </div>
      </div>
    </div>
  );
}

function DocumentsPage({
  documents,
  onReindexDocument,
  onDeleteDocument,
  reindexingDocId,
  totalDocuments,
  totalChunks,
}: {
  documents: DocumentRecord[];
  onReindexDocument: (documentId: string) => void;
  onDeleteDocument: (documentId: string) => void;
  reindexingDocId: string | null;
  totalDocuments: number;
  totalChunks: number;
}) {
  const PAGE_SIZE = 10;
  const [page, setPage] = useState(1);
  const [pendingAction, setPendingAction] = useState<ConfirmAction | null>(
    null,
  );
  const totalPages = Math.max(1, Math.ceil(documents.length / PAGE_SIZE));

  useEffect(() => {
    setPage(1);
  }, [documents.length]);

  const safePage = Math.min(page, totalPages);
  const start = (safePage - 1) * PAGE_SIZE;
  const pageDocs = documents.slice(start, start + PAGE_SIZE);

  if (!documents.length) {
    return (
      <div className="docs-empty">
        <FileSearch size={32} />
        <strong>暂无文档</strong>
        <span>上传文档后将在此处显示</span>
      </div>
    );
  }

  return (
    <div className="docs-page">
      <div className="kb-stats">
        <div className="kb-stat-item">
          <Database size={16} />
          <span>
            文档 <strong>{totalDocuments}</strong> 份
          </span>
        </div>
        <span className="kb-stat-divider">|</span>
        <div className="kb-stat-item">
          <FileSearch size={16} />
          <span>
            总分块 <strong>{totalChunks}</strong>
          </span>
        </div>
      </div>
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
                  className={reindexingDocId === doc.document_id ? "spin" : ""}
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
    </div>
  );
}

function ErrorPanel({ message }: { message: string }) {
  return (
    <section className="error-panel" aria-live="polite">
      <AlertCircle size={18} />
      <span>{message}</span>
    </section>
  );
}

export default App;
