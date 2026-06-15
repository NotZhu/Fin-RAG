import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  Database,
  FileText,
  Loader2,
  MessageSquareText,
  RefreshCw,
  Search,
  Send,
  Square,
  Timer,
  ToggleLeft,
  ToggleRight,
  Upload
} from "lucide-react";

import {
  ApiError,
  AskResponse,
  AskStreamEvent,
  DocumentRecord,
  ReadyResponse,
  askQuestionStream,
  getReady,
  listDocuments,
  uploadDocument,
  warmupKnowledgeBase
} from "../api/client";
import "../styles/app.css";

type StatusVariant = "ready" | "muted" | "error";

type AnswerMeta = {
  requestId: string;
  processTime: string;
};

const emptyReady: ReadyResponse = {
  ready: false,
  status: "not_ready",
  total_documents: 0,
  total_chunks: 0,
  last_error: null
};

function errorMessage(error: unknown, fallback: string) {
  const apiError = error as ApiError;
  if (apiError?.message) {
    return apiError.request_id ? `${apiError.message} Request ID: ${apiError.request_id}` : apiError.message;
  }
  return fallback;
}

function formatRetrievalScore(score: number | null) {
  if (score === null || Number.isNaN(score)) {
    return "检索分数：-";
  }
  return `检索分数：${score.toFixed(2)}`;
}

function App() {
  const [readyState, setReadyState] = useState<ReadyResponse>(emptyReady);
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [statusVariant, setStatusVariant] = useState<StatusVariant>("muted");
  const [statusText, setStatusText] = useState("正在检查资料库状态...");
  const [question, setQuestion] = useState("");
  const [knowledgeBaseId, setKnowledgeBaseId] = useState("kb-finance");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [returnTrace, setReturnTrace] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [streamedAnswer, setStreamedAnswer] = useState("");
  const [streamEvents, setStreamEvents] = useState<AskStreamEvent[]>([]);
  const [answerMeta, setAnswerMeta] = useState<AnswerMeta | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const statusLabel = useMemo(() => {
    if (statusVariant === "ready") return "Ready";
    if (statusVariant === "error") return "Error";
    return "Not ready";
  }, [statusVariant]);

  async function refreshDocuments() {
    const payload = await listDocuments();
    setDocuments(payload.documents.filter((doc) => doc.status !== "deleted"));
  }

  async function refreshReady() {
    setError("");
    try {
      const payload = await getReady();
      setReadyState(payload);
      if (payload.ready) {
        setStatusVariant("ready");
        setStatusText(`资料库已就绪：${payload.total_documents} 个文档，${payload.total_chunks} 个分块。`);
      } else if (payload.status === "error") {
        setStatusVariant("error");
        setStatusText(payload.last_error || "资料库初始化失败。");
      } else {
        setStatusVariant("muted");
        setStatusText("资料库未初始化，请先预热或上传金融文档。");
      }
      await refreshDocuments();
    } catch (caught) {
      setReadyState(emptyReady);
      setStatusVariant("error");
      setStatusText(errorMessage(caught, "无法连接后端服务。"));
    }
  }

  async function handleWarmup() {
    setBusy(true);
    setError("");
    setStatusVariant("muted");
    setStatusText("正在预热资料库...");
    try {
      const payload = await warmupKnowledgeBase();
      setReadyState(payload);
      setStatusVariant("ready");
      setStatusText(`资料库已就绪：${payload.total_documents} 个文档，${payload.total_chunks} 个分块。`);
      await refreshDocuments();
    } catch (caught) {
      setStatusVariant("error");
      setStatusText("资料库预热失败。");
      setError(errorMessage(caught, "预热请求失败，请确认 API 服务仍在运行。"));
    } finally {
      setBusy(false);
    }
  }

  async function handleUpload() {
    if (!selectedFile) {
      setError("请选择要上传的金融文档。");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await uploadDocument(selectedFile, knowledgeBaseId);
      setSelectedFile(null);
      await refreshReady();
    } catch (caught) {
      setError(errorMessage(caught, "上传失败，请检查文件格式或后端服务。"));
    } finally {
      setBusy(false);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) {
      setError("请输入问题。");
      return;
    }

    setBusy(true);
    setError("");
    setResponse(null);
    setStreamedAnswer("");
    setStreamEvents([]);
    setAnswerMeta(null);
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    try {
      const result = await askQuestionStream(trimmedQuestion, returnTrace, knowledgeBaseId, {
        signal: abortController.signal,
        onEvent(streamEvent) {
          if (returnTrace) {
            setStreamEvents((current) => [...current, streamEvent]);
          }
          if (streamEvent.type === "token") {
            setStreamedAnswer((current) => current + String(streamEvent.data.text ?? ""));
          }
          if (streamEvent.type === "source" && streamEvent.data.source) {
            setResponse((current) => {
              const source = streamEvent.data.source as AskResponse["sources"][number];
              const base =
                current ??
                ({
                  question: trimmedQuestion,
                  route_type: "",
                  retrieval_strategy: "llamaindex_router",
                  answer: "",
                  sources: []
                } satisfies AskResponse);
              return { ...base, sources: [...base.sources, source] };
            });
          }
          if (streamEvent.type === "done" && streamEvent.data.response) {
            setResponse(streamEvent.data.response as AskResponse);
          }
        }
      });
      setResponse(result.payload);
      setStreamedAnswer(result.payload.answer);
      setAnswerMeta({
        requestId: result.requestId,
        processTime: result.processTime
      });
    } catch (caught) {
      if ((caught as Error)?.name === "AbortError") {
        setError("已中断当前回答。");
      } else {
        setError(errorMessage(caught, "问答请求失败，请确认 API 服务仍在运行。"));
      }
    } finally {
      setBusy(false);
      abortControllerRef.current = null;
    }
  }

  function handleAbortAnswer() {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setBusy(false);
  }

  useEffect(() => {
    void refreshReady();
  }, []);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>FinRAG</h1>
        </div>
        <div className={`status-pill status-${statusVariant}`}>{statusLabel}</div>
      </header>

      <section className="workspace">
        <form className="question-panel" onSubmit={handleSubmit}>
          <div className="panel-header">
            <div>
              <h2>问答</h2>
              <p>{statusText}</p>
            </div>
            <button className="secondary-button" type="button" onClick={handleWarmup} disabled={busy}>
              {busy ? <Loader2 className="spin" size={17} /> : <RefreshCw size={17} />}
              预热资料库
            </button>
          </div>

          <label className="question-label" htmlFor="question-input">
            问题
          </label>
          <textarea
            id="question-input"
            rows={5}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="例如：客户风险等级如何与产品风险等级匹配？"
          />

          <div className="action-row">
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={returnTrace}
                onChange={(event) => setReturnTrace(event.target.checked)}
              />
              {returnTrace ? <ToggleRight size={18} /> : <ToggleLeft size={18} />}
              <span>显示调试信息</span>
            </label>
            <button className="primary-button" type="submit" disabled={busy}>
              {busy ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
              提交问题
            </button>
            {busy ? (
              <button className="secondary-button" type="button" onClick={handleAbortAnswer}>
                <Square size={17} />
                中断回答
              </button>
            ) : null}
          </div>
        </form>

        <aside className="stats-panel" aria-label="资料库状态">
          <div className="metric">
            <Database size={18} />
            <div>
              <strong>{readyState.total_documents}</strong>
              <span>文档</span>
            </div>
          </div>
          <div className="metric">
            <Search size={18} />
            <div>
              <strong>{readyState.total_chunks}</strong>
              <span>分块</span>
            </div>
          </div>
          <div className="metric">
            <Timer size={18} />
            <div>
              <strong>{answerMeta?.processTime ?? "-"}</strong>
              <span>ms</span>
            </div>
          </div>
        </aside>
      </section>

      <section className="workspace">
        <div className="question-panel">
          <div className="panel-header">
            <div>
              <h2>文档接入</h2>
              <p>上传金融制度、产品说明、研报或监管资料。</p>
            </div>
            <button className="secondary-button" type="button" onClick={handleUpload} disabled={busy}>
              {busy ? <Loader2 className="spin" size={17} /> : <Upload size={17} />}
              上传并索引
            </button>
          </div>
          <input value={knowledgeBaseId} onChange={(event) => setKnowledgeBaseId(event.target.value)} aria-label="资料库 ID" />
          <input type="file" onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)} aria-label="上传文件" />
        </div>

        <aside className="stats-panel" aria-label="文档列表">
          {documents.length ? (
            documents.map((doc) => (
              <div className="metric" key={doc.document_id}>
                <FileText size={18} />
                <div>
                  <strong>{doc.filename}</strong>
                  <span>
                    {doc.status} · {doc.file_type} · {doc.chunk_count} chunks
                  </span>
                </div>
              </div>
            ))
          ) : (
            <div className="empty-state">暂无已索引文档。</div>
          )}
        </aside>
      </section>

      {error ? (
        <section className="error-panel" aria-live="polite">
          <AlertCircle size={18} />
          <span>{error}</span>
        </section>
      ) : null}

      <section className="answer-panel" aria-label="回答">
        <div className="panel-header">
          <h2>回答</h2>
          <span className="meta-info">
            {answerMeta ? (returnTrace ? `request_id=${answerMeta.requestId} · ${answerMeta.processTime} ms` : `${answerMeta.processTime} ms`) : "等待提问"}
          </span>
        </div>
        <div className={response?.answer ? "answer-text" : "answer-text empty-state"}>
          {streamedAnswer || response?.answer || (busy ? "正在检索资料库..." : "答案会显示在这里。")}
        </div>
      </section>

      {returnTrace && streamEvents.length ? (
        <section className="trace-panel" aria-label="RAG thinking">
          <div className="panel-header">
            <h2>RAG thinking</h2>
            <span className="meta-info">{streamEvents.length} events</span>
          </div>
          <div className="event-list">
            {streamEvents.map((streamEvent, index) => (
              <span className="event-pill" key={`${streamEvent.type}-${index}`}>
                {streamEvent.type}
              </span>
            ))}
          </div>
        </section>
      ) : null}

      <section className="sources-panel" aria-label="来源">
        <div className="panel-header">
          <h2>证据块来源</h2>
          <span className="meta-info">{response?.sources.length ?? 0} 条</span>
        </div>
        {response?.sources.length ? (
          <div className="sources-list">
            {response.sources.map((source) => (
              <article className="source-card" key={`${source.source_id}-${source.filename}`}>
                <div className="source-title">
                  <MessageSquareText size={17} />
                  <span className="source-citation">证据块 [{source.source_id}]</span>
                  <span>{source.filename}</span>
                </div>
                <div className="source-meta">
                  <span>{formatRetrievalScore(source.score)}</span>
                  {source.page_number === null ? null : <span>第 {source.page_number} 页</span>}
                </div>
                <p>{source.snippet}</p>
              </article>
            ))}
          </div>
        ) : (
          <div className="empty-state">来源会显示在这里。</div>
        )}
      </section>

      {returnTrace && response?.trace ? (
        <section className="trace-panel" aria-label="调试信息">
          <div className="panel-header">
            <h2>Trace</h2>
            <span className="meta-info">{response.trace.retrieval_strategy}</span>
          </div>
          <div className="trace-grid">
            {Object.entries(response.trace.timings_ms).map(([name, value]) => (
              <div className="trace-item" key={name}>
                <span>{name}</span>
                <strong>{value.toFixed(2)} ms</strong>
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </main>
  );
}

export default App;
