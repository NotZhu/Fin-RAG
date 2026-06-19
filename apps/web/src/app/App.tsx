import { useEffect, useRef, useState, type FormEvent } from "react";

import {
  AskResponse,
  AskStreamEvent,
  DocumentRecord,
  KnowledgeBaseRecord,
  PipelineStep,
  askQuestionStream,
  createKnowledgeBase as createKnowledgeBaseApi,
  deleteDocument as deleteDocumentApi,
  getKnowledgeBaseReady,
  listKnowledgeBases,
  listDocuments,
  reindexDocument as reindexDocumentApi,
  uploadDocument,
  warmupKnowledgeBase as warmupKnowledgeBaseApi,
} from "../api/client";
import {
  ChatPanel,
  RetrievalTimeline,
  SidebarHeader,
  UploadPanel,
  type ActiveTab,
  type KnowledgeBaseLoadState,
} from "../components";
import { errorMessage } from "../components/utils";
import "../styles/app.css";

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
  const [activeTab, setActiveTab] = useState<ActiveTab>("qa");
  const [question, setQuestion] = useState("");
  const [submittedQuestion, setSubmittedQuestion] = useState("");
  const [knowledgeBaseId, setKnowledgeBaseId] = useState("finance");
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBaseRecord[]>(
    [],
  );
  const [knowledgeBaseLoadState, setKnowledgeBaseLoadState] =
    useState<KnowledgeBaseLoadState>("loading");
  const [knowledgeBaseLoadError, setKnowledgeBaseLoadError] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [answerBusy, setAnswerBusy] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [warmupBusy, setWarmupBusy] = useState(false);
  const [reindexingDocId, setReindexingDocId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [streamedAnswer, setStreamedAnswer] = useState("");
  const [pipelineSteps, setPipelineSteps] = useState<PipelineStep[]>([]);
  const [timingsMs, setTimingsMs] = useState<Record<string, number>>({});
  const [totalDocuments, setTotalDocuments] = useState(0);
  const [totalChunks, setTotalChunks] = useState(0);
  const abortControllerRef = useRef<AbortController | null>(null);
  const warmedKnowledgeBasesRef = useRef<Set<string>>(new Set());
  const currentKnowledgeBase = knowledgeBases.find(
    (item) => item.knowledge_base_id === knowledgeBaseId,
  );

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
      await reindexDocumentApi(documentId, knowledgeBaseId);
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
      await deleteDocumentApi(documentId, knowledgeBaseId);
      await refreshDocuments();
    } catch (caught) {
      setError(errorMessage(caught, "删除文档失败"));
    } finally {
      setUploadBusy(false);
    }
  }

  async function refreshDocuments(nextKnowledgeBaseId = knowledgeBaseId) {
    const payload = await listDocuments(nextKnowledgeBaseId);
    const visibleDocuments = payload.documents.filter(
      (doc) => doc.status !== "deleted",
    );
    setDocuments(visibleDocuments);
    setTotalDocuments(visibleDocuments.length);
    setTotalChunks(
      visibleDocuments.reduce(
        (total, doc) => total + (doc.chunk_count || 0),
        0,
      ),
    );
  }

  async function refreshReady(nextKnowledgeBaseId = knowledgeBaseId) {
    setError("");
    try {
      const ready = await getKnowledgeBaseReady(nextKnowledgeBaseId);
      setTotalDocuments(ready.total_documents);
      setTotalChunks(ready.total_chunks);
      await refreshDocuments(nextKnowledgeBaseId);
    } catch (caught) {
      setError(errorMessage(caught, "无法连接后端服务。"));
    }
  }

  function warmupKnowledgeBaseInBackground(nextKnowledgeBaseId: string) {
    if (!nextKnowledgeBaseId || warmedKnowledgeBasesRef.current.has(nextKnowledgeBaseId)) {
      return;
    }
    warmedKnowledgeBasesRef.current.add(nextKnowledgeBaseId);
    void warmupKnowledgeBaseApi(nextKnowledgeBaseId).catch(() => {
      warmedKnowledgeBasesRef.current.delete(nextKnowledgeBaseId);
    });
  }

  async function handleWarmupKnowledgeBase() {
    if (!knowledgeBaseId || warmupBusy) {
      return;
    }
    const targetKnowledgeBaseId = knowledgeBaseId;
    setWarmupBusy(true);
    setError("");
    try {
      const ready = await warmupKnowledgeBaseApi(targetKnowledgeBaseId);
      warmedKnowledgeBasesRef.current.add(targetKnowledgeBaseId);
      setTotalDocuments(ready.total_documents);
      setTotalChunks(ready.total_chunks);
      await refreshDocuments(targetKnowledgeBaseId);
    } catch (caught) {
      warmedKnowledgeBasesRef.current.delete(targetKnowledgeBaseId);
      setError(errorMessage(caught, "知识库预热失败。"));
    } finally {
      setWarmupBusy(false);
    }
  }

  async function refreshKnowledgeBases() {
    setKnowledgeBaseLoadState("loading");
    setKnowledgeBaseLoadError("");
    try {
      const payload = await listKnowledgeBases();
      setKnowledgeBases(payload.knowledge_bases);
      setKnowledgeBaseLoadState("ready");
      return payload.knowledge_bases;
    } catch (caught) {
      setKnowledgeBaseLoadState("error");
      setKnowledgeBaseLoadError(errorMessage(caught, "无法加载知识库列表。"));
      throw caught;
    }
  }

  async function initializeApp() {
    setError("");
    try {
      const nextKnowledgeBases = await refreshKnowledgeBases();
      const initialKnowledgeBaseId = nextKnowledgeBases.some(
        (item) => item.knowledge_base_id === "finance",
      )
        ? "finance"
        : (nextKnowledgeBases[0]?.knowledge_base_id ?? "finance");
      setKnowledgeBaseId(initialKnowledgeBaseId);
      await refreshReady(initialKnowledgeBaseId);
      warmupKnowledgeBaseInBackground(initialKnowledgeBaseId);
    } catch (caught) {
      setError(errorMessage(caught, "无法连接后端服务。"));
    }
  }

  async function switchKnowledgeBase(nextKnowledgeBaseId: string) {
    if (nextKnowledgeBaseId === knowledgeBaseId) {
      return;
    }
    handleClearConversation();
    setSelectedFile(null);
    setKnowledgeBaseId(nextKnowledgeBaseId);
    try {
      await refreshDocuments(nextKnowledgeBaseId);
      warmupKnowledgeBaseInBackground(nextKnowledgeBaseId);
    } catch (caught) {
      setError(errorMessage(caught, "无法加载知识库文档。"));
    }
  }

  async function createKnowledgeBase(nextKnowledgeBaseId: string) {
    setError("");
    const record = await createKnowledgeBaseApi(nextKnowledgeBaseId);
    await refreshKnowledgeBases();
    await switchKnowledgeBase(record.knowledge_base_id);
  }

  async function retryKnowledgeBases() {
    setError("");
    try {
      const nextKnowledgeBases = await refreshKnowledgeBases();
      const nextKnowledgeBaseId = nextKnowledgeBases.some(
        (item) => item.knowledge_base_id === knowledgeBaseId,
      )
        ? knowledgeBaseId
        : (nextKnowledgeBases[0]?.knowledge_base_id ?? knowledgeBaseId);
      setKnowledgeBaseId(nextKnowledgeBaseId);
      await refreshReady(nextKnowledgeBaseId);
      warmupKnowledgeBaseInBackground(nextKnowledgeBaseId);
    } catch (caught) {
      setError(errorMessage(caught, "无法加载知识库列表。"));
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
    void initializeApp();
  }, []);

  useEffect(() => {
    const hasParsing = documents.some((doc) => doc.status === "parsing");
    if (!hasParsing) return;
    const interval = setInterval(() => {
      void refreshDocuments();
    }, 3000);
    return () => clearInterval(interval);
  }, [documents, knowledgeBaseId]);

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
            knowledgeBaseId={knowledgeBaseId}
            knowledgeBaseLoadError={knowledgeBaseLoadError}
            knowledgeBaseLoadState={knowledgeBaseLoadState}
            knowledgeBases={knowledgeBases}
            onKnowledgeBaseChange={(nextId) => void switchKnowledgeBase(nextId)}
            onRetryKnowledgeBases={() => void retryKnowledgeBases()}
            onToggle={() => setIsSidebarCollapsed((current) => !current)}
          />
          <UploadPanel
            collapsed={isSidebarCollapsed}
            selectedFile={selectedFile}
            uploadBusy={uploadBusy}
            uploadDisabled={answerBusy}
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
          knowledgeBaseId={knowledgeBaseId}
          knowledgeBaseIsAvailable={Boolean(currentKnowledgeBase)}
          knowledgeBaseLoadState={knowledgeBaseLoadState}
          knowledgeBaseUpdatedAt={currentKnowledgeBase?.updated_at ?? ""}
          question={question}
          response={response}
          streamedAnswer={streamedAnswer}
          submittedQuestion={submittedQuestion}
          submitDisabled={uploadBusy}
          totalDocuments={totalDocuments}
          totalChunks={totalChunks}
          warmupBusy={warmupBusy}
          onAbort={handleAbortAnswer}
          onClearConversation={handleClearConversation}
          onNewChat={handleClearConversation}
          onQuestionChange={setQuestion}
          onSubmit={handleSubmit}
          onTabChange={setActiveTab}
          onReindexDocument={reindexDocument}
          onDeleteDocument={deleteDocument}
          onCreateKnowledgeBase={(nextId) => void createKnowledgeBase(nextId)}
          onWarmupKnowledgeBase={() => void handleWarmupKnowledgeBase()}
          reindexingDocId={reindexingDocId}
        />
      </div>
    </main>
  );
}

export default App;
