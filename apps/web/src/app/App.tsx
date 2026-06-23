import { useEffect, useRef, useState, type FormEvent } from "react";

import {
  AskResponse,
  AskStreamEvent,
  DocumentRecord,
  KnowledgeBaseRecord,
  PipelineStep,
  RebuildJobResponse,
  archiveKnowledgeBase as archiveKnowledgeBaseApi,
  askQuestionStream,
  createKnowledgeBase as createKnowledgeBaseApi,
  deleteKnowledgeBase as deleteKnowledgeBaseApi,
  deleteDocument as deleteDocumentApi,
  getKnowledgeBaseRebuildJob,
  getKnowledgeBaseReady,
  listKnowledgeBases,
  listDocuments,
  rebuildKnowledgeBase as rebuildKnowledgeBaseApi,
  reindexDocument as reindexDocumentApi,
  restoreKnowledgeBase as restoreKnowledgeBaseApi,
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

const REBUILD_POLL_INTERVAL_MS = 1200;
const REBUILD_POLL_TIMEOUT_MS = 30 * 60 * 1000;
const DEFAULT_KNOWLEDGE_BASE_ID = "finance";
const ANSWER_CHUNK_PLAYBACK_MS = 16;
const ANSWER_CHUNK_SIZE = 2;

function delay(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function isActiveKnowledgeBase(record: KnowledgeBaseRecord | undefined) {
  return Boolean(record && record.status === "active");
}

function chooseKnowledgeBase(
  records: KnowledgeBaseRecord[],
  preferredId: string,
) {
  return (
    records.find((item) => item.knowledge_base_id === preferredId)?.knowledge_base_id ??
    records.find((item) => item.status === "active")?.knowledge_base_id ??
    records[0]?.knowledge_base_id ??
    preferredId
  );
}

async function waitForRebuildCompletion(
  knowledgeBaseId: string,
  initialJob: RebuildJobResponse,
  onPoll?: () => Promise<void>,
) {
  let job = initialJob;
  const startedAt = Date.now();
  while (job.status === "queued" || job.status === "running") {
    if (Date.now() - startedAt > REBUILD_POLL_TIMEOUT_MS) {
      throw { message: "全量重建超时。" };
    }
    await delay(REBUILD_POLL_INTERVAL_MS);
    job = await getKnowledgeBaseRebuildJob(knowledgeBaseId, job.job_id);
    await onPoll?.();
  }
  if (job.status === "failed") {
    throw { message: job.error ?? "全量重建失败。" };
  }
  return job;
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
  const [rebuildBusy, setRebuildBusy] = useState(false);
  const [warmupBusy, setWarmupBusy] = useState(false);
  const [knowledgeBaseActionBusy, setKnowledgeBaseActionBusy] = useState(false);
  const [reindexingDocId, setReindexingDocId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [streamedAnswer, setStreamedAnswer] = useState("");
  const [pipelineSteps, setPipelineSteps] = useState<PipelineStep[]>([]);
  const [timingsMs, setTimingsMs] = useState<Record<string, number>>({});
  const [totalDocuments, setTotalDocuments] = useState(0);
  const [totalChunks, setTotalChunks] = useState(0);
  const abortControllerRef = useRef<AbortController | null>(null);
  const knowledgeBaseIdRef = useRef(knowledgeBaseId);
  const warmedKnowledgeBasesRef = useRef<Set<string>>(new Set());
  const answerTimersRef = useRef<number[]>([]);
  const answerNextPlaybackAtRef = useRef(0);
  const pendingAnswerTimersRef = useRef(0);
  const requestFinishedRef = useRef(false);
  const sawAnswerTokenRef = useRef(false);
  const handledDoneRef = useRef(false);
  const currentKnowledgeBase = knowledgeBases.find(
    (item) => item.knowledge_base_id === knowledgeBaseId,
  );
  const currentKnowledgeBaseIsActive = isActiveKnowledgeBase(currentKnowledgeBase);

  function handleClearConversation() {
    clearPlaybackTimers();
    setSubmittedQuestion("");
    setResponse(null);
    setStreamedAnswer("");
    setPipelineSteps([]);
    setTimingsMs({});
    setError("");
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setAnswerBusy(false);
  }

  function clearPlaybackTimers() {
    answerTimersRef.current.forEach((timerId) => window.clearTimeout(timerId));
    answerTimersRef.current = [];
    answerNextPlaybackAtRef.current = 0;
    pendingAnswerTimersRef.current = 0;
    requestFinishedRef.current = false;
    sawAnswerTokenRef.current = false;
    handledDoneRef.current = false;
  }

  function finishAnswerIfSettled() {
    if (requestFinishedRef.current && pendingAnswerTimersRef.current === 0) {
      setAnswerBusy(false);
    }
  }

  function scheduleAnswerText(text: string) {
    const characters = Array.from(text || "");
    for (let index = 0; index < characters.length; index += ANSWER_CHUNK_SIZE) {
      const chunk = characters.slice(index, index + ANSWER_CHUNK_SIZE).join("");
      const now = window.performance.now();
      const scheduledAt = Math.max(now, answerNextPlaybackAtRef.current);
      const delayMs = scheduledAt - now;
      answerNextPlaybackAtRef.current = scheduledAt + ANSWER_CHUNK_PLAYBACK_MS;
      pendingAnswerTimersRef.current += 1;
      const timerId = window.setTimeout(() => {
        setStreamedAnswer((current) => current + chunk);
        pendingAnswerTimersRef.current = Math.max(
          pendingAnswerTimersRef.current - 1,
          0,
        );
        finishAnswerIfSettled();
      }, delayMs);
      answerTimersRef.current.push(timerId);
    }
  }

  function displayResponseShell(payload: AskResponse): AskResponse {
    return { ...payload, answer: "" };
  }

  function handleDoneResponse(doneResponse: AskResponse) {
    handledDoneRef.current = true;
    setResponse(displayResponseShell(doneResponse));
    setPipelineSteps(
      [...(doneResponse.trace?.pipeline_steps ?? [])].sort(
        (a, b) => a.order - b.order,
      ),
    );
    if (!sawAnswerTokenRef.current && doneResponse.answer) {
      scheduleAnswerText(doneResponse.answer);
    }
    if (doneResponse.trace?.timings_ms) {
      setTimingsMs(doneResponse.trace.timings_ms);
    }
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

  function warmupKnowledgeBaseInBackground(
    nextKnowledgeBaseId: string,
    records = knowledgeBases,
  ) {
    const target = records.find(
      (item) => item.knowledge_base_id === nextKnowledgeBaseId,
    );
    if (target && target.status !== "active") {
      return;
    }
    if (!nextKnowledgeBaseId || warmedKnowledgeBasesRef.current.has(nextKnowledgeBaseId)) {
      return;
    }
    warmedKnowledgeBasesRef.current.add(nextKnowledgeBaseId);
    void warmupKnowledgeBaseApi(nextKnowledgeBaseId).catch(() => {
      warmedKnowledgeBasesRef.current.delete(nextKnowledgeBaseId);
    });
  }

  async function handleWarmupKnowledgeBase() {
    if (!knowledgeBaseId || warmupBusy || !currentKnowledgeBaseIsActive) {
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

  async function handleRebuildKnowledgeBase() {
    if (!knowledgeBaseId || rebuildBusy || !currentKnowledgeBaseIsActive) {
      return;
    }
    const targetKnowledgeBaseId = knowledgeBaseId;
    setRebuildBusy(true);
    setError("");
    setDocuments((current) =>
      current.map((doc) => ({
        ...doc,
        status: "parsing" as const,
        last_error: null,
      })),
    );
    try {
      const job = await rebuildKnowledgeBaseApi(targetKnowledgeBaseId);
      await waitForRebuildCompletion(targetKnowledgeBaseId, job, async () => {
        if (knowledgeBaseIdRef.current === targetKnowledgeBaseId) {
          await Promise.all([
            refreshDocuments(targetKnowledgeBaseId),
            refreshKnowledgeBases({ showLoading: false }),
          ]);
        }
      });
      warmedKnowledgeBasesRef.current.add(targetKnowledgeBaseId);
      await refreshKnowledgeBases();
      if (knowledgeBaseIdRef.current === targetKnowledgeBaseId) {
        await refreshReady(targetKnowledgeBaseId);
      }
    } catch (caught) {
      warmedKnowledgeBasesRef.current.delete(targetKnowledgeBaseId);
      setError(errorMessage(caught, "全量重建失败。"));
      if (knowledgeBaseIdRef.current === targetKnowledgeBaseId) {
        await refreshDocuments(targetKnowledgeBaseId);
      }
    } finally {
      setRebuildBusy(false);
    }
  }

  async function refreshKnowledgeBases(options: { showLoading?: boolean } = {}) {
    const showLoading = options.showLoading ?? true;
    if (showLoading) {
      setKnowledgeBaseLoadState("loading");
    }
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
      const initialKnowledgeBaseId = chooseKnowledgeBase(
        nextKnowledgeBases,
        DEFAULT_KNOWLEDGE_BASE_ID,
      );
      setKnowledgeBaseId(initialKnowledgeBaseId);
      await refreshReady(initialKnowledgeBaseId);
      if (
        nextKnowledgeBases.find(
          (item) => item.knowledge_base_id === initialKnowledgeBaseId,
        )?.status === "active"
      ) {
        warmupKnowledgeBaseInBackground(initialKnowledgeBaseId, nextKnowledgeBases);
      }
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
      const target = knowledgeBases.find(
        (item) => item.knowledge_base_id === nextKnowledgeBaseId,
      );
      if (!target || target.status === "active") {
        warmupKnowledgeBaseInBackground(nextKnowledgeBaseId);
      }
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
      const nextKnowledgeBaseId = chooseKnowledgeBase(nextKnowledgeBases, knowledgeBaseId);
      setKnowledgeBaseId(nextKnowledgeBaseId);
      await refreshReady(nextKnowledgeBaseId);
      if (
        nextKnowledgeBases.find(
          (item) => item.knowledge_base_id === nextKnowledgeBaseId,
        )?.status === "active"
      ) {
        warmupKnowledgeBaseInBackground(nextKnowledgeBaseId, nextKnowledgeBases);
      }
    } catch (caught) {
      setError(errorMessage(caught, "无法加载知识库列表。"));
    }
  }

  async function handleUpload() {
    if (!currentKnowledgeBaseIsActive) {
      setError("知识库已归档，恢复后才能上传文档。");
      return;
    }
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
    if (!currentKnowledgeBaseIsActive) {
      setError("知识库已归档，恢复后才能提问。");
      return;
    }

    clearPlaybackTimers();
    setAnswerBusy(true);
    setError("");
    setResponse(null);
    setStreamedAnswer("");
    setPipelineSteps([]);
    setTimingsMs({});
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
            if (streamEvent.type === "token") {
              sawAnswerTokenRef.current = true;
              scheduleAnswerText(String(streamEvent.data.text ?? ""));
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
              handleDoneResponse(doneResponse);
            }
          },
        },
      );
      if (!handledDoneRef.current) {
        handleDoneResponse(result.payload);
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
      requestFinishedRef.current = true;
      finishAnswerIfSettled();
      abortControllerRef.current = null;
    }
  }

  function handleAbortAnswer() {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    clearPlaybackTimers();
    setAnswerBusy(false);
  }

  async function handleArchiveKnowledgeBase() {
    if (!knowledgeBaseId || knowledgeBaseActionBusy || currentKnowledgeBase?.status !== "active") {
      return;
    }
    setKnowledgeBaseActionBusy(true);
    setError("");
    try {
      await archiveKnowledgeBaseApi(knowledgeBaseId);
      warmedKnowledgeBasesRef.current.delete(knowledgeBaseId);
      await refreshKnowledgeBases();
    } catch (caught) {
      setError(errorMessage(caught, "归档知识库失败。"));
    } finally {
      setKnowledgeBaseActionBusy(false);
    }
  }

  async function handleRestoreKnowledgeBase() {
    if (!knowledgeBaseId || knowledgeBaseActionBusy) {
      return;
    }
    const targetKnowledgeBaseId = knowledgeBaseId;
    setKnowledgeBaseActionBusy(true);
    setError("");
    try {
      await restoreKnowledgeBaseApi(targetKnowledgeBaseId);
      const nextKnowledgeBases = await refreshKnowledgeBases();
      if (knowledgeBaseIdRef.current === targetKnowledgeBaseId) {
        await refreshReady(targetKnowledgeBaseId);
        warmupKnowledgeBaseInBackground(targetKnowledgeBaseId, nextKnowledgeBases);
      }
    } catch (caught) {
      setError(errorMessage(caught, "恢复知识库失败。"));
    } finally {
      setKnowledgeBaseActionBusy(false);
    }
  }

  async function handleDeleteKnowledgeBase() {
    if (!knowledgeBaseId || knowledgeBaseActionBusy) {
      return;
    }
    const deletedKnowledgeBaseId = knowledgeBaseId;
    setKnowledgeBaseActionBusy(true);
    setError("");
    try {
      await deleteKnowledgeBaseApi(deletedKnowledgeBaseId);
      warmedKnowledgeBasesRef.current.delete(deletedKnowledgeBaseId);
      const nextKnowledgeBases = await refreshKnowledgeBases();
      const nextKnowledgeBaseId = chooseKnowledgeBase(
        nextKnowledgeBases,
        DEFAULT_KNOWLEDGE_BASE_ID,
      );
      handleClearConversation();
      setSelectedFile(null);
      setKnowledgeBaseId(nextKnowledgeBaseId);
      await refreshReady(nextKnowledgeBaseId);
      if (
        nextKnowledgeBases.find(
          (item) => item.knowledge_base_id === nextKnowledgeBaseId,
        )?.status === "active"
      ) {
        warmupKnowledgeBaseInBackground(nextKnowledgeBaseId, nextKnowledgeBases);
      }
    } catch (caught) {
      setError(errorMessage(caught, "删除知识库失败。"));
    } finally {
      setKnowledgeBaseActionBusy(false);
    }
  }

  useEffect(() => {
    void initializeApp();
  }, []);

  useEffect(() => () => clearPlaybackTimers(), []);

  useEffect(() => {
    knowledgeBaseIdRef.current = knowledgeBaseId;
  }, [knowledgeBaseId]);

  useEffect(() => {
    const hasPendingDocument = documents.some(
      (doc) => doc.status === "uploaded" || doc.status === "parsing",
    );
    if (!hasPendingDocument) return;
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
            uploadDisabled={answerBusy || !currentKnowledgeBaseIsActive}
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
          knowledgeBaseStatus={currentKnowledgeBase?.status ?? ""}
          knowledgeBaseUpdatedAt={currentKnowledgeBase?.updated_at ?? ""}
          isDefaultKnowledgeBase={knowledgeBaseId === DEFAULT_KNOWLEDGE_BASE_ID}
          question={question}
          response={response}
          streamedAnswer={streamedAnswer}
          submittedQuestion={submittedQuestion}
          submitDisabled={uploadBusy || !currentKnowledgeBaseIsActive}
          totalDocuments={totalDocuments}
          totalChunks={totalChunks}
          rebuildBusy={rebuildBusy}
          warmupBusy={warmupBusy}
          knowledgeBaseActionBusy={knowledgeBaseActionBusy}
          onAbort={handleAbortAnswer}
          onClearConversation={handleClearConversation}
          onNewChat={handleClearConversation}
          onQuestionChange={setQuestion}
          onSubmit={handleSubmit}
          onTabChange={setActiveTab}
          onReindexDocument={reindexDocument}
          onDeleteDocument={deleteDocument}
          onCreateKnowledgeBase={(nextId) => void createKnowledgeBase(nextId)}
          onRebuildKnowledgeBase={() => void handleRebuildKnowledgeBase()}
          onArchiveKnowledgeBase={() => void handleArchiveKnowledgeBase()}
          onRestoreKnowledgeBase={() => void handleRestoreKnowledgeBase()}
          onDeleteKnowledgeBase={() => void handleDeleteKnowledgeBase()}
          onWarmupKnowledgeBase={() => void handleWarmupKnowledgeBase()}
          reindexingDocId={reindexingDocId}
        />
      </div>
    </main>
  );
}

export default App;
