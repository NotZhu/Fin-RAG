import type {
  ApiError,
  AskResponse,
  AskStreamEvent,
  DocumentRecord,
  DocumentsResponse,
  KnowledgeBaseRecord,
  KnowledgeBasesResponse,
  ReadyResponse,
} from "../types/api";

export type {
  ApiError,
  AskResponse,
  AskStreamEvent,
  DocumentRecord,
  DocumentsResponse,
  KnowledgeBaseRecord,
  KnowledgeBasesResponse,
  PipelineStep,
  ReadyResponse,
} from "../types/api";

async function readError(response: Response): Promise<ApiError> {
  try {
    const payload = (await response.json()) as { error?: ApiError };
    return payload.error ?? { message: `请求失败，HTTP ${response.status}` };
  } catch {
    return { message: `请求失败，HTTP ${response.status}` };
  }
}

async function fetchJson<T>(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    throw await readError(response);
  }
  return (await response.json()) as T;
}

export async function warmupKnowledgeBase(
  knowledgeBaseId: string,
): Promise<ReadyResponse> {
  return fetchJson<ReadyResponse>(
    `/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/warmup`,
    { method: "POST" },
  );
}

export async function getKnowledgeBaseReady(
  knowledgeBaseId: string,
): Promise<ReadyResponse> {
  return fetchJson<ReadyResponse>(
    `/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/ready`,
  );
}

export async function listKnowledgeBases(): Promise<KnowledgeBasesResponse> {
  return fetchJson<KnowledgeBasesResponse>("/knowledge-bases");
}

export async function createKnowledgeBase(
  knowledgeBaseId: string,
): Promise<KnowledgeBaseRecord> {
  return fetchJson<KnowledgeBaseRecord>("/knowledge-bases", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ knowledge_base_id: knowledgeBaseId }),
  });
}

export async function listDocuments(
  knowledgeBaseId: string,
): Promise<DocumentsResponse> {
  return fetchJson<DocumentsResponse>(
    `/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/documents`,
  );
}

export async function uploadDocument(
  file: File,
  knowledgeBaseId: string,
  asyncIndex = true,
) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("async_index", String(asyncIndex));
  return fetchJson(
    `/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/documents/upload`,
    { method: "POST", body: formData },
  );
}

export async function reindexDocument(
  documentId: string,
  knowledgeBaseId: string,
): Promise<DocumentRecord> {
  return fetchJson<DocumentRecord>(
    `/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/documents/${encodeURIComponent(documentId)}/reindex`,
    { method: "POST" },
  );
}

export async function deleteDocument(
  documentId: string,
  knowledgeBaseId: string,
): Promise<DocumentRecord> {
  return fetchJson<DocumentRecord>(
    `/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/documents/${encodeURIComponent(documentId)}`,
    { method: "DELETE" },
  );
}

type AskStreamOptions = {
  onEvent?: (event: AskStreamEvent) => void;
  signal?: AbortSignal;
};

export async function askQuestionStream(
  question: string,
  returnTrace: boolean,
  knowledgeBaseId: string,
  options: AskStreamOptions = {},
) {
  const startTime = performance.now();
  const response = await fetch(
    `/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/ask`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: options.signal,
      body: JSON.stringify({
        question,
        return_sources: true,
        return_trace: returnTrace,
      }),
    },
  );

  if (!response.ok) {
    throw await readError(response);
  }
  if (!response.body) {
    throw { message: "浏览器不支持流式回答。" } satisfies ApiError;
  }

  const payload = await readAskStream(response.body, options.onEvent);
  const elapsedMs = (performance.now() - startTime).toFixed(2);

  return {
    payload,
    requestId: response.headers.get("X-Request-ID") ?? "-",
    processTime: elapsedMs,
  };
}

async function readAskStream(
  body: ReadableStream<Uint8Array>,
  onEvent?: (event: AskStreamEvent) => void,
): Promise<AskResponse> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let donePayload: AskResponse | null = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split(/\r?\n\r?\n/);
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      const event = parseSseEvent(part);
      if (!event) continue;
      onEvent?.(event);
      if (event.type === "done" && isAskResponse(event.data.response)) {
        donePayload = event.data.response;
      }
      if (event.type === "error") {
        throw {
          message: String(event.data.message ?? "流式问答失败。"),
        } satisfies ApiError;
      }
    }
  }

  if (buffer.trim()) {
    const event = parseSseEvent(buffer);
    if (event) {
      onEvent?.(event);
      if (event.type === "done" && isAskResponse(event.data.response)) {
        donePayload = event.data.response;
      }
    }
  }
  if (!donePayload) {
    throw { message: "流式回答未返回完成事件。" } satisfies ApiError;
  }
  return donePayload;
}

function parseSseEvent(block: string): AskStreamEvent | null {
  let type = "message";
  const dataLines: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) {
      type = line.slice("event:".length).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart());
    }
  }
  if (!dataLines.length) {
    return null;
  }
  return {
    type,
    data: JSON.parse(dataLines.join("\n")) as Record<string, unknown>,
  };
}

function isAskResponse(value: unknown): value is AskResponse {
  return Boolean(
    value &&
    typeof value === "object" &&
    "answer" in value &&
    "sources" in value,
  );
}
