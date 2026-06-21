export type ReadyResponse = {
  ready: boolean;
  status: "ready" | "not_ready" | "error";
  total_documents: number;
  total_chunks: number;
  last_error: string | null;
};

export type RebuildJobStatus = "queued" | "running" | "succeeded" | "failed";

export type RebuildJobResponse = {
  job_id: string;
  knowledge_base_id: string;
  status: RebuildJobStatus;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  result: {
    document_count: number;
    chunk_count: number;
    manifest_schema_version: number;
  } | null;
};

export type DocumentRecord = {
  document_id: string;
  filename: string;
  file_type: string;
  knowledge_base_id: string;
  status: "uploaded" | "parsing" | "indexed" | "failed" | "deleted";
  chunk_count: number;
  upload_time?: string;
  last_error: string | null;
};

export type KnowledgeBaseRecord = {
  knowledge_base_id: string;
  document_count: number;
  status: "active" | "archived" | "deleted";
  created_at: string;
  updated_at: string;
  archived_at: string | null;
  deleted_at: string | null;
};

export type KnowledgeBasesResponse = {
  knowledge_bases: KnowledgeBaseRecord[];
};

export type DocumentsResponse = {
  documents: DocumentRecord[];
};

export type PipelineStep = {
  id: string;
  order: number;
  label: string;
  detail: string;
  status: "running" | "complete" | "error" | "skipped";
  duration_ms: number | null;
  meta: Record<string, unknown>;
};

export type RetrievedSource = {
  source_id: number;
  filename: string;
  page_number: number | null;
  score: number | null;
  snippet: string;
};

export type RagTrace = {
  retrieval_strategy: string;
  route_type?: string;
  filters: Record<string, unknown>;
  timings_ms: Record<string, number>;
  pipeline_steps?: PipelineStep[];
  retrieval_params: {
    top_k: number;
    candidate_k: number;
    rrf_k: number;
  };
  retrieved_nodes?: Array<Record<string, unknown>>;
  evidence_nodes?: Array<Record<string, unknown>>;
  events?: Array<Record<string, unknown>>;
  fusion?: Record<string, unknown>;
  reranker?: Record<string, unknown>;
  auto_merge?: Record<string, unknown>;
  final_decision?: string;
  source_count: number;
};

export type AskResponse = {
  question: string;
  route_type: string;
  retrieval_strategy: string;
  answer: string;
  sources: RetrievedSource[];
  trace?: RagTrace;
};

export type ApiError = {
  message: string;
  request_id?: string;
};

export type AskStreamEvent =
  | { type: "pipeline_step"; data: PipelineStep }
  | { type: "token"; data: { text?: string } & Record<string, unknown> }
  | { type: "source"; data: { source?: RetrievedSource } & Record<string, unknown> }
  | { type: "done"; data: { response?: AskResponse; final_decision?: string } & Record<string, unknown> }
  | { type: "error"; data: { message?: string } & Record<string, unknown> }
  | { type: string; data: Record<string, unknown> };
