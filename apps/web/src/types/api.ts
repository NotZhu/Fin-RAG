export type ReadyResponse = {
  ready: boolean;
  status: "ready" | "not_ready" | "error";
  total_documents: number;
  total_chunks: number;
  last_error: string | null;
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

export type DocumentsResponse = {
  documents: DocumentRecord[];
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

export type AskStreamEvent = {
  type: string;
  data: Record<string, unknown>;
};
