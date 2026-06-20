import { afterEach, describe, expect, test, vi } from "vitest";

import {
  askQuestionStream,
  createKnowledgeBase,
  deleteDocument,
  getKnowledgeBaseReady,
  getKnowledgeBaseRebuildJob,
  listKnowledgeBases,
  listDocuments,
  rebuildKnowledgeBase,
  reindexDocument,
  uploadDocument,
  warmupKnowledgeBase,
} from "./client";

function streamResponse(chunks: string[], headers: Record<string, string> = {}) {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(encoder.encode(chunk));
        }
        controller.close();
      }
    }),
    {
      headers: {
        "Content-Type": "text/event-stream",
        "X-Request-ID": "req-stream",
        "X-Process-Time-MS": "22.5",
        ...headers
      }
    }
  );
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("askQuestionStream", () => {
  test("parses SSE events and returns the done response payload", async () => {
    const events: string[] = [];
    vi.spyOn(performance, "now").mockReturnValueOnce(100).mockReturnValueOnce(145.67);
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      streamResponse([
        'event: analysis\ndata: {"route_type":"knowledge"}\n\n',
        'event: pipeline_step\ndata: {"id":"hybrid_search","order":3,"label":"Milvus Hybrid Search","detail":"dense+sparse · candidate_k 10","status":"complete","duration_ms":24,"meta":{}}\n\n',
        'event: token\ndata: {"text":"客户风险"}\n\n',
        'event: token\ndata: {"text":"等级匹配。[1]"}\n\n',
        'event: done\ndata: {"response":{"question":"q","route_type":"knowledge","retrieval_strategy":"llamaindex_router","answer":"客户风险等级匹配。[1]","sources":[]}}\n\n'
      ])
    );

    const result = await askQuestionStream("q", true, "kb-finance", {
      onEvent(event) {
        events.push(event.type);
      }
    });

    expect(events).toEqual(["analysis", "pipeline_step", "token", "token", "done"]);
    expect(result.payload.answer).toBe("客户风险等级匹配。[1]");
    expect(result.requestId).toBe("req-stream");
    expect(result.processTime).toBe("45.67");
  });

  test("passes AbortController signal to fetch", async () => {
    const controller = new AbortController();
    let observedSignal: AbortSignal | undefined;
    vi.spyOn(globalThis, "fetch").mockImplementationOnce((_input, init) => {
      observedSignal = init?.signal ?? undefined;
      return Promise.resolve(
        streamResponse([
          'event: done\ndata: {"response":{"question":"q","route_type":"general","retrieval_strategy":"llamaindex_router","answer":"ok","sources":[]}}\n\n'
        ])
      );
    });

    await askQuestionStream("q", false, "kb-finance", { signal: controller.signal });

    expect(observedSignal).toBe(controller.signal);
  });

  test("does not send retrieval_strategy in ask request body", async () => {
    let observedInput = "";
    let requestBody = "";
    vi.spyOn(globalThis, "fetch").mockImplementationOnce((input, init) => {
      observedInput = String(input);
      requestBody = String(init?.body ?? "");
      return Promise.resolve(
        streamResponse([
          'event: done\ndata: {"response":{"question":"q","route_type":"knowledge","retrieval_strategy":"llamaindex_router","answer":"ok","sources":[]}}\n\n'
        ])
      );
    });

    await askQuestionStream("q", false, "kb-finance");

    expect(observedInput).toBe("/knowledge-bases/kb-finance/ask");
    expect(JSON.parse(requestBody)).toEqual({
      question: "q",
      return_sources: true,
      return_trace: false
    });
  });
});

describe("uploadDocument", () => {
  test("does not send unused document category metadata", async () => {
    let observedInput = "";
    const submittedBodies: FormData[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementationOnce((input, init) => {
      observedInput = String(input);
      submittedBodies.push(init?.body as FormData);
      return Promise.resolve(
        new Response(JSON.stringify({ status: "indexed" }), {
          headers: { "Content-Type": "application/json" }
        })
      );
    });

    await uploadDocument(new File(["content"], "policy.md", { type: "text/markdown" }), "kb-finance");

    const submittedBody = submittedBodies[0];
    expect(observedInput).toBe("/knowledge-bases/kb-finance/documents/upload");
    expect(submittedBody.get("file")).toBeInstanceOf(File);
    expect(submittedBody.get("knowledge_base_id")).toBeNull();
    expect(submittedBody.get("document_category")).toBeNull();
  });
});

describe("knowledge base client", () => {
  test("gets scoped ready state for a knowledge base", async () => {
    let observedInput = "";
    vi.spyOn(globalThis, "fetch").mockImplementationOnce((input) => {
      observedInput = String(input);
      return Promise.resolve(
        new Response(
          JSON.stringify({
            ready: true,
            status: "ready",
            total_documents: 1,
            total_chunks: 3,
            last_error: null,
          }),
          { headers: { "Content-Type": "application/json" } },
        ),
      );
    });

    const result = await getKnowledgeBaseReady("kb-finance");

    expect(observedInput).toBe("/knowledge-bases/kb-finance/ready");
    expect(result.ready).toBe(true);
  });

  test("warms up a scoped knowledge base", async () => {
    let observedInput = "";
    let observedMethod = "";
    vi.spyOn(globalThis, "fetch").mockImplementationOnce((input, init) => {
      observedInput = String(input);
      observedMethod = String(init?.method ?? "");
      return Promise.resolve(
        new Response(
          JSON.stringify({
            ready: true,
            status: "ready",
            total_documents: 1,
            total_chunks: 3,
            last_error: null,
          }),
          { headers: { "Content-Type": "application/json" } },
        ),
      );
    });

    const result = await warmupKnowledgeBase("kb-finance");

    expect(observedInput).toBe("/knowledge-bases/kb-finance/warmup");
    expect(observedMethod).toBe("POST");
    expect(result.ready).toBe(true);
  });

  test("starts and reads a scoped knowledge base rebuild job", async () => {
    const rebuildPayload = {
      job_id: "job-1",
      knowledge_base_id: "kb-finance",
      status: "succeeded",
      created_at: "2026-06-20T00:00:00+00:00",
      started_at: "2026-06-20T00:00:00+00:00",
      completed_at: "2026-06-20T00:00:01+00:00",
      error: null,
      result: {
        document_count: 1,
        chunk_count: 5,
        manifest_schema_version: 1,
      },
    };
    const observed: Array<{ input: string; method: string }> = [];
    vi.spyOn(globalThis, "fetch")
      .mockImplementationOnce((input, init) => {
        observed.push({
          input: String(input),
          method: String(init?.method ?? "GET"),
        });
        return Promise.resolve(
          new Response(JSON.stringify(rebuildPayload), {
            status: 202,
            headers: { "Content-Type": "application/json" },
          }),
        );
      })
      .mockImplementationOnce((input, init) => {
        observed.push({
          input: String(input),
          method: String(init?.method ?? "GET"),
        });
        return Promise.resolve(
          new Response(JSON.stringify(rebuildPayload), {
            headers: { "Content-Type": "application/json" },
          }),
        );
      });

    const started = await rebuildKnowledgeBase("kb-finance");
    const current = await getKnowledgeBaseRebuildJob("kb-finance", "job-1");

    expect(started.status).toBe("succeeded");
    expect(current.result?.chunk_count).toBe(5);
    expect(observed).toEqual([
      {
        input: "/knowledge-bases/kb-finance/rebuilds",
        method: "POST",
      },
      {
        input: "/knowledge-bases/kb-finance/rebuilds/job-1",
        method: "GET",
      },
    ]);
  });

  test("lists knowledge bases", async () => {
    let observedInput = "";
    vi.spyOn(globalThis, "fetch").mockImplementationOnce((input) => {
      observedInput = String(input);
      return Promise.resolve(
        new Response(
          JSON.stringify({
            knowledge_bases: [
              {
                knowledge_base_id: "finance",
                document_count: 1,
                created_at: "2026-06-18T00:00:00+00:00",
                updated_at: "2026-06-18T00:00:00+00:00",
              },
            ],
          }),
          { headers: { "Content-Type": "application/json" } },
        ),
      );
    });

    const result = await listKnowledgeBases();

    expect(observedInput).toBe("/knowledge-bases");
    expect(result.knowledge_bases[0].knowledge_base_id).toBe("finance");
  });

  test("creates a knowledge base from the user supplied id", async () => {
    let observedInput = "";
    let observedMethod = "";
    let observedBody = "";
    vi.spyOn(globalThis, "fetch").mockImplementationOnce((input, init) => {
      observedInput = String(input);
      observedMethod = String(init?.method ?? "");
      observedBody = String(init?.body ?? "");
      return Promise.resolve(
        new Response(
          JSON.stringify({
            knowledge_base_id: "risk",
            document_count: 0,
            created_at: "2026-06-18T00:01:00+00:00",
            updated_at: "2026-06-18T00:01:00+00:00",
          }),
          { headers: { "Content-Type": "application/json" } },
        ),
      );
    });

    const result = await createKnowledgeBase("risk");

    expect(observedInput).toBe("/knowledge-bases");
    expect(observedMethod).toBe("POST");
    expect(JSON.parse(observedBody)).toEqual({ knowledge_base_id: "risk" });
    expect(result.knowledge_base_id).toBe("risk");
  });
});

describe("document lifecycle client", () => {
  test("lists documents for a knowledge base", async () => {
    let observedInput = "";
    vi.spyOn(globalThis, "fetch").mockImplementationOnce((input) => {
      observedInput = String(input);
      return Promise.resolve(
        new Response(JSON.stringify({ documents: [] }), {
          headers: { "Content-Type": "application/json" }
        })
      );
    });

    await listDocuments("kb-finance");

    expect(observedInput).toBe("/knowledge-bases/kb-finance/documents");
  });

  test("reindexes a document by id", async () => {
    let observedInput = "";
    let observedMethod = "";
    vi.spyOn(globalThis, "fetch").mockImplementationOnce((input, init) => {
      observedInput = String(input);
      observedMethod = String(init?.method ?? "");
      return Promise.resolve(
        new Response(JSON.stringify({ document_id: "doc-1", status: "indexed" }), {
          headers: { "Content-Type": "application/json" }
        })
      );
    });

    await reindexDocument("doc-1", "kb-finance");

    expect(observedInput).toBe("/knowledge-bases/kb-finance/documents/doc-1/reindex");
    expect(observedMethod).toBe("POST");
  });

  test("deletes a document by id", async () => {
    let observedInput = "";
    let observedMethod = "";
    vi.spyOn(globalThis, "fetch").mockImplementationOnce((input, init) => {
      observedInput = String(input);
      observedMethod = String(init?.method ?? "");
      return Promise.resolve(
        new Response(JSON.stringify({ document_id: "doc-1", status: "deleted" }), {
          headers: { "Content-Type": "application/json" }
        })
      );
    });

    await deleteDocument("doc-1", "kb-finance");

    expect(observedInput).toBe("/knowledge-bases/kb-finance/documents/doc-1");
    expect(observedMethod).toBe("DELETE");
  });
});
