import { afterEach, describe, expect, test, vi } from "vitest";

import { askQuestionStream, uploadDocument } from "./client";

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

    expect(events).toEqual(["analysis", "token", "token", "done"]);
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
    let requestBody = "";
    vi.spyOn(globalThis, "fetch").mockImplementationOnce((_input, init) => {
      requestBody = String(init?.body ?? "");
      return Promise.resolve(
        streamResponse([
          'event: done\ndata: {"response":{"question":"q","route_type":"knowledge","retrieval_strategy":"llamaindex_router","answer":"ok","sources":[]}}\n\n'
        ])
      );
    });

    await askQuestionStream("q", false, "kb-finance");

    expect(JSON.parse(requestBody)).toEqual({
      question: "q",
      knowledge_base_id: "kb-finance",
      return_sources: true,
      return_trace: false
    });
  });
});

describe("uploadDocument", () => {
  test("does not send unused document category metadata", async () => {
    const submittedBodies: FormData[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementationOnce((_input, init) => {
      submittedBodies.push(init?.body as FormData);
      return Promise.resolve(
        new Response(JSON.stringify({ status: "indexed" }), {
          headers: { "Content-Type": "application/json" }
        })
      );
    });

    await uploadDocument(new File(["content"], "policy.md", { type: "text/markdown" }), "kb-finance");

    const submittedBody = submittedBodies[0];
    expect(submittedBody.get("file")).toBeInstanceOf(File);
    expect(submittedBody.get("knowledge_base_id")).toBe("kb-finance");
    expect(submittedBody.get("document_category")).toBeNull();
  });
});
