import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import App from "./App";

const readyPayload = {
  ready: true,
  status: "ready",
  total_documents: 2,
  total_chunks: 5,
  last_error: null
};

const documentsPayload = {
  documents: [
    {
      document_id: "doc-1",
      filename: "适当性管理办法.md",
      file_type: "md",
      knowledge_base_id: "kb-finance",
      status: "indexed",
      chunk_count: 3,
      last_error: null
    }
  ]
};

const answerPayload = {
  question: "客户风险等级如何匹配？",
  route_type: "knowledge",
  retrieval_strategy: "llamaindex_router",
  answer: "客户风险等级应与产品风险等级匹配。[1]",
  sources: [
    {
      source_id: 1,
      filename: "policy.md",
      page_number: null,
      score: 0.9,
      snippet: "客户风险等级应与产品风险等级匹配。"
    }
  ],
  trace: {
    retrieval_strategy: "llamaindex_router",
    route_type: "knowledge",
    filters: { knowledge_base_id: "kb-finance" },
    timings_ms: {
      analysis: 1,
      retrieval: 3,
      context_build: 4,
      generation: 5,
      total: 13
    },
    retrieval_params: {
      top_k: 3,
      candidate_k: 10,
      rrf_k: 60,
      auto_merge_enabled: true
    },
    retrieved_nodes: [],
    evidence_nodes: [],
    events: [{ stage: "query_analysis" }],
    fusion: { fusion_provider: "llamaindex", fusion_mode: "reciprocal_rerank" },
    source_count: 1
  }
};

function sseResponse(chunks: string[], headers: Record<string, string> = {}) {
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
        "X-Request-ID": "req-test",
        "X-Process-Time-MS": "18.5",
        ...headers
      }
    }
  );
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("FinRAG React app", () => {
  test("renders ready status and indexed documents", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify(readyPayload), { headers: { "Content-Type": "application/json" } }))
      .mockResolvedValueOnce(new Response(JSON.stringify(documentsPayload), { headers: { "Content-Type": "application/json" } }));

    render(<App />);

    expect(await screen.findByRole("heading", { name: "FinRAG" })).toBeInTheDocument();
    expect(screen.queryByText("金融资料库工作台")).not.toBeInTheDocument();
    expect(screen.getByText(/2 个文档/)).toBeInTheDocument();
    expect(screen.getByText("适当性管理办法.md")).toBeInTheDocument();
    expect(screen.queryByLabelText("人工分类")).not.toBeInTheDocument();
  });

  test("submits a financial question and renders readable sources without debug trace by default", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify(readyPayload), { headers: { "Content-Type": "application/json" } }))
      .mockResolvedValueOnce(new Response(JSON.stringify(documentsPayload), { headers: { "Content-Type": "application/json" } }))
      .mockResolvedValueOnce(
        sseResponse([
          'event: analysis\ndata: {"route_type":"knowledge"}\n\n',
          'event: retrieval_hits\ndata: {"hit_count":1}\n\n',
          'event: grading\ndata: {"relevant":true}\n\n',
          'event: context\ndata: {"source_count":1}\n\n',
          'event: token\ndata: {"text":"客户风险等级应与产品风险等级匹配。"}\n\n',
          'event: token\ndata: {"text":"[1]"}\n\n',
          `event: source\ndata: {"source":${JSON.stringify(answerPayload.sources[0])}}\n\n`,
          `event: done\ndata: {"response":${JSON.stringify(answerPayload)}}\n\n`
        ])
      );

    const { container } = render(<App />);

    await screen.findByRole("heading", { name: "FinRAG" });
    await userEvent.type(screen.getByLabelText("问题"), "客户风险等级如何匹配？");
    await userEvent.click(screen.getByRole("button", { name: /提交问题/ }));

    expect(await screen.findByText("客户风险等级应与产品风险等级匹配。[1]")).toBeInTheDocument();
    expect(container.querySelector(".source-title")).toHaveTextContent("证据块 [1]");
    expect(container.querySelector(".source-title")).toHaveTextContent("policy.md");
    expect(screen.getByText("检索分数：0.90")).toBeInTheDocument();
    expect(screen.queryByText(/相关度/)).not.toBeInTheDocument();
    expect(screen.queryByText(/chunk=/)).not.toBeInTheDocument();
    expect(screen.queryByText(/score=/)).not.toBeInTheDocument();
    expect(screen.queryByText("RAG thinking")).not.toBeInTheDocument();
    expect(screen.queryByText("Trace")).not.toBeInTheDocument();
    expect(screen.queryByText(/request_id=/)).not.toBeInTheDocument();
  });

  test("shows trace details when debug info is enabled", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify(readyPayload), { headers: { "Content-Type": "application/json" } }))
      .mockResolvedValueOnce(new Response(JSON.stringify(documentsPayload), { headers: { "Content-Type": "application/json" } }))
      .mockResolvedValueOnce(
        sseResponse([
          'event: analysis\ndata: {"route_type":"knowledge"}\n\n',
          'event: retrieval_hits\ndata: {"hit_count":1}\n\n',
          'event: token\ndata: {"text":"客户风险等级应与产品风险等级匹配。[1]"}\n\n',
          `event: source\ndata: {"source":${JSON.stringify(answerPayload.sources[0])}}\n\n`,
          `event: done\ndata: {"response":${JSON.stringify(answerPayload)}}\n\n`
        ])
      );

    render(<App />);

    await screen.findByRole("heading", { name: "FinRAG" });
    await userEvent.type(screen.getByLabelText("问题"), "客户风险等级如何匹配？");
    await userEvent.click(screen.getByLabelText("显示调试信息"));
    await userEvent.click(screen.getByRole("button", { name: /提交问题/ }));

    expect(await screen.findByText("RAG thinking")).toBeInTheDocument();
    expect(screen.getByText("Trace")).toBeInTheDocument();
    expect(screen.getByText(/request_id=req-test/)).toBeInTheDocument();
    expect(screen.getByText("retrieval")).toBeInTheDocument();
  });

  test("can abort the current streaming answer", async () => {
    let askSignal: AbortSignal | undefined;
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify(readyPayload), { headers: { "Content-Type": "application/json" } }))
      .mockResolvedValueOnce(new Response(JSON.stringify(documentsPayload), { headers: { "Content-Type": "application/json" } }))
      .mockImplementationOnce((_input, init) => {
        askSignal = init?.signal ?? undefined;
        return Promise.resolve(
          new Response(
            new ReadableStream({
              start(controller) {
                controller.enqueue(new TextEncoder().encode('event: analysis\ndata: {"route_type":"knowledge"}\n\n'));
              }
            }),
            { headers: { "Content-Type": "text/event-stream" } }
          )
        );
      });

    render(<App />);

    await screen.findByRole("heading", { name: "FinRAG" });
    await userEvent.type(screen.getByLabelText("问题"), "客户风险等级如何匹配？");
    await userEvent.click(screen.getByRole("button", { name: /提交问题/ }));
    await userEvent.click(await screen.findByRole("button", { name: /中断回答/ }));

    expect(askSignal?.aborted).toBe(true);
  });
});
