import { readFileSync } from "node:fs";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import App from "./App";

const appStyles = readFileSync("src/styles/app.css", "utf8");

const readyPayload = {
  ready: true,
  status: "ready",
  total_documents: 2,
  total_chunks: 5,
  last_error: null,
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
      upload_time: "2026-06-14T13:35:37.897082+00:00",
      last_error: null,
    },
  ],
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
      snippet: "客户风险等级应与产品风险等级匹配。",
    },
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
      total: 13,
    },
    retrieval_params: {
      top_k: 3,
      candidate_k: 10,
      rrf_k: 60,
      auto_merge_enabled: true,
    },
    retrieved_nodes: [],
    evidence_nodes: [],
    events: [{ stage: "query_analysis" }],
    fusion: { fusion_provider: "llamaindex", fusion_mode: "reciprocal_rerank" },
    source_count: 1,
  },
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
      },
    }),
    {
      headers: {
        "Content-Type": "text/event-stream",
        "X-Request-ID": "req-test",
        "X-Process-Time-MS": "18.5",
        ...headers,
      },
    },
  );
}

function deferredResponse() {
  let resolve!: (response: Response) => void;
  const promise = new Promise<Response>((innerResolve) => {
    resolve = innerResolve;
  });
  return { promise, resolve };
}

function mockInitialLoad() {
  return vi
    .spyOn(globalThis, "fetch")
    .mockResolvedValueOnce(
      new Response(JSON.stringify(readyPayload), {
        headers: { "Content-Type": "application/json" },
      }),
    )
    .mockResolvedValueOnce(
      new Response(JSON.stringify(documentsPayload), {
        headers: { "Content-Type": "application/json" },
      }),
    );
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("FinRAG chat interface", () => {
  test("uses a sparse ChatGPT-like layout with a collapsible sidebar", async () => {
    mockInitialLoad();

    const { container } = render(<App />);

    await screen.findByText("文档就绪，随时提问");

    expect(appStyles).toContain("color-scheme: light;");
    expect(appStyles).toContain("--bg: #ffffff;");
    expect(appStyles).toContain("--rail: #f9f9f9;");
    expect(appStyles).toContain("grid-template-columns: 320px minmax(0, 1fr);");
    expect(appStyles).toContain(".app-layout.sidebar-collapsed");
    expect(appStyles).toContain(".left-rail.is-collapsed");
    expect(appStyles).toContain(".rail-card {");
    expect(appStyles).toContain("border: 1px solid var(--border);");
    expect(appStyles).toContain("border-radius: 16px;");
    expect(appStyles).toContain(".chat-panel.is-empty");
    expect(appStyles).toContain(".chat-composer-form.is-floating");
    expect(appStyles).toContain(".chat-composer-form.is-docked");
    expect(appStyles).not.toContain("color-scheme: dark;");
    expect(appStyles).not.toContain('[data-theme="dark"]');

    const layout = container.querySelector(".app-layout");
    const sidebar = container.querySelector(".left-rail") as HTMLElement;
    const chat = container.querySelector(".chat-panel") as HTMLElement;

    expect(layout).not.toBeNull();
    expect(sidebar).not.toBeNull();
    expect(chat).not.toBeNull();
    expect(Array.from(layout?.children ?? [])).toEqual([sidebar, chat]);
    expect(
      within(sidebar).getByRole("region", { name: "文件上传" }),
    ).toHaveClass("rail-card");
    expect(
      within(sidebar).getByRole("region", { name: "实时检索链路" }),
    ).toHaveClass("rail-card");
    expect(within(chat).getByText("文档就绪，随时提问")).toBeInTheDocument();
    const emptyComposerForm = within(chat)
      .getByPlaceholderText("查询文档信息、定位条款、总结内容")
      .closest("form");
    expect(emptyComposerForm).toHaveClass("chat-composer-form");
    expect(emptyComposerForm).toHaveClass("is-floating");

    await userEvent.click(screen.getByRole("button", { name: "收起侧栏" }));

    expect(container.querySelector(".app-layout")).toHaveClass(
      "sidebar-collapsed",
    );
    expect(container.querySelector(".left-rail")).toHaveClass("is-collapsed");
    expect(
      screen.getByRole("button", { name: "展开侧栏" }),
    ).toBeInTheDocument();
  });

  test("has no theme toggle button", async () => {
    mockInitialLoad();

    render(<App />);

    await screen.findByText("文档就绪，随时提问");

    expect(
      screen.queryByRole("button", { name: /切换日间模式|切换夜间模式/ }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /日间模式|夜间模式/ }),
    ).not.toBeInTheDocument();
  });

  test("has tab bar on the left and new chat button on the right in the header", async () => {
    mockInitialLoad();

    render(<App />);

    await screen.findByText("文档就绪，随时提问");

    const tabBar = screen.getByRole("navigation");
    const newChatButton = screen.getByRole("button", { name: "新聊天" });

    expect(tabBar).toBeInTheDocument();
    expect(newChatButton).toBeInTheDocument();
    expect(within(tabBar).getByText("问答")).toBeInTheDocument();
    expect(within(tabBar).getByText("文档")).toBeInTheDocument();
  });

  test("removes descriptive prompts, refresh controls, latency, document management, and debug tabs", async () => {
    mockInitialLoad();

    render(<App />);

    await screen.findByText("文档就绪，随时提问");

    expect(
      screen.queryByText(/请先上传|知识库已就绪|上传文档后|提交问题后/),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "预热知识库" }),
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/耗时/)).not.toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "已索引文档" }),
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/文档：/)).not.toBeInTheDocument();
    expect(screen.queryByText(/分块：/)).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "回答" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "引用" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "调试" })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("显示调试信息")).not.toBeInTheDocument();
  });

  test("opens a compact confirmation dialog after selecting a file and indexes on confirm", async () => {
    const uploadRequest = deferredResponse();
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify(readyPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(documentsPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockImplementationOnce(() => uploadRequest.promise)
      .mockResolvedValueOnce(
        new Response(JSON.stringify(readyPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(documentsPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      );

    render(<App />);

    await screen.findByText("文档就绪，随时提问");
    await userEvent.upload(
      screen.getByLabelText("上传文件"),
      new File(["policy"], "policy.md", { type: "text/markdown" }),
    );

    const dialog = await screen.findByRole("dialog", { name: "确认索引文档" });
    expect(within(dialog).getByLabelText("policy.md")).toBeInTheDocument();
    expect(within(dialog).getByRole("textbox", { name: "知识库" })).toHaveValue(
      "kb-finance",
    );
    expect(within(dialog).getByRole("button", { name: "索引" })).toHaveClass(
      "modal-action-button",
    );

    await userEvent.clear(
      within(dialog).getByRole("textbox", { name: "知识库" }),
    );
    await userEvent.type(
      within(dialog).getByRole("textbox", { name: "知识库" }),
      "kb-credit",
    );
    await userEvent.click(within(dialog).getByRole("button", { name: "索引" }));

    await waitFor(() => {
      expect(
        screen.queryByRole("dialog", { name: "确认索引文档" }),
      ).not.toBeInTheDocument();
    });
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/documents/upload",
        expect.objectContaining({ method: "POST", body: expect.any(FormData) }),
      );
    });
    const uploadCall = fetchMock.mock.calls.find(
      ([url]) => url === "/documents/upload",
    );
    expect(
      ((uploadCall?.[1] as RequestInit).body as FormData).get(
        "knowledge_base_id",
      ),
    ).toBe("kb-credit");
    uploadRequest.resolve(
      new Response(JSON.stringify({ status: "indexed" }), {
        headers: { "Content-Type": "application/json" },
      }),
    );
  });

  test("only advertises upload suffixes supported by the backend", async () => {
    mockInitialLoad();

    render(<App />);

    await screen.findByText("文档就绪，随时提问");

    expect(screen.getByLabelText("上传文件")).toHaveAttribute(
      "accept",
      ".pdf,.docx,.md,.txt",
    );
  });

  test("submits a question from the compact composer and renders answer, sources, and timeline without latency text", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify(readyPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(documentsPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        sseResponse([
          'event: analysis\ndata: {"route_type":"knowledge"}\n\n',
          'event: pipeline_step\ndata: {"id":"query_analysis","order":1,"label":"Query Analysis","detail":"识别金融风控问题","status":"complete","duration_ms":12,"meta":{}}\n\n',
          'event: pipeline_step\ndata: {"id":"hybrid_search","order":3,"label":"Milvus Hybrid Search","detail":"dense+sparse · candidate_k 10","status":"complete","duration_ms":24,"meta":{}}\n\n',
          'event: token\ndata: {"text":"客户风险等级应与产品风险等级匹配。"}\n\n',
          'event: token\ndata: {"text":"[1]"}\n\n',
          `event: source\ndata: {"source":${JSON.stringify(answerPayload.sources[0])}}\n\n`,
          `event: done\ndata: {"response":${JSON.stringify(answerPayload)}}\n\n`,
        ]),
      );

    const { container } = render(<App />);

    await screen.findByText("文档就绪，随时提问");
    const emptyComposerForm = screen
      .getByPlaceholderText("查询文档信息、定位条款、总结内容")
      .closest("form");
    expect(emptyComposerForm).toHaveClass("chat-composer-form");
    expect(emptyComposerForm).toHaveClass("is-floating");

    await userEvent.type(
      screen.getByLabelText("问题"),
      "客户风险等级如何匹配？",
    );
    await userEvent.click(screen.getByRole("button", { name: "提交问题" }));

    expect(await screen.findByText("客户风险等级如何匹配？")).toHaveClass(
      "chat-bubble",
    );
    expect(
      await screen.findByText("客户风险等级应与产品风险等级匹配。[1]"),
    ).toBeInTheDocument();
    expect(container.querySelector(".source-card")).toHaveTextContent(
      "policy.md",
    );
    expect(
      container.querySelector(".chat-panel > .chat-composer-form"),
    ).toHaveClass("is-docked");
    expect(screen.getByText("Milvus Hybrid Search")).toBeInTheDocument();
    expect(screen.queryByText(/耗时/)).not.toBeInTheDocument();
  });

  test("can abort the current streaming answer from the circular composer button", async () => {
    let askSignal: AbortSignal | undefined;
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify(readyPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(documentsPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockImplementationOnce((_input, init) => {
        askSignal = init?.signal ?? undefined;
        return Promise.resolve(
          new Response(
            new ReadableStream({
              start(controller) {
                controller.enqueue(
                  new TextEncoder().encode(
                    'event: analysis\ndata: {"route_type":"knowledge"}\n\n',
                  ),
                );
              },
            }),
            { headers: { "Content-Type": "text/event-stream" } },
          ),
        );
      });

    render(<App />);

    await screen.findByText("文档就绪，随时提问");
    await userEvent.type(
      screen.getByLabelText("问题"),
      "客户风险等级如何匹配？",
    );
    await userEvent.click(screen.getByRole("button", { name: "提交问题" }));
    const stopButton = await screen.findByRole("button", { name: "中断回答" });

    expect(stopButton).toHaveClass("answer-action-button");
    await userEvent.click(stopButton);

    expect(askSignal?.aborted).toBe(true);
  });

  test("shows reindex and delete buttons on document cards", async () => {
    mockInitialLoad();

    render(<App />);

    await screen.findByText("文档就绪，随时提问");

    await userEvent.click(screen.getByRole("button", { name: /文档/ }));

    expect(
      screen.getByRole("button", { name: "重新索引" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "删除" })).toBeInTheDocument();
  });

  test("new chat button clears the conversation", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify(readyPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(documentsPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        sseResponse([
          `event: done\ndata: {"response":${JSON.stringify(answerPayload)}}\n\n`,
        ]),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(documentsPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      );

    render(<App />);

    await screen.findByText("文档就绪，随时提问");
    await userEvent.type(
      screen.getByLabelText("问题"),
      "客户风险等级如何匹配？",
    );
    await userEvent.click(screen.getByRole("button", { name: "提交问题" }));

    expect(
      await screen.findByText("客户风险等级如何匹配？"),
    ).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "新聊天" }));

    expect(
      screen.queryByText("客户风险等级如何匹配？"),
    ).not.toBeInTheDocument();
  });

  test("polls for document status when a document is parsing", async () => {
    const parsingPayload = {
      documents: [
        {
          document_id: "doc-2",
          filename: "report.pdf",
          file_type: "pdf",
          knowledge_base_id: "kb-finance",
          status: "parsing",
          chunk_count: 0,
          upload_time: "2026-06-14T13:35:37.897082+00:00",
          last_error: null,
        },
      ],
    };

    const indexedPayload = {
      documents: [
        {
          document_id: "doc-2",
          filename: "report.pdf",
          file_type: "pdf",
          knowledge_base_id: "kb-finance",
          status: "indexed",
          chunk_count: 5,
          upload_time: "2026-06-14T13:35:37.897082+00:00",
          last_error: null,
        },
      ],
    };

    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify(readyPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(parsingPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(indexedPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      );

    render(<App />);

    await screen.findByText("文档就绪，随时提问");

    await userEvent.click(screen.getByRole("button", { name: /文档/ }));

    expect(screen.getByText("解析中")).toBeInTheDocument();

    await waitFor(
      () => {
        expect(screen.queryByText("解析中")).not.toBeInTheDocument();
      },
      { timeout: 5000 },
    );
  });

  test("shows last_error on failed document cards", async () => {
    const failedPayload = {
      documents: [
        {
          document_id: "doc-3",
          filename: "corrupt.pdf",
          file_type: "pdf",
          knowledge_base_id: "kb-finance",
          status: "failed",
          chunk_count: 0,
          upload_time: "2026-06-15T10:00:00.000000+00:00",
          last_error: "PDF 解析失败：文件已损坏",
        },
      ],
    };

    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify(readyPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(failedPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      );

    render(<App />);

    await screen.findByText("文档就绪，随时提问");

    await userEvent.click(screen.getByRole("button", { name: /文档/ }));

    expect(screen.getByText(/索引失败/)).toBeInTheDocument();
    expect(screen.getByText("PDF 解析失败：文件已损坏")).toBeInTheDocument();
  });
});
