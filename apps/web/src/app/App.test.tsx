import { readFileSync } from "node:fs";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, test, vi } from "vitest";

import App from "./App";

const appStyles = readFileSync("src/styles/app.css", "utf8");
const indexHtml = readFileSync("index.html", "utf8");

const readyPayload = {
  ready: true,
  status: "ready",
  total_documents: 2,
  total_chunks: 5,
  last_error: null,
};

const knowledgeBasesPayload = {
  knowledge_bases: [
    {
      knowledge_base_id: "finance",
      document_count: 1,
      status: "active",
      created_at: "2026-06-18T00:00:00+00:00",
      updated_at: "2026-06-18T00:00:00+00:00",
      archived_at: null,
      deleted_at: null,
    },
    {
      knowledge_base_id: "risk",
      document_count: 0,
      status: "active",
      created_at: "2026-06-18T00:01:00+00:00",
      updated_at: "2026-06-18T00:01:00+00:00",
      archived_at: null,
      deleted_at: null,
    },
  ],
};

const documentsPayload = {
  documents: [
    {
      document_id: "doc-1",
      filename: "适当性管理办法.md",
      file_type: "md",
      knowledge_base_id: "finance",
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
    filters: { knowledge_base_id: "finance" },
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

function jsonResponse(payload: unknown) {
  return new Response(JSON.stringify(payload), {
    headers: { "Content-Type": "application/json" },
  });
}

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
    .mockResolvedValueOnce(jsonResponse(knowledgeBasesPayload))
    .mockResolvedValueOnce(jsonResponse(readyPayload))
    .mockResolvedValueOnce(jsonResponse(documentsPayload))
    .mockResolvedValueOnce(jsonResponse(readyPayload));
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("FinRAG chat interface", () => {
  test("shows a disabled knowledge base loading state during initial load", () => {
    const knowledgeBaseRequest = deferredResponse();
    vi.spyOn(globalThis, "fetch").mockImplementationOnce(
      () => knowledgeBaseRequest.promise,
    );

    render(<App />);

    const loadingButton = screen.getByRole("button", {
      name: "知识库加载中",
    });
    expect(loadingButton).toBeDisabled();
    expect(loadingButton).toHaveTextContent("加载中");
    expect(
      screen.queryByRole("listbox", { name: "知识库列表" }),
    ).not.toBeInTheDocument();
  });

  test("shows knowledge base load failure and retries from the switcher", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("{}", { status: 503 }))
      .mockResolvedValueOnce(
        new Response(JSON.stringify(knowledgeBasesPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
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
      .mockResolvedValueOnce(jsonResponse(readyPayload));

    const { container } = render(<App />);

    const retryButton = await screen.findByRole("button", {
      name: "重新加载知识库",
    });
    expect(retryButton).toHaveTextContent("加载失败");

    await userEvent.click(screen.getByRole("button", { name: /文档/ }));

    const stats = container.querySelector(".kb-stats") as HTMLElement;
    expect(within(stats).getByText("加载失败")).toHaveClass("kb-stat-value");
    expect(within(stats).queryByText("Finance")).not.toBeInTheDocument();
    expect(within(stats).getByText("更新")).toHaveClass("kb-stat-label");
    expect(within(stats).queryByText("-")).not.toBeInTheDocument();

    await userEvent.click(retryButton);

    expect(
      await screen.findByRole("button", { name: "切换知识库 finance" }),
    ).toBeInTheDocument();
    expect(
      fetchMock.mock.calls.filter(([url]) => url === "/knowledge-bases"),
    ).toHaveLength(2);
  });

  test("uses a sparse ChatGPT-like layout with a collapsible sidebar", async () => {
    mockInitialLoad();

    const { container } = render(<App />);

    await screen.findByText("文档就绪，随时提问");

    expect(appStyles).toContain("color-scheme: light;");
    expect(appStyles).toContain("--bg: #fcfaf5;");
    expect(appStyles).toContain("--rail: #f7f5ee;");
    expect(appStyles).toContain("grid-template-columns: 320px minmax(0, 1fr);");
    expect(appStyles).toContain(".app-layout.sidebar-collapsed");
    expect(appStyles).toContain(".left-rail.is-collapsed");
    expect(appStyles).toContain(".rail-card {");
    expect(appStyles).toContain("border: 1px solid var(--border);");
    expect(appStyles).toContain("border-radius: 16px;");
    expect(appStyles).toContain(".chat-panel.is-empty");
    expect(appStyles).toContain(".chat-composer-form.is-floating");
    expect(appStyles).toContain(".chat-composer-form.is-docked");
    expect(appStyles).toContain("grid-template-columns: auto minmax(0, 1fr) auto;");
    expect(appStyles).toContain("justify-self: center;");
    expect(appStyles).toContain("--kb-switch-width: 112px;");
    expect(appStyles).toContain("--kb-switch-height: 28px;");
    expect(appStyles).toContain("border-radius: 8px;");
    expect(appStyles).toContain("height: var(--kb-switch-height);");
    expect(appStyles).toContain("width: var(--kb-switch-width);");
    expect(appStyles).toContain("color: var(--muted);");
    expect(appStyles).toContain("justify-content: center;");
    expect(appStyles).toContain("position: absolute;");
    expect(appStyles).toContain("padding-left: 1px;");
    expect(appStyles).toContain("justify-self: start;");
    expect(appStyles).toContain(".kb-stat-label");
    expect(appStyles).toContain(".kb-stat-value");
    expect(appStyles).toContain(".kb-stat-item svg");
    expect(appStyles).toContain(".kb-stat-divider");
    expect(appStyles).toMatch(/\.kb-stat-label\s*\{[^}]*color: var\(--text\);/s);
    expect(appStyles).toMatch(/\.kb-stat-value\s*\{[^}]*color: var\(--muted\);/s);
    expect(appStyles).toMatch(/\.kb-stat-item svg\s*\{[^}]*color: var\(--text\);/s);
    expect(appStyles).toMatch(/\.kb-stat-divider\s*\{[^}]*color: var\(--text\);/s);
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
    const kbSwitchButton = within(sidebar).getByRole("button", {
      name: "切换知识库 finance",
    });
    expect(kbSwitchButton).toHaveClass("kb-switch-button");
    expect(kbSwitchButton.querySelectorAll("svg")).toHaveLength(1);
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

  test("uses the Fin RAG browser tab title with a blank favicon", () => {
    expect(indexHtml).toContain("<title>Fin RAG</title>");
    expect(indexHtml).toContain('<link rel="icon" href="data:," />');
    expect(indexHtml).not.toMatch(/href=["']\/?favicon/i);
  });

  test("loads ready state through the selected knowledge base scope", async () => {
    const fetchMock = mockInitialLoad();

    render(<App />);

    await screen.findByText("文档就绪，随时提问");

    const requestedUrls = fetchMock.mock.calls.map(([url]) => String(url));
    expect(requestedUrls).toContain("/knowledge-bases/finance/ready");
    expect(requestedUrls).not.toContain("/ready");
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
        new Response(JSON.stringify(knowledgeBasesPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
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
      .mockResolvedValueOnce(jsonResponse(readyPayload))
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
    const summaryRows = dialog.querySelectorAll(".upload-summary-row");
    expect(summaryRows).toHaveLength(1);
    expect(within(summaryRows[0] as HTMLElement).getByText("文件")).toBeInTheDocument();
    expect(within(summaryRows[0] as HTMLElement).getByText("大小")).toBeInTheDocument();
    expect(within(summaryRows[0] as HTMLElement).getByText("0.0 KB")).toBeInTheDocument();
    expect(dialog.querySelector(".upload-close-button")).toBeNull();
    expect(dialog.querySelector(".modal-action-button")).toBeNull();
    expect(
      within(dialog).queryByRole("textbox", { name: "知识库" }),
    ).not.toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "取消" })).toHaveClass(
      "confirm-btn",
      "cancel",
    );
    expect(within(dialog).getByRole("button", { name: "确认" })).toHaveClass(
      "confirm-btn",
      "primary",
    );

    await userEvent.click(within(dialog).getByRole("button", { name: "确认" }));

    await waitFor(() => {
      expect(
        screen.queryByRole("dialog", { name: "确认索引文档" }),
      ).not.toBeInTheDocument();
    });
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/knowledge-bases/finance/documents/upload",
        expect.objectContaining({ method: "POST", body: expect.any(FormData) }),
      );
    });
    const uploadCall = fetchMock.mock.calls.find(
      ([url]) => url === "/knowledge-bases/finance/documents/upload",
    );
    expect(
      ((uploadCall?.[1] as RequestInit).body as FormData).get(
        "knowledge_base_id",
      ),
    ).toBeNull();
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

  test("switches knowledge base from the sidebar dropdown and refreshes documents", async () => {
    const riskDocumentsPayload = {
      documents: [
        {
          document_id: "doc-risk",
          filename: "风险制度.md",
          file_type: "md",
          knowledge_base_id: "risk",
          status: "indexed",
          chunk_count: 2,
          upload_time: "2026-06-18T08:00:00.000000+00:00",
          last_error: null,
        },
      ],
    };
    const fetchMock = mockInitialLoad().mockResolvedValueOnce(
      new Response(JSON.stringify(riskDocumentsPayload), {
        headers: { "Content-Type": "application/json" },
      }),
    ).mockResolvedValueOnce(jsonResponse(readyPayload));

    render(<App />);

    await screen.findByText("文档就绪，随时提问");
    await userEvent.click(screen.getByRole("button", { name: "切换知识库 finance" }));
    await userEvent.click(screen.getByRole("option", { name: "risk" }));
    await userEvent.click(screen.getByRole("button", { name: /文档/ }));

    expect(await screen.findByText("风险制度.md")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "切换知识库 risk" })).toBeInTheDocument();
    expect(
      fetchMock.mock.calls.some(
        ([url]) => url === "/knowledge-bases/risk/documents",
      ),
    ).toBe(true);
  });

  test("warms the selected knowledge base after initial load and switch", async () => {
    const riskDocumentsPayload = {
      documents: [
        {
          document_id: "doc-risk",
          filename: "风险制度.md",
          file_type: "md",
          knowledge_base_id: "risk",
          status: "indexed",
          chunk_count: 2,
          upload_time: "2026-06-18T08:00:00.000000+00:00",
          last_error: null,
        },
      ],
    };
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify(knowledgeBasesPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
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
        new Response(JSON.stringify(readyPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(riskDocumentsPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(readyPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      );

    render(<App />);

    await screen.findByText("文档就绪，随时提问");
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/knowledge-bases/finance/warmup",
        expect.objectContaining({ method: "POST" }),
      );
    });

    await userEvent.click(
      screen.getByRole("button", { name: "切换知识库 finance" }),
    );
    await userEvent.click(screen.getByRole("option", { name: "risk" }));
    await userEvent.click(screen.getByRole("button", { name: /文档/ }));

    expect(await screen.findByText("风险制度.md")).toBeInTheDocument();
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/knowledge-bases/risk/warmup",
        expect.objectContaining({ method: "POST" }),
      );
    });
  });

  test("manually warms the current knowledge base from the documents toolbar", async () => {
    const fetchMock = mockInitialLoad()
      .mockResolvedValueOnce(
        jsonResponse({
          ...readyPayload,
          total_documents: 1,
          total_chunks: 9,
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          documents: [
            {
              ...documentsPayload.documents[0],
              chunk_count: 9,
            },
          ],
        }),
      );

    render(<App />);

    await screen.findByText("文档就绪，随时提问");
    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter(
          ([url]) => url === "/knowledge-bases/finance/warmup",
        ),
      ).toHaveLength(1);
    });
    await userEvent.click(screen.getByRole("button", { name: /文档/ }));

    await userEvent.click(
      screen.getByRole("button", { name: "刷新知识库" }),
    );
    const warmupMessage = await screen.findByText("确定要刷新 Finance 吗？");
    const warmupDialog = warmupMessage.closest(".confirm-dialog") as HTMLElement;
    await userEvent.click(within(warmupDialog).getByRole("button", { name: "刷新" }));

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter(
          ([url]) => url === "/knowledge-bases/finance/warmup",
        ),
      ).toHaveLength(2);
    });
    const stats = document.querySelector(".kb-stats") as HTMLElement;
    expect(within(stats).getByText("9")).toHaveClass("kb-stat-value");
  });

  test("confirms and starts a full rebuild for the current knowledge base", async () => {
    const fetchMock = mockInitialLoad()
      .mockResolvedValueOnce(
        jsonResponse({
          job_id: "job-finance",
          knowledge_base_id: "finance",
          status: "succeeded",
          created_at: "2026-06-20T00:00:00+00:00",
          started_at: "2026-06-20T00:00:00+00:00",
          completed_at: "2026-06-20T00:00:01+00:00",
          error: null,
          result: {
            document_count: 1,
            chunk_count: 11,
            manifest_schema_version: 1,
          },
        }),
      )
      .mockResolvedValueOnce(jsonResponse(knowledgeBasesPayload))
      .mockResolvedValueOnce(
        jsonResponse({
          ...readyPayload,
          total_documents: 1,
          total_chunks: 11,
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          documents: [
            {
              ...documentsPayload.documents[0],
              chunk_count: 11,
            },
          ],
        }),
      );

    render(<App />);

    await screen.findByText("文档就绪，随时提问");
    await userEvent.click(screen.getByRole("button", { name: /文档/ }));
    expect(
      screen.queryByRole("menuitem", { name: "重建知识库" }),
    ).not.toBeInTheDocument();
    await userEvent.click(
      screen.getByRole("button", { name: "更多操作" }),
    );
    await userEvent.click(
      screen.getByRole("menuitem", { name: "重建知识库" }),
    );

    const message = await screen.findByText("确定要全量重建 Finance 吗？");
    const dialog = message.closest(".confirm-dialog") as HTMLElement;
    expect(within(dialog).getByRole("button", { name: "取消" })).toHaveClass(
      "confirm-btn",
      "cancel",
    );
    expect(within(dialog).getByRole("button", { name: "重建" })).toHaveClass(
      "confirm-btn",
      "dark",
    );

    await userEvent.click(within(dialog).getByRole("button", { name: "重建" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/knowledge-bases/finance/rebuilds",
        expect.objectContaining({ method: "POST" }),
      );
    });
    const stats = document.querySelector(".kb-stats") as HTMLElement;
    expect(await within(stats).findByText("11")).toHaveClass("kb-stat-value");
  });

  test("archives and restores the current knowledge base from the more menu", async () => {
    const archivedRisk = {
      ...knowledgeBasesPayload.knowledge_bases[1],
      status: "archived",
      archived_at: "2026-06-18T00:03:00+00:00",
    };
    const activeRisk = {
      ...knowledgeBasesPayload.knowledge_bases[1],
      status: "active",
      archived_at: null,
    };
    const archivedKnowledgeBases = {
      knowledge_bases: [knowledgeBasesPayload.knowledge_bases[0], archivedRisk],
    };
    const restoredKnowledgeBases = {
      knowledge_bases: [knowledgeBasesPayload.knowledge_bases[0], activeRisk],
    };
    const fetchMock = mockInitialLoad()
      .mockResolvedValueOnce(jsonResponse({ documents: [] }))
      .mockResolvedValueOnce(jsonResponse(readyPayload))
      .mockResolvedValueOnce(jsonResponse(archivedRisk))
      .mockResolvedValueOnce(jsonResponse(archivedKnowledgeBases))
      .mockResolvedValueOnce(jsonResponse(activeRisk))
      .mockResolvedValueOnce(jsonResponse(restoredKnowledgeBases))
      .mockResolvedValueOnce(jsonResponse(readyPayload))
      .mockResolvedValueOnce(jsonResponse({ documents: [] }));

    render(<App />);

    await screen.findByText("文档就绪，随时提问");
    await userEvent.click(screen.getByRole("button", { name: "切换知识库 finance" }));
    await userEvent.click(screen.getByRole("option", { name: "risk" }));
    await userEvent.click(screen.getByRole("button", { name: /文档/ }));
    await userEvent.click(screen.getByRole("button", { name: "更多操作" }));
    await userEvent.click(screen.getByRole("menuitem", { name: "归档知识库" }));
    await userEvent.click(screen.getByRole("button", { name: "归档" }));

    expect(fetchMock).toHaveBeenCalledWith(
      "/knowledge-bases/risk/archive",
      expect.objectContaining({ method: "POST" }),
    );
    expect(await screen.findByText("已归档")).toHaveClass("kb-status-pill");
    expect(screen.getByRole("button", { name: "刷新知识库" })).toBeDisabled();

    await userEvent.click(screen.getByRole("button", { name: "更多操作" }));
    await userEvent.click(screen.getByRole("menuitem", { name: "恢复知识库" }));
    await userEvent.click(screen.getByRole("button", { name: "恢复" }));

    expect(fetchMock).toHaveBeenCalledWith(
      "/knowledge-bases/risk/restore",
      expect.objectContaining({ method: "POST" }),
    );
    await waitFor(() => {
      expect(screen.queryByText("已归档")).not.toBeInTheDocument();
    });
  });

  test("deletes the current knowledge base and falls back to finance", async () => {
    const deletedRisk = {
      ...knowledgeBasesPayload.knowledge_bases[1],
      status: "deleted",
      deleted_at: "2026-06-18T00:04:00+00:00",
    };
    const financeOnlyKnowledgeBases = {
      knowledge_bases: [knowledgeBasesPayload.knowledge_bases[0]],
    };
    const fetchMock = mockInitialLoad()
      .mockResolvedValueOnce(jsonResponse({ documents: [] }))
      .mockResolvedValueOnce(jsonResponse(readyPayload))
      .mockResolvedValueOnce(jsonResponse(deletedRisk))
      .mockResolvedValueOnce(jsonResponse(financeOnlyKnowledgeBases))
      .mockResolvedValueOnce(jsonResponse(readyPayload))
      .mockResolvedValueOnce(jsonResponse(documentsPayload))
      .mockResolvedValueOnce(jsonResponse(readyPayload));

    render(<App />);

    await screen.findByText("文档就绪，随时提问");
    await userEvent.click(screen.getByRole("button", { name: "切换知识库 finance" }));
    await userEvent.click(screen.getByRole("option", { name: "risk" }));
    await userEvent.click(screen.getByRole("button", { name: /文档/ }));
    await userEvent.click(screen.getByRole("button", { name: "更多操作" }));
    await userEvent.click(screen.getByRole("menuitem", { name: "删除知识库" }));
    await userEvent.click(screen.getByRole("button", { name: "删除" }));

    expect(fetchMock).toHaveBeenCalledWith(
      "/knowledge-bases/risk",
      expect.objectContaining({ method: "DELETE" }),
    );
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "切换知识库 finance" })).toBeInTheDocument();
    });
  });

  test("places refresh before the more knowledge base actions", async () => {
    mockInitialLoad();

    const { container } = render(<App />);

    await screen.findByText("文档就绪，随时提问");
    await userEvent.click(screen.getByRole("button", { name: /文档/ }));

    const stats = container.querySelector(".kb-stats") as HTMLElement;
    const createButton = within(stats).getByRole("button", {
      name: "新建知识库",
    });
    const refreshButton = within(stats).getByRole("button", {
      name: "刷新知识库",
    });
    const moreButton = within(stats).getByRole("button", {
      name: "更多操作",
    });

    expect(createButton).toHaveClass("docs-create-button");
    expect(createButton).not.toHaveTextContent("新建知识库");
    expect(moreButton).toHaveClass("kb-more-button");
    expect(moreButton).not.toHaveTextContent("全量重建知识库");
    expect(
      createButton.compareDocumentPosition(refreshButton) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      refreshButton.compareDocumentPosition(moreButton) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  test("closes the knowledge base dropdown when clicking outside it", async () => {
    mockInitialLoad();

    render(<App />);

    await screen.findByText("文档就绪，随时提问");
    await userEvent.click(
      screen.getByRole("button", { name: "切换知识库 finance" }),
    );

    expect(screen.getByRole("listbox", { name: "知识库列表" })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "新聊天" }));

    expect(
      screen.queryByRole("listbox", { name: "知识库列表" }),
    ).not.toBeInTheDocument();
  });

  test("creates a knowledge base from the documents page and switches to it", async () => {
    const creditPayload = {
      knowledge_base_id: "credit",
      document_count: 0,
      status: "active",
      created_at: "2026-06-18T00:02:00+00:00",
      updated_at: "2026-06-18T00:02:00+00:00",
      archived_at: null,
      deleted_at: null,
    };
    const updatedKnowledgeBases = {
      knowledge_bases: [...knowledgeBasesPayload.knowledge_bases, creditPayload],
    };
    const fetchMock = mockInitialLoad()
      .mockResolvedValueOnce(
        new Response(JSON.stringify(creditPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(updatedKnowledgeBases), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ documents: [] }), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(jsonResponse(readyPayload));

    render(<App />);

    await screen.findByText("文档就绪，随时提问");
    await userEvent.click(screen.getByRole("button", { name: /文档/ }));
    await userEvent.click(screen.getByRole("button", { name: "新建知识库" }));
    const dialog = await screen.findByRole("dialog", { name: "新建知识库" });
    expect(dialog.querySelector(".create-kb-header svg")).toBeNull();
    expect(within(dialog).queryByText("知识库 ID")).not.toBeInTheDocument();
    await userEvent.type(within(dialog).getByRole("textbox", { name: "知识库 ID" }), "credit");
    await userEvent.click(within(dialog).getByRole("button", { name: "创建" }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "切换知识库 credit" })).toBeInTheDocument();
    });
    expect(fetchMock).toHaveBeenCalledWith(
      "/knowledge-bases",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ knowledge_base_id: "credit" }),
      }),
    );
    expect(
      fetchMock.mock.calls.some(
        ([url]) => url === "/knowledge-bases/credit/documents",
      ),
    ).toBe(true);
  });

  test("asks questions against the selected knowledge base", async () => {
    let askBody = "";
    mockInitialLoad()
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ documents: [] }), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(jsonResponse(readyPayload))
      .mockImplementationOnce((_input, init) => {
        askBody = String(init?.body ?? "");
        return Promise.resolve(
          sseResponse([
            `event: done\ndata: {"response":${JSON.stringify({
              ...answerPayload,
              trace: {
                ...answerPayload.trace,
                filters: { knowledge_base_id: "risk" },
              },
            })}}\n\n`,
          ]),
        );
      });

    render(<App />);

    await screen.findByText("文档就绪，随时提问");
    await userEvent.click(screen.getByRole("button", { name: "切换知识库 finance" }));
    await userEvent.click(screen.getByRole("option", { name: "risk" }));
    await userEvent.type(screen.getByLabelText("问题"), "客户风险等级如何匹配？");
    await userEvent.click(screen.getByRole("button", { name: "提交问题" }));

    await screen.findByText("客户风险等级应与产品风险等级匹配。[1]");
    expect(JSON.parse(askBody)).toEqual({
      question: "客户风险等级如何匹配？",
      return_sources: true,
      return_trace: true,
    });
  });

  test("submits a question from the compact composer and renders answer, sources, and timeline without latency text", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify(knowledgeBasesPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
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
      .mockResolvedValueOnce(jsonResponse(readyPayload))
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
        new Response(JSON.stringify(knowledgeBasesPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
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
      .mockResolvedValueOnce(jsonResponse(readyPayload))
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

    const { container } = render(<App />);

    await screen.findByText("文档就绪，随时提问");

    await userEvent.click(screen.getByRole("button", { name: /文档/ }));

    const stats = container.querySelector(".kb-stats") as HTMLElement;
    expect(within(stats).getByText("Finance")).toHaveClass("kb-stat-value");
    expect(within(stats).getByText("文档")).toHaveClass("kb-stat-label");
    expect(within(stats).getByText("总分块")).toHaveClass("kb-stat-label");
    expect(within(stats).getByText("更新")).toHaveClass("kb-stat-label");
    expect(within(stats).getByText(/2026-06-18/)).toHaveClass("kb-stat-value");
    expect(within(stats).getByText("1")).toHaveClass("kb-stat-value");
    expect(within(stats).getByText("3")).toHaveClass("kb-stat-value");
    expect(within(stats).getAllByText("|")[0]).toHaveClass("kb-stat-divider");
    expect(
      screen.getByRole("button", { name: "重新索引" }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "删除" })).toBeInTheDocument();
  });

  test("new chat button clears the conversation", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify(knowledgeBasesPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
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
      .mockResolvedValueOnce(jsonResponse(readyPayload))
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
          knowledge_base_id: "finance",
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
          knowledge_base_id: "finance",
          status: "indexed",
          chunk_count: 5,
          upload_time: "2026-06-14T13:35:37.897082+00:00",
          last_error: null,
        },
      ],
    };

    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify(knowledgeBasesPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
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
      .mockResolvedValueOnce(jsonResponse(readyPayload))
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
          knowledge_base_id: "finance",
          status: "failed",
          chunk_count: 0,
          upload_time: "2026-06-15T10:00:00.000000+00:00",
          last_error: "PDF 解析失败：文件已损坏",
        },
      ],
    };

    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify(knowledgeBasesPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(readyPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(failedPayload), {
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(jsonResponse(readyPayload));

    render(<App />);

    await screen.findByText("文档就绪，随时提问");

    await userEvent.click(screen.getByRole("button", { name: /文档/ }));

    expect(screen.getByText(/索引失败/)).toBeInTheDocument();
    expect(screen.getByText("PDF 解析失败：文件已损坏")).toBeInTheDocument();
  });
});
