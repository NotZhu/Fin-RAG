import {
  Activity,
  Database,
  FileText,
  GitMerge,
  Loader2,
  MessageSquareText,
  Route,
  Search,
} from "lucide-react";

import type { PipelineStep } from "../api/client";

type RetrievalTimelineProps = {
  collapsed: boolean;
  steps: PipelineStep[];
  timingsMs: Record<string, number>;
};

const TIMINGS_LABELS: Record<string, string> = {
  analysis: "分析",
  total: "总计",
};
const STATUS_LABELS: Record<PipelineStep["status"], string> = {
  running: "运行中",
  complete: "完成",
  skipped: "跳过",
  error: "失败",
};
const STEP_LABELS: Record<string, string> = {
  query_router: "请求路由分发",
  knowledge_engine: "知识库检索引擎",
  hybrid_search: "多路混合召回",
  ranking_postprocess: "精排后处理",
  context_expansion: "上下文扩展",
  evidence_window: "证据窗口",
  streaming_answer: "回答生成",
};
const ROUTE_LABELS: Record<string, string> = {
  knowledge_router: "知识库路由",
  knowledge: "知识库路由",
  general_router: "通用路由",
  general: "通用路由",
};
const KNOWLEDGE_ENGINE_LABELS: Record<string, string> = {
  auto_merge: "自动合并引擎",
  knowledge_router: "自动合并引擎",
  hyde: "HyDE 查询引擎",
  step_back: "后退一步查询引擎",
};

function formatMs(ms: number): string {
  if (ms < 0.1) return "<0.1 毫秒";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} 秒` : `${ms.toFixed(1)} 毫秒`;
}

function metaString(step: PipelineStep, key: string): string {
  const value = step.meta[key];
  return value == null ? "" : String(value);
}

function metaText(step: PipelineStep, key: string): string {
  const value = metaString(step, key);
  return value && value !== "none" ? value : "";
}

function stepLabel(step: PipelineStep): string {
  if (step.id === "hybrid_search" && metaString(step, "hybrid_mode") === "dense_only") {
    return "Dense 检索";
  }
  return STEP_LABELS[step.id] ?? step.label;
}

function stepDetail(step: PipelineStep): string {
  if (step.detail) return step.detail;
  if (
    step.status === "skipped" &&
    ["hybrid_search", "ranking_postprocess", "context_expansion"].includes(step.id)
  ) {
    return "";
  }
  if (step.id === "query_router") {
    const selected = metaText(step, "selected_query_engine") || metaText(step, "route_type");
    return selected ? `选中：${ROUTE_LABELS[selected] ?? selected}` : "选中：判断中";
  }
  if (step.id === "knowledge_engine") {
    if (metaString(step, "route_type") === "general") return "选择：通用路由";
    const selected = metaText(step, "selected_knowledge_engine") || "auto_merge";
    return `选择：${KNOWLEDGE_ENGINE_LABELS[selected] ?? selected}`;
  }
  if (step.id === "hybrid_search") {
    if (metaString(step, "hybrid_mode") === "dense_only") {
      const candidateK = metaString(step, "candidate_k");
      const topK = metaString(step, "top_k");
      return [
        "Dense 向量召回",
        [
          candidateK && `候选集数量 ${candidateK}`,
          topK && `最终召回条数 ${topK}`,
        ].filter(Boolean).join("・"),
      ].filter(Boolean).join("\n");
    }
    const provider = metaText(step, "hybrid_provider");
    const ranker = metaText(step, "hybrid_ranker");
    const candidateK = metaString(step, "candidate_k");
    const topK = metaString(step, "top_k");
    const rrfK = metaString(step, "rrf_k");
    const header = provider || ranker ? `密疏混合・${ranker || provider}` : "";
    const params = [
      candidateK && `候选集数量 ${candidateK}`,
      topK && `最终召回条数 ${topK}`,
      rrfK && `融合参数 ${rrfK}`,
    ].filter(Boolean);
    return [header, params.join("・")].filter(Boolean).join("\n");
  }
  if (step.id === "ranking_postprocess") {
    const threshold = Number(step.meta.score_threshold ?? 0);
    const rerankerProvider = metaString(step, "reranker_provider");
    const rerankerTopN = metaString(step, "reranker_top_n");
    const thresholdText = Number.isInteger(threshold)
      ? threshold.toFixed(0)
      : String(threshold);
    const similarity = threshold > 0 ? `阈值 ${thresholdText}` : "未启用";
    const reranker =
      rerankerProvider && rerankerProvider !== "none"
        ? `${rerankerProvider.toUpperCase()} 取前 ${rerankerTopN || "-"} 条`
        : "未启用";
    return [
      `相似度过滤器：${similarity}`,
      `重排模型：${reranker}`,
      `前后片段扩展：${metaString(step, "prev_next") || "-"}`,
      `Token 上限：${metaString(step, "context_token_budget") || "-"}`,
    ].join("\n");
  }
  if (step.id === "context_expansion") {
    const threshold = metaString(step, "simple_ratio_thresh");
    return threshold ? `自动合并・阈值 ${threshold}` : "自动合并";
  }
  if (step.id === "evidence_window") {
    const sourceCount = metaString(step, "source_count") || "0";
    const evidenceCount = metaString(step, "evidence_count") || "0";
    return `${sourceCount} 个信息来源・${evidenceCount} 条证据片段`;
  }
  if (step.id === "streaming_answer") {
    const answerChars = metaString(step, "answer_chars");
    return answerChars ? `总字数 ${answerChars} 字` : "生成中...";
  }
  return "";
}

function PipelineIcon({ step }: { step: PipelineStep }) {
  if (step.id.includes("router")) return <Route size={16} />;
  if (step.id.includes("hybrid") || step.id.includes("search")) {
    return <Database size={16} />;
  }
  if (step.id.includes("merge")) return <GitMerge size={16} />;
  if (step.id.includes("evidence")) return <FileText size={16} />;
  if (step.id.includes("answer")) return <MessageSquareText size={16} />;
  return <Search size={16} />;
}

export function RetrievalTimeline({
  collapsed,
  steps,
  timingsMs,
}: RetrievalTimelineProps) {
  const hasTimings = Object.keys(timingsMs).length > 0;
  return (
    <section className="rail-card timeline-panel" aria-label="实时检索链路">
      <div className="rail-item rail-label">
        <Activity size={20} />
        <span>实时检索链路</span>
      </div>
      {!collapsed && (
        <div className="timeline-inner">
          {steps.length ? (
            <>
              <ol className="timeline-list">
                {steps.map((step) => (
                  <li
                    className={`timeline-item step-${step.status}`}
                    key={step.id}
                  >
                    <div className="step-body">
                      <div className="step-title">
                        <PipelineIcon step={step} />
                        <strong>{stepLabel(step)}</strong>
                      </div>
                      <span className="step-detail">{stepDetail(step)}</span>
                    </div>
                    <div className="step-status">
                      {step.status === "running" && (
                        <Loader2 className="spin" size={14} />
                      )}
                      <span className="step-state-label">
                        {STATUS_LABELS[step.status]}
                      </span>
                      {step.duration_ms != null && (
                        <span className="step-duration">
                          ・{formatMs(step.duration_ms)}
                        </span>
                      )}
                    </div>
                  </li>
                ))}
              </ol>
              {hasTimings && (
                <div className="timings-section">
                  {Object.entries(timingsMs).map(([key, value]) => (
                    <div className="timings-row" key={key}>
                      <span className="timings-label">
                        {TIMINGS_LABELS[key] ?? key}
                      </span>
                      <span className="timings-value">{formatMs(value)}</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <div className="timeline-empty" aria-hidden="true">
              <Search size={24} />
            </div>
          )}
        </div>
      )}
    </section>
  );
}
