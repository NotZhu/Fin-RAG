import {
  Activity,
  Check,
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

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${ms.toFixed(1)}ms`;
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
                {steps.map((step, index) => (
                  <li
                    className={`timeline-item step-${step.status}`}
                    key={step.id}
                  >
                    <div className="step-number">{index + 1}</div>
                    <div className="step-body">
                      <div className="step-title">
                        <PipelineIcon step={step} />
                        <strong>{step.label}</strong>
                      </div>
                      <span>{step.detail}</span>
                    </div>
                    <div className="step-status">
                      {step.status === "running" ? (
                        <Loader2 className="spin" size={16} />
                      ) : step.duration_ms != null ? (
                        <span className="step-duration">
                          {formatMs(step.duration_ms)}
                        </span>
                      ) : (
                        <Check size={15} />
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
