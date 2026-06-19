import { FileSearch } from "lucide-react";

import type { AskResponse } from "../api/client";
import { formatRetrievalScore } from "./utils";

export function SourcesPane({ response }: { response: AskResponse | null }) {
  if (!response?.sources.length) {
    return null;
  }
  return (
    <div className="sources-list">
      {response.sources.map((source) => (
        <article
          className="source-card"
          key={`${source.source_id}-${source.filename}`}
        >
          <div className="source-title">
            <FileSearch size={16} />
            <span className="source-citation">证据块 [{source.source_id}]</span>
            <span>{source.filename}</span>
          </div>
          <div className="source-meta">
            <span>{formatRetrievalScore(source.score)}</span>
            {source.page_number === null ? null : (
              <span>第 {source.page_number} 页</span>
            )}
          </div>
          <p>{source.snippet}</p>
        </article>
      ))}
    </div>
  );
}
