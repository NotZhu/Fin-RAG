import { useState, type FormEvent } from "react";

import { errorMessage } from "./utils";

type CreateKnowledgeBaseDialogProps = {
  onCancel: () => void;
  onCreate: (knowledgeBaseId: string) => Promise<void> | void;
};

export function CreateKnowledgeBaseDialog({
  onCancel,
  onCreate,
}: CreateKnowledgeBaseDialogProps) {
  const [knowledgeBaseId, setKnowledgeBaseId] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedId = knowledgeBaseId.trim();
    if (!trimmedId) {
      setError("请输入知识库 ID。");
      return;
    }
    setBusy(true);
    setError("");
    try {
      await onCreate(trimmedId);
      onCancel();
    } catch (caught) {
      setError(errorMessage(caught, "创建知识库失败。"));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="confirm-overlay" onClick={busy ? undefined : onCancel}>
      <form
        aria-labelledby="create-kb-title"
        aria-modal="true"
        className="create-kb-dialog"
        role="dialog"
        onClick={(event) => event.stopPropagation()}
        onSubmit={handleSubmit}
      >
        <div className="create-kb-header">
          <h3 id="create-kb-title">新建知识库</h3>
        </div>
        <label className="create-kb-field">
          <input
            aria-label="知识库 ID"
            autoFocus
            disabled={busy}
            value={knowledgeBaseId}
            onChange={(event) => setKnowledgeBaseId(event.target.value)}
          />
        </label>
        {error ? <p className="create-kb-error">{error}</p> : null}
        <div className="confirm-actions">
          <button
            className="confirm-btn cancel"
            disabled={busy}
            type="button"
            onClick={onCancel}
          >
            取消
          </button>
          <button className="confirm-btn primary" disabled={busy} type="submit">
            创建
          </button>
        </div>
      </form>
    </div>
  );
}
