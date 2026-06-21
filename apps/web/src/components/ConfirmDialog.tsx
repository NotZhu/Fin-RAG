export type ConfirmAction =
  | {
      type: "reindex" | "delete";
      documentId: string;
      filename: string;
    }
  | {
      type:
        | "rebuild"
        | "archiveKnowledgeBase"
        | "restoreKnowledgeBase"
        | "deleteKnowledgeBase"
        | "warmup";
      knowledgeBaseName: string;
    };

type ConfirmDialogProps = {
  action: ConfirmAction;
  onConfirm: () => void;
  onCancel: () => void;
};

export function ConfirmDialog({
  action,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  let message = "";
  let confirmText = "";
  let confirmTone: "danger" | "primary" | "dark" = "primary";

  switch (action.type) {
    case "rebuild":
      message = `确定要全量重建 ${action.knowledgeBaseName} 吗？`;
      confirmText = "重建";
      confirmTone = "dark";
      break;
    case "warmup":
      message = `确定要刷新 ${action.knowledgeBaseName} 吗？`;
      confirmText = "刷新";
      confirmTone = "primary";
      break;
    case "archiveKnowledgeBase":
      message = `确定要归档 ${action.knowledgeBaseName} 吗？`;
      confirmText = "归档";
      confirmTone = "dark";
      break;
    case "restoreKnowledgeBase":
      message = `确定要恢复 ${action.knowledgeBaseName} 吗？`;
      confirmText = "恢复";
      break;
    case "deleteKnowledgeBase":
      message = `确定要删除 ${action.knowledgeBaseName} 吗？`;
      confirmText = "删除";
      confirmTone = "danger";
      break;
    case "delete":
      message = `确定要删除「${action.filename}」吗？此操作不可撤销。`;
      confirmText = "删除";
      confirmTone = "danger";
      break;
    case "reindex":
      message = `确定要重新索引「${action.filename}」吗？`;
      confirmText = "重新索引";
      break;
  }

  return (
    <div className="confirm-overlay" onClick={onCancel}>
      <div className="confirm-dialog" onClick={(e) => e.stopPropagation()}>
        <p className="confirm-message">{message}</p>
        <div className="confirm-actions">
          <button
            className="confirm-btn cancel"
            type="button"
            onClick={onCancel}
          >
            取消
          </button>
          <button
            className={`confirm-btn ${confirmTone}`}
            type="button"
            onClick={onConfirm}
          >
            {confirmText}
          </button>
        </div>
      </div>
    </div>
  );
}
