export type ConfirmAction =
  | {
      type: "reindex" | "delete";
      documentId: string;
      filename: string;
    }
  | {
      type: "rebuild";
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
  const isDelete = action.type === "delete";
  const isRebuild = action.type === "rebuild";
  const message = isRebuild
    ? `确定要全量重建 ${action.knowledgeBaseName} 吗？`
    : isDelete
      ? `确定要删除「${action.filename}」吗？此操作不可撤销。`
      : `确定要重新索引「${action.filename}」吗？`;
  const confirmText = isRebuild ? "重建" : isDelete ? "删除" : "重新索引";
  const confirmTone = isDelete || isRebuild ? "danger" : "primary";

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
