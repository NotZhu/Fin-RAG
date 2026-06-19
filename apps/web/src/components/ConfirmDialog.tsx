export type ConfirmAction = {
  type: "reindex" | "delete";
  documentId: string;
  filename: string;
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
  return (
    <div className="confirm-overlay" onClick={onCancel}>
      <div className="confirm-dialog" onClick={(e) => e.stopPropagation()}>
        <p className="confirm-message">
          {isDelete
            ? `确定要删除「${action.filename}」吗？此操作不可撤销。`
            : `确定要重新索引「${action.filename}」吗？`}
        </p>
        <div className="confirm-actions">
          <button
            className="confirm-btn cancel"
            type="button"
            onClick={onCancel}
          >
            取消
          </button>
          <button
            className={`confirm-btn ${isDelete ? "danger" : "primary"}`}
            type="button"
            onClick={onConfirm}
          >
            {isDelete ? "删除" : "重新索引"}
          </button>
        </div>
      </div>
    </div>
  );
}
