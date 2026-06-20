import {
  useEffect,
  useRef,
  useState,
  type DragEvent,
  type KeyboardEvent,
} from "react";
import { CloudUpload, FileText } from "lucide-react";

import { formatFileSize, splitFilename } from "./utils";

type UploadPanelProps = {
  collapsed: boolean;
  selectedFile: File | null;
  uploadBusy: boolean;
  uploadDisabled: boolean;
  onSelectedFileChange: (value: File | null) => void;
  onUpload: () => void;
};

export function UploadPanel({
  collapsed,
  selectedFile,
  uploadBusy,
  uploadDisabled,
  onSelectedFileChange,
  onUpload,
}: UploadPanelProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const intakeDisabled = uploadBusy || uploadDisabled;
  const selectedFilename = selectedFile
    ? splitFilename(selectedFile.name)
    : null;

  function handleSelectedFile(file: File | null) {
    onSelectedFileChange(file);
    setIsConfirmOpen(Boolean(file));
  }

  function handleCancelUpload() {
    onSelectedFileChange(null);
    setIsConfirmOpen(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  function handleDragOver(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    if (intakeDisabled) {
      return;
    }
    setIsDragOver(true);
  }

  function handleDragLeave(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragOver(false);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragOver(false);
    if (intakeDisabled) {
      return;
    }
    handleSelectedFile(event.dataTransfer.files?.[0] ?? null);
  }

  function handleDropZoneKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (!intakeDisabled && (event.key === "Enter" || event.key === " ")) {
      fileInputRef.current?.click();
    }
  }

  useEffect(() => {
    if (!selectedFile) {
      setIsConfirmOpen(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }, [selectedFile]);

  return (
    <section className="rail-card upload-panel" aria-label="文件上传">
      <div className="rail-item rail-label" title="文件上传">
        <CloudUpload size={20} />
        <span>文件上传</span>
      </div>
      <div
        className={`drop-zone${isDragOver ? " is-drag-over" : ""}${selectedFile ? " has-file" : ""}`}
        role="button"
        tabIndex={collapsed || intakeDisabled ? -1 : 0}
        aria-disabled={intakeDisabled}
        onClick={() => {
          if (!intakeDisabled) {
            fileInputRef.current?.click();
          }
        }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onKeyDown={handleDropZoneKeyDown}
      >
        <FileText size={18} />
        <strong>{selectedFile ? selectedFile.name : "点击或拖入文件"}</strong>
        {selectedFile ? <span>{formatFileSize(selectedFile.size)}</span> : null}
      </div>
      <input
        ref={fileInputRef}
        className="visually-hidden"
        type="file"
        accept=".pdf,.docx,.md,.txt"
        aria-label="上传文件"
        onChange={(event) =>
          handleSelectedFile(event.target.files?.[0] ?? null)
        }
      />
      {selectedFile && isConfirmOpen ? (
        <div className="upload-confirm-backdrop">
          <div
            aria-labelledby="upload-confirm-title"
            aria-modal="true"
            className="upload-confirm-dialog"
            role="dialog"
          >
            <div className="upload-confirm-header">
              <FileText size={22} />
              <h3 id="upload-confirm-title">确认索引文档</h3>
            </div>
            <div className="upload-file-summary">
              <div className="upload-summary-row">
                <span>文件</span>
                <strong
                  aria-label={selectedFile.name}
                  className="upload-filename"
                  title={selectedFile.name}
                >
                  <span className="upload-filename-stem">
                    {selectedFilename?.stem}
                  </span>
                  {selectedFilename?.extension ? (
                    <span className="upload-filename-extension">
                      {selectedFilename.extension}
                    </span>
                  ) : null}
                </strong>
                <span>大小</span>
                <strong className="upload-summary-size">
                  {formatFileSize(selectedFile.size)}
                </strong>
              </div>
            </div>
            <div className="confirm-actions">
              <button
                className="confirm-btn cancel"
                type="button"
                onClick={handleCancelUpload}
                disabled={uploadBusy}
              >
                取消
              </button>
              <button
                className="confirm-btn primary"
                type="button"
                onClick={() => {
                  setIsConfirmOpen(false);
                  void onUpload();
                }}
                disabled={uploadBusy || uploadDisabled}
              >
                确认
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
