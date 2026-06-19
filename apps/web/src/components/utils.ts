import type { ApiError } from "../api/client";

export function errorMessage(error: unknown, fallback: string) {
  const apiError = error as ApiError;
  if (apiError?.message) {
    return apiError.request_id
      ? `${apiError.message} Request ID: ${apiError.request_id}`
      : apiError.message;
  }
  return fallback;
}

export function displayName(id: string) {
  if (/^[a-z]/.test(id)) {
    return id.charAt(0).toUpperCase() + id.slice(1);
  }
  return id;
}

export function formatFileSize(bytes: number) {
  return `${(bytes / 1024).toFixed(1)} KB`;
}

export function formatRetrievalScore(score: number | null) {
  if (score === null || Number.isNaN(score)) {
    return "检索分数：-";
  }
  return `检索分数：${score.toFixed(2)}`;
}

export function formatUploadTime(iso: string | undefined) {
  if (!iso) return "";
  const date = new Date(iso);
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  const h = String(date.getHours()).padStart(2, "0");
  const min = String(date.getMinutes()).padStart(2, "0");
  return `${y}-${m}-${d} ${h}:${min}`;
}

export function splitFilename(filename: string) {
  const extensionStart = filename.lastIndexOf(".");
  if (extensionStart <= 0 || extensionStart === filename.length - 1) {
    return { stem: filename, extension: "" };
  }
  return {
    stem: filename.slice(0, extensionStart),
    extension: filename.slice(extensionStart),
  };
}
