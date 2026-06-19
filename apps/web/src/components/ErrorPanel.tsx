import { AlertCircle } from "lucide-react";

export function ErrorPanel({ message }: { message: string }) {
  return (
    <section className="error-panel" aria-live="polite">
      <AlertCircle size={18} />
      <span>{message}</span>
    </section>
  );
}
