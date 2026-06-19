import type { FormEvent } from "react";
import { Send, Square } from "lucide-react";

type QuestionComposerProps = {
  answerBusy: boolean;
  floating: boolean;
  onAbort: () => void;
  onQuestionChange: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  question: string;
  submitDisabled: boolean;
};

export function QuestionComposer({
  answerBusy,
  floating,
  onAbort,
  onQuestionChange,
  onSubmit,
  question,
  submitDisabled,
}: QuestionComposerProps) {
  return (
    <form
      className={`chat-composer-form${floating ? " is-floating" : " is-docked"}`}
      onSubmit={onSubmit}
    >
      <div className="chat-composer">
        <textarea
          aria-label="问题"
          className="chat-input"
          maxLength={100}
          rows={1}
          value={question}
          onChange={(event) => onQuestionChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              if (!answerBusy && !submitDisabled) {
                (event.target as HTMLTextAreaElement).form?.requestSubmit();
              }
            }
          }}
          placeholder="查询文档信息、定位条款、总结内容"
        />
        <button
          aria-label={answerBusy ? "中断回答" : "提交问题"}
          className={`primary-button answer-action-button${answerBusy ? " is-stop" : ""}`}
          disabled={!answerBusy && submitDisabled}
          title={answerBusy ? "中断回答" : "提交问题"}
          type={answerBusy ? "button" : "submit"}
          onClick={answerBusy ? onAbort : undefined}
        >
          {answerBusy ? <Square size={17} /> : <Send size={18} />}
        </button>
      </div>
    </form>
  );
}
