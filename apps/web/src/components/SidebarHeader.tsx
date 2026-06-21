import { useEffect, useRef, useState } from "react";
import {
  Check,
  ChevronDown,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";

import type { KnowledgeBaseRecord } from "../api/client";
import { displayName } from "./utils";

export type KnowledgeBaseLoadState = "loading" | "ready" | "error";

type SidebarHeaderProps = {
  collapsed: boolean;
  knowledgeBaseId: string;
  knowledgeBaseLoadError: string;
  knowledgeBaseLoadState: KnowledgeBaseLoadState;
  knowledgeBases: KnowledgeBaseRecord[];
  onKnowledgeBaseChange: (knowledgeBaseId: string) => void;
  onRetryKnowledgeBases: () => void;
  onToggle: () => void;
};

export function SidebarHeader({
  collapsed,
  knowledgeBaseId,
  knowledgeBaseLoadError,
  knowledgeBaseLoadState,
  knowledgeBases,
  onKnowledgeBaseChange,
  onRetryKnowledgeBases,
  onToggle,
}: SidebarHeaderProps) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const switcherRef = useRef<HTMLDivElement | null>(null);
  const options = knowledgeBases;
  const isLoading = knowledgeBaseLoadState === "loading";
  const isError = knowledgeBaseLoadState === "error";
  const canOpenMenu = knowledgeBaseLoadState === "ready" && options.length > 0;
  const buttonText = isLoading
    ? "加载中"
    : isError
      ? "加载失败"
      : options.length
        ? displayName(knowledgeBaseId)
        : "暂无知识库";
  const buttonLabel = isLoading
    ? "知识库加载中"
    : isError
      ? "重新加载知识库"
      : options.length
        ? `切换知识库 ${knowledgeBaseId}`
        : "暂无知识库";
  const buttonTitle = isError
    ? knowledgeBaseLoadError || "重新加载知识库"
    : canOpenMenu
      ? "切换知识库"
      : buttonText;

  useEffect(() => {
    if (collapsed || !canOpenMenu) {
      setIsMenuOpen(false);
    }
  }, [collapsed, canOpenMenu]);

  useEffect(() => {
    if (!isMenuOpen || !canOpenMenu) {
      return;
    }

    function handlePointerDown(event: PointerEvent) {
      if (!switcherRef.current?.contains(event.target as Node)) {
        setIsMenuOpen(false);
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setIsMenuOpen(false);
      }
    }

    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [canOpenMenu, isMenuOpen]);

  return (
    <div className="rail-header">
      {!collapsed && <span className="rail-title">知识库问答</span>}
      {!collapsed && (
        <div className="kb-switcher" ref={switcherRef}>
          <button
            aria-expanded={canOpenMenu ? isMenuOpen : undefined}
            aria-haspopup={canOpenMenu ? "listbox" : undefined}
            aria-label={buttonLabel}
            className={`kb-switch-button${canOpenMenu ? "" : " is-status"}`}
            disabled={isLoading || (!isError && !canOpenMenu)}
            title={buttonTitle}
            type="button"
            onClick={() => {
              if (isError) {
                onRetryKnowledgeBases();
                return;
              }
              if (canOpenMenu) {
                setIsMenuOpen((current) => !current);
              }
            }}
          >
            <span>{buttonText}</span>
            {canOpenMenu ? <ChevronDown size={16} /> : null}
          </button>
          {isMenuOpen && canOpenMenu ? (
            <div className="kb-menu" role="listbox" aria-label="知识库列表">
              {options.map((knowledgeBase) => {
                const selected =
                  knowledgeBase.knowledge_base_id === knowledgeBaseId;
                return (
                  <button
                    aria-label={knowledgeBase.knowledge_base_id}
                    aria-selected={selected}
                    className={`kb-menu-item${selected ? " is-selected" : ""}`}
                    key={knowledgeBase.knowledge_base_id}
                    role="option"
                    type="button"
                    onClick={() => {
                      setIsMenuOpen(false);
                      onKnowledgeBaseChange(knowledgeBase.knowledge_base_id);
                    }}
                  >
                    <span>
                      {knowledgeBase.knowledge_base_id}
                      {knowledgeBase.status === "archived" ? " (已归档)" : null}
                    </span>
                  </button>
                );
              })}
            </div>
          ) : null}
        </div>
      )}
      <div className="rail-actions">
        <button
          aria-label={collapsed ? "展开侧栏" : "收起侧栏"}
          className="rail-toggle"
          title={collapsed ? "展开侧栏" : "收起侧栏"}
          type="button"
          onClick={onToggle}
        >
          {collapsed ? (
            <PanelLeftOpen size={20} />
          ) : (
            <PanelLeftClose size={20} />
          )}
        </button>
      </div>
    </div>
  );
}
