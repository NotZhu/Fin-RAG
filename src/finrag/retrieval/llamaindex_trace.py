"""
知识库问答 LlamaIndex 回调轨迹处理程序
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.callbacks.base import BaseCallbackHandler


_EVENT_TYPE_KEY = "event_type"
_PHASE_KEY = "phase"
_START_TIME_KEY = "_start_time"
SUMMARY_WHITELIST = frozenset(
    {
        "query_str",
        "top_k",
        "node_count",
        "model_name",
        "exception_type",
        "response_mode",
    }
)


class FinRAGTraceHandler(BaseCallbackHandler):
    """LlamaIndex 回调轨迹处理程序"""

    MAX_EVENTS = 80

    def __init__(self) -> None:
        # 初始化父类，所有时间的 start/end 都不要忽略
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        # 保存已经完成的事件
        self._events: List[Dict[str, Any]] = []
        # 已经开始但还没结束的事件
        self._active: Dict[str, Dict[str, Any]] = {}

    @property
    def events(self) -> List[Dict[str, Any]]:
        # 如果事件数量超过最大限制，返回截断后的事件列表
        if len(self._events) > self.MAX_EVENTS:
            return self._events[: self.MAX_EVENTS] + [
                {_EVENT_TYPE_KEY: "truncated", "dropped_count": len(self._events) - self.MAX_EVENTS}
            ]
        return self._events

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        """
        处理开始事件
        Args:
            event_type: 事件类型
            payload: 事件有效负载
            event_id: 事件ID
            parent_id: 父事件ID
            kwargs: 其他关键字参数
        Returns:
            事件ID
        """
        entry: Dict[str, Any] = {
            _EVENT_TYPE_KEY: event_type.value,
            _PHASE_KEY: "start",
            "event_id": event_id,
            "parent_id": parent_id,
            _START_TIME_KEY: time.perf_counter(),
        }
        payload = payload or {}
        summary = _extract_summary(event_type, payload)
        if summary:
            entry["summary"] = summary
        self._active[event_id] = entry
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Dict[str, Any] | None = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> None:
        """
        处理结束事件
        Args:
            event_type: 事件类型
            payload: 事件有效负载
            event_id: 事件ID
            parent_id: 父事件ID
            kwargs: 其他关键字参数
        """
        entry = self._active.pop(event_id, None)
        if entry is None:
            entry = {
                _EVENT_TYPE_KEY: event_type.value,
                _PHASE_KEY: "end",
                "event_id": event_id,
                "parent_id": parent_id,
            }
        else:
            entry[_PHASE_KEY] = "end"
        start_time = entry.pop(_START_TIME_KEY, None)
        if start_time is not None:
            entry["duration_ms"] = round((time.perf_counter() - float(start_time)) * 1000, 2)
        payload = payload or {}
        summary = _extract_summary(event_type, payload)
        if summary:
            existing_summary = entry.get("summary") or {}
            existing_summary.update(summary)
            entry["summary"] = existing_summary
        self._events.append(entry)

    def start_trace(self, trace_id: str | None = None) -> None:
        """开始轨迹"""
        pass

    def end_trace(
        self,
        trace_id: str | None = None,
        trace_map: Dict[str, List[str]] | None = None,
    ) -> None:
        """结束轨迹"""
        pass


def _extract_summary(event_type: CBEventType, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    从 LlamaIndex 事件有效负载中提取摘要字段
    Args:
        event_type: 事件类型
        payload: 事件有效负载
    Returns:
        摘要字段字典
    """
    summary: Dict[str, Any] = {}
    if event_type == CBEventType.QUERY:
        if EventPayload.QUERY_STR in payload:
            query_str = str(payload.get(EventPayload.QUERY_STR) or "")
            if len(query_str) > 200:
                query_str = query_str[:197] + "..."
            summary["query_str"] = query_str
    elif event_type == CBEventType.RETRIEVE:
        nodes = payload.get(EventPayload.NODES, [])
        summary["node_count"] = len(list(nodes)) if nodes else 0
    elif event_type == CBEventType.SYNTHESIZE:
        if EventPayload.QUERY_STR in payload:
            summary["response_mode"] = "compact"
    elif event_type == CBEventType.LLM:
        if EventPayload.MODEL_NAME in payload:
            summary["model_name"] = str(payload.get(EventPayload.MODEL_NAME) or "")
    elif event_type == CBEventType.EXCEPTION:
        exc = payload.get(EventPayload.EXCEPTION)
        if exc is not None:
            summary["exception_type"] = exc.__class__.__name__
    for key in SUMMARY_WHITELIST:
        value = payload.get(key)
        if value is not None and key not in summary:
            if isinstance(value, (str, int, float, bool)):
                summary[key] = value
    return summary
