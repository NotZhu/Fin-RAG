from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from finrag.core import FinRAGResponse, RAGTrace, RetrievedSource


class FakeFinRAGSystem:
    def __init__(self):
        self.initialize_calls = 0
        self.build_calls = 0
        self.ensure_calls = 0
        self.ask_calls = []
        self.retrieval_module = None
        self.config = SimpleNamespace(
            knowledge_base_id="kb-config-default",
            upload_dir="storage/uploads",
            max_upload_bytes=20 * 1024 * 1024,
        )
        self.data_module = SimpleNamespace(get_statistics=lambda: {"total_documents": 2, "total_chunks": 5})

    def initialize_system(self):
        self.initialize_calls += 1

    def build_knowledge_base(self):
        self.build_calls += 1
        self.retrieval_module = object()

    def ensure_knowledge_base_ready(self):
        self.ensure_calls += 1
        if self.retrieval_module is not None and self.data_module is not None:
            return
        self.initialize_system()
        self.build_knowledge_base()

    def ready(self):
        return {"ready": True, "status": "ready", "total_documents": 2, "total_chunks": 5, "last_error": None}

    def list_documents(self):
        return []

    def ask_question(self, question, **kwargs):
        assert "stream" not in kwargs
        self.ask_calls.append({"question": question, **kwargs})
        sources = [
            RetrievedSource(
                source_id=1,
                filename="policy.md",
                file_type="md",
                page_number=None,
                chunk_id="node-1",
                parent_chunk_id="parent-1",
                root_chunk_id="root-1",
                chunk_level=3,
                chunk_idx=0,
                score=0.9,
                snippet="客户风险等级应与产品风险等级匹配",
            )
        ]
        return FinRAGResponse(
            question=question,
            answer="客户风险等级应与产品风险等级匹配[1]",
            sources=sources if kwargs["return_sources"] else [],
            trace=RAGTrace(retrieval_strategy="llamaindex_router", route_type="knowledge", source_count=1) if kwargs["return_trace"] else None,
            retrieval_strategy="llamaindex_router",
            route_type="knowledge",
        )


class GeneralOnlySystem(FakeFinRAGSystem):
    def initialize_system(self):
        raise AssertionError("general 问答不应初始化知识库")

    def build_knowledge_base(self):
        raise AssertionError("general 问答不应构建知识库")

    def ask_question(self, question, **kwargs):
        self.ask_calls.append({"question": question, **kwargs})
        return FinRAGResponse(
            question=question,
            answer="普通问答回答",
            sources=[],
            trace=RAGTrace(
                retrieval_strategy="llamaindex_router",
                route_type="general",
                final_decision="general_answer",
            )
            if kwargs["return_trace"]
            else None,
            retrieval_strategy="llamaindex_router",
            route_type="general",
        )


def build_client(factory=FakeFinRAGSystem):
    from finrag.api import create_app

    systems = []

    def wrapped():
        system = factory()
        systems.append(system)
        return system

    return TestClient(create_app(system_factory=wrapped)), systems


def parse_sse_events(body: str):
    events = []
    for block in body.strip().split("\n\n"):
        if not block:
            continue
        event_name = "message"
        data = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        events.append((event_name, data))
    return events


def test_health_and_ready_do_not_initialize_until_needed():
    client, systems = build_client()

    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready").json()

    assert ready == {
        "ready": False,
        "status": "not_ready",
        "total_documents": 0,
        "total_chunks": 0,
        "last_error": None,
    }
    assert systems == []


def test_stats_endpoint_is_not_exposed():
    client, _systems = build_client()

    assert client.get("/stats").status_code == 404


def test_get_system_creates_once_without_initializing_knowledge_base():
    from finrag.api import RAGService

    created = []

    def factory():
        system = FakeFinRAGSystem()
        created.append(system)
        return system

    service = RAGService(factory)
    with ThreadPoolExecutor(max_workers=4) as executor:
        systems = list(executor.map(lambda _: service.get_system(), range(4)))

    assert len(created) == 1
    assert all(system is created[0] for system in systems)
    assert created[0].initialize_calls == 0
    assert created[0].build_calls == 0


def test_warmup_initializes_knowledge_base_once_for_concurrent_callers():
    from finrag.api import RAGService

    created = []

    def factory():
        system = FakeFinRAGSystem()
        created.append(system)
        return system

    service = RAGService(factory)
    with ThreadPoolExecutor(max_workers=4) as executor:
        systems = list(executor.map(lambda _: service.ensure_knowledge_base_ready(), range(4)))

    assert len(created) == 1
    assert all(system is created[0] for system in systems)
    assert created[0].initialize_calls == 1
    assert created[0].build_calls == 1


def test_service_delegates_knowledge_ready_check_to_system():
    from finrag.api import RAGService

    class ReadySystem(FakeFinRAGSystem):
        def __init__(self):
            super().__init__()
            self.retrieval_module = object()

    created = []

    def factory():
        system = ReadySystem()
        created.append(system)
        return system

    service = RAGService(factory)
    service.ensure_knowledge_base_ready()

    assert created[0].ensure_calls == 1
    assert created[0].initialize_calls == 0
    assert created[0].build_calls == 0


def test_ready_reports_knowledge_base_initialization_failure_after_system_created():
    from finrag.api import RAGService

    class FailingWarmupSystem(FakeFinRAGSystem):
        def build_knowledge_base(self):
            raise RuntimeError("Milvus 不可用")

    service = RAGService(FailingWarmupSystem)

    try:
        service.ensure_knowledge_base_ready()
    except RuntimeError:
        pass

    ready = service.ready()

    assert ready["ready"] is False
    assert ready["status"] == "error"
    assert ready["last_error"] == "RuntimeError: Milvus 不可用"


def test_api_components_are_available_from_focused_modules():
    from finrag.api import RAGService as ExportedRAGService
    from finrag.api.schemas import AskRequest
    from finrag.api.rag_service import RAGService

    assert RAGService is ExportedRAGService
    request = AskRequest(question="  hello  ")
    assert request.question == "hello"
    assert request.knowledge_base_id is None
    assert not hasattr(request, "retrieval_strategy")


def test_ask_initializes_knowledge_base_before_question():
    client, systems = build_client()

    with client.stream(
        "POST",
        "/ask",
        json={"question": "客户风险等级如何匹配？", "return_sources": True, "return_trace": True},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    events = parse_sse_events(body)
    assert "error" not in [name for name, _ in events]
    done_payload = json.loads(next(data for name, data in events if name == "done"))
    payload = done_payload["response"]
    assert payload["route_type"] == "knowledge"
    assert systems[0].ensure_calls == 1
    assert systems[0].build_calls == 1
    assert systems[0].ask_calls[0]["question"] == "客户风险等级如何匹配？"


def test_ask_returns_sse_events_sources_and_trace():
    client, systems = build_client()

    with client.stream(
        "POST",
        "/ask",
        json={
            "question": "客户风险等级如何匹配？",
            "knowledge_base_id": "kb-finance",
            "return_sources": True,
            "return_trace": True,
        },
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    event_names = [name for name, _ in parse_sse_events(body)]
    assert event_names == ["done"]
    assert "客户风险等级应与产品风险等级匹配" in body
    assert "node-1" not in body
    assert systems[0].ask_calls[0]["knowledge_base_id"] == "kb-finance"
    assert "retrieval_strategy" not in systems[0].ask_calls[0]


def test_ask_omitted_knowledge_base_id_is_left_for_system_config_default():
    client, systems = build_client()

    with client.stream(
        "POST",
        "/ask",
        json={"question": "客户风险等级如何匹配？"},
    ) as response:
        "".join(response.iter_text())

    assert response.status_code == 200
    assert systems[0].ask_calls[0]["knowledge_base_id"] is None


def test_ask_stream_respects_return_sources_and_trace_flags():
    client, systems = build_client()

    with client.stream(
        "POST",
        "/ask",
        json={
            "question": "客户风险等级如何匹配？",
            "knowledge_base_id": "kb-finance",
            "return_sources": False,
            "return_trace": False,
        },
    ) as response:
        body = "".join(response.iter_text())

    events = parse_sse_events(body)
    event_names = [name for name, _ in events]
    done_payload = json.loads(next(data for name, data in events if name == "done"))

    assert response.status_code == 200
    assert "source" not in event_names
    assert done_payload["response"]["sources"] == []
    assert "trace" not in done_payload["response"]
    assert systems[0].ask_calls[0]["return_sources"] is False
    assert systems[0].ask_calls[0]["return_trace"] is False


def test_completion_events_only_emit_done_when_missing():
    from finrag.api import RAGService

    response = FinRAGResponse(
        question="帮我写一段会议通知",
        answer="普通问答回答",
        sources=[],
        trace=RAGTrace(
            retrieval_strategy="llamaindex_router",
            route_type="general",
            final_decision="general_answer",
        ),
        retrieval_strategy="llamaindex_router",
        route_type="general",
    )

    events = RAGService._completion_events(response, set())
    event_names = [event["type"] for event in events]

    assert event_names == ["done"]


def test_completion_events_do_not_duplicate_done():
    from finrag.api import RAGService

    response = FinRAGResponse(
        question="帮我写一段会议通知",
        answer="普通问答回答",
        sources=[],
        retrieval_strategy="llamaindex_router",
        route_type="general",
    )

    assert RAGService._completion_events(response, {"done"}) == []


class CancellationAwareSystem(FakeFinRAGSystem):
    def __init__(self):
        super().__init__()
        self.cancel_observed = False

    def ask_question(self, question, **kwargs):
        event_sink = kwargs.get("event_sink")
        cancel_event = kwargs.get("cancel_event")
        if event_sink is not None:
            event_sink({"type": "analysis", "question": question})
        deadline = time.time() + 1
        while cancel_event is not None and not cancel_event.is_set() and time.time() < deadline:
            time.sleep(0.01)
        self.cancel_observed = bool(cancel_event is not None and cancel_event.is_set())
        return super().ask_question(question, **kwargs)


def test_service_stream_sets_cancel_event_when_client_stops_reading():
    from finrag.api import RAGService
    from finrag.api.schemas import AskRequest

    created = []

    def factory():
        system = CancellationAwareSystem()
        created.append(system)
        return system

    service = RAGService(factory)

    async def read_one_chunk_and_close():
        stream = service.ask_stream(AskRequest(question="客户风险等级如何匹配？"), is_disconnected=lambda: asyncio.sleep(0, result=False))
        first = await stream.__anext__()
        await stream.aclose()
        return first

    first_chunk = asyncio.run(read_one_chunk_and_close())

    assert "event: analysis" in first_chunk
    assert created[0].cancel_observed is True
