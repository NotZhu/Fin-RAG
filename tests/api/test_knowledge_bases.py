from types import SimpleNamespace

from fastapi.testclient import TestClient

from finrag.api import create_app
from finrag.storage.knowledge_base_registry import DuplicateKnowledgeBaseError


class FakeKnowledgeBaseSystem:
    def __init__(self):
        self.created = []
        self.config = SimpleNamespace(
            knowledge_base_id="finance",
            upload_dir="storage/uploads",
            max_upload_bytes=20 * 1024 * 1024,
        )

    def list_knowledge_bases(self):
        return [
            {
                "knowledge_base_id": "finance",
                "document_count": 1,
                "created_at": "2026-06-18T00:00:00+00:00",
                "updated_at": "2026-06-18T00:00:00+00:00",
            }
        ]

    def create_knowledge_base(self, knowledge_base_id: str):
        if knowledge_base_id == "finance":
            raise DuplicateKnowledgeBaseError(knowledge_base_id)
        payload = {
            "knowledge_base_id": knowledge_base_id,
            "document_count": 0,
            "created_at": "2026-06-18T00:01:00+00:00",
            "updated_at": "2026-06-18T00:01:00+00:00",
        }
        self.created.append(payload)
        return payload


def build_client():
    systems = []

    def factory():
        system = FakeKnowledgeBaseSystem()
        systems.append(system)
        return system

    return TestClient(create_app(system_factory=factory)), systems


def test_list_knowledge_bases_returns_existing_workspaces():
    client, _systems = build_client()

    response = client.get("/knowledge-bases")

    assert response.status_code == 200
    assert response.json()["knowledge_bases"][0]["knowledge_base_id"] == "finance"
    assert response.json()["knowledge_bases"][0]["document_count"] == 1


def test_create_knowledge_base_accepts_user_supplied_knowledge_base_id():
    client, systems = build_client()

    response = client.post(
        "/knowledge-bases",
        json={"knowledge_base_id": "risk"},
    )

    assert response.status_code == 201
    assert response.json()["knowledge_base_id"] == "risk"
    assert systems[0].created[0]["knowledge_base_id"] == "risk"


def test_create_knowledge_base_returns_conflict_for_duplicate_id():
    client, _systems = build_client()

    response = client.post("/knowledge-bases", json={"knowledge_base_id": "finance"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "duplicate_knowledge_base"


def test_create_knowledge_base_rejects_invalid_id_before_system_creation():
    client, systems = build_client()

    response = client.post(
        "/knowledge-bases",
        json={"knowledge_base_id": "../outside"},
    )

    assert response.status_code == 422
    assert systems == []
