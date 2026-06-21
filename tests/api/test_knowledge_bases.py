from types import SimpleNamespace

from fastapi.testclient import TestClient

from finrag.api import create_app
from finrag.storage.knowledge_base_registry import (
    DuplicateKnowledgeBaseError,
    KnowledgeBaseNotFoundError,
    ProtectedKnowledgeBaseError,
)


class FakeKnowledgeBaseSystem:
    def __init__(self):
        self.created = []
        self.archived = []
        self.restored = []
        self.deleted = []
        self.config = SimpleNamespace(
            knowledge_base_id="finance",
            upload_dir="storage/uploads",
            max_upload_bytes=20 * 1024 * 1024,
        )
        self.knowledge_bases = {
            "finance": {
                "knowledge_base_id": "finance",
                "document_count": 1,
                "status": "active",
                "created_at": "2026-06-18T00:00:00+00:00",
                "updated_at": "2026-06-18T00:00:00+00:00",
                "archived_at": None,
                "deleted_at": None,
            },
            "risk": {
                "knowledge_base_id": "risk",
                "document_count": 0,
                "status": "active",
                "created_at": "2026-06-18T00:02:00+00:00",
                "updated_at": "2026-06-18T00:02:00+00:00",
                "archived_at": None,
                "deleted_at": None,
            },
        }

    def list_knowledge_bases(self):
        return [
            record
            for record in self.knowledge_bases.values()
            if record["status"] != "deleted"
        ]

    def create_knowledge_base(self, knowledge_base_id: str):
        if knowledge_base_id in self.knowledge_bases:
            raise DuplicateKnowledgeBaseError(knowledge_base_id)
        payload = {
            "knowledge_base_id": knowledge_base_id,
            "document_count": 0,
            "status": "active",
            "created_at": "2026-06-18T00:01:00+00:00",
            "updated_at": "2026-06-18T00:01:00+00:00",
            "archived_at": None,
            "deleted_at": None,
        }
        self.knowledge_bases[knowledge_base_id] = payload
        self.created.append(payload)
        return payload

    def archive_knowledge_base(self, knowledge_base_id: str):
        if knowledge_base_id == self.config.knowledge_base_id:
            raise ProtectedKnowledgeBaseError(knowledge_base_id)
        record = self.knowledge_bases[knowledge_base_id]
        record["status"] = "archived"
        record["archived_at"] = "2026-06-18T00:03:00+00:00"
        self.archived.append(knowledge_base_id)
        return record

    def restore_knowledge_base(self, knowledge_base_id: str):
        record = self.knowledge_bases[knowledge_base_id]
        if record["status"] == "deleted":
            raise KnowledgeBaseNotFoundError(knowledge_base_id)
        record["status"] = "active"
        record["archived_at"] = None
        self.restored.append(knowledge_base_id)
        return record

    def delete_knowledge_base(self, knowledge_base_id: str):
        if knowledge_base_id == self.config.knowledge_base_id:
            raise ProtectedKnowledgeBaseError(knowledge_base_id)
        record = self.knowledge_bases[knowledge_base_id]
        record["status"] = "deleted"
        record["deleted_at"] = "2026-06-18T00:04:00+00:00"
        self.deleted.append(knowledge_base_id)
        return record


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
    assert response.json()["knowledge_bases"][0]["status"] == "active"


def test_create_knowledge_base_accepts_user_supplied_knowledge_base_id():
    client, systems = build_client()

    response = client.post(
        "/knowledge-bases",
        json={"knowledge_base_id": "credit"},
    )

    assert response.status_code == 201
    assert response.json()["knowledge_base_id"] == "credit"
    assert systems[0].created[0]["knowledge_base_id"] == "credit"


def test_archive_knowledge_base_marks_workspace_archived():
    client, systems = build_client()

    response = client.post("/knowledge-bases/risk/archive")

    assert response.status_code == 200
    assert response.json()["knowledge_base_id"] == "risk"
    assert response.json()["status"] == "archived"
    assert systems[0].archived == ["risk"]


def test_restore_knowledge_base_marks_workspace_active():
    client, systems = build_client()
    client.post("/knowledge-bases/risk/archive")

    response = client.post("/knowledge-bases/risk/restore")

    assert response.status_code == 200
    assert response.json()["knowledge_base_id"] == "risk"
    assert response.json()["status"] == "active"
    assert systems[0].restored == ["risk"]


def test_delete_knowledge_base_marks_workspace_deleted_and_hides_it_from_list():
    client, systems = build_client()

    response = client.delete("/knowledge-bases/risk")
    listed = client.get("/knowledge-bases")

    assert response.status_code == 200
    assert response.json()["knowledge_base_id"] == "risk"
    assert response.json()["status"] == "deleted"
    assert systems[0].deleted == ["risk"]
    assert [item["knowledge_base_id"] for item in listed.json()["knowledge_bases"]] == ["finance"]


def test_deleted_knowledge_base_cannot_be_restored():
    client, systems = build_client()
    client.delete("/knowledge-bases/risk")

    response = client.post("/knowledge-bases/risk/restore")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "knowledge_base_not_found"
    assert systems[0].restored == []


def test_archive_and_delete_default_knowledge_base_are_rejected():
    client, systems = build_client()

    archived = client.post("/knowledge-bases/finance/archive")
    deleted = client.delete("/knowledge-bases/finance")

    assert archived.status_code == 409
    assert archived.json()["error"]["code"] == "default_knowledge_base_protected"
    assert deleted.status_code == 409
    assert deleted.json()["error"]["code"] == "default_knowledge_base_protected"
    assert systems[0].archived == []
    assert systems[0].deleted == []


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
