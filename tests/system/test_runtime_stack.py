from finrag.core.config import PROJECT_ROOT


def test_runtime_stack_docker_compose_declares_required_services():
    compose_text = (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    for service_name in ["postgres", "milvus", "etcd", "minio", "attu"]:
        assert f"  {service_name}:" in compose_text
    assert "  redis:" not in compose_text

    for port in ["5432", "19530", "9091", "9000", "3000"]:
        assert port in compose_text
    assert "6379" not in compose_text
