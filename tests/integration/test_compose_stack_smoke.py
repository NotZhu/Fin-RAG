import os
import socket

import pytest


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def test_compose_stack_ports_are_available_when_enabled():
    if os.environ.get("FINRAG_RUN_INTEGRATION") != "1":
        pytest.skip("docker compose up 后设置 FINRAG_RUN_INTEGRATION=1 才运行真实栈冒烟测试")

    assert _port_open("127.0.0.1", 5432)
    assert _port_open("127.0.0.1", 6379)
    assert _port_open("127.0.0.1", 19530)
