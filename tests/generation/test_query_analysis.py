import json

from finrag.generation import GenerationIntegrationModule


def test_generation_module_defaults_to_qwen_3_7_max(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    module = GenerationIntegrationModule()

    assert module.model_name == "qwen3.7-max"


def test_generation_module_only_configures_llm(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    module = GenerationIntegrationModule()

    assert module.llm is None


def test_generation_module_uses_dashscope_compatible_llm(monkeypatch):
    from finrag.generation.answering import DashScopeCompatibleLLM

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    module = GenerationIntegrationModule(model_name="qwen3.7-max")

    assert isinstance(module.llm, DashScopeCompatibleLLM)
    assert module.llm.model_name == "qwen3.7-max"


def test_dashscope_compatible_llm_posts_chat_completions(monkeypatch):
    from finrag.generation.answering import DashScopeCompatibleLLM

    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": '[{"choice": 1, "reason": "知识库问题"}]'}}]},
                ensure_ascii=False,
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["authorization"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = DashScopeCompatibleLLM(api_key="test-key", model_name="qwen3.7-max", temperature=0.1, max_tokens=128)

    response = llm.complete("请选择最合适的一个工具，并按要求输出 JSON。")

    assert str(response) == '[{"choice": 1, "reason": "知识库问题"}]'
    assert captured["url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    assert captured["authorization"] == "Bearer test-key"
    assert captured["timeout"] == 60
    assert captured["payload"] == {
        "model": "qwen3.7-max",
        "messages": [{"role": "user", "content": "请选择最合适的一个工具，并按要求输出 JSON。"}],
        "temperature": 0.1,
        "max_tokens": 128,
    }


def test_generation_module_does_not_expose_removed_query_analysis(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    module = GenerationIntegrationModule()

    assert not hasattr(module, "analyze_query")
    assert not hasattr(module, "_route_query")
    assert not hasattr(module, "_route_prompt")


def test_generation_module_does_not_expose_json_router_parser(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    module = GenerationIntegrationModule()

    assert not hasattr(module, "_parse_json_object")


def test_generation_module_does_not_expose_removed_context_builder(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    module = GenerationIntegrationModule()

    assert not hasattr(module, "_build_context")


def test_generation_module_does_not_expose_removed_sync_answer_wrappers(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    module = GenerationIntegrationModule()

    assert not hasattr(module, "stream_general_answer")
    assert not hasattr(module, "stream_grounded_answer")
    assert not hasattr(module, "generate_general_answer")
    assert not hasattr(module, "generate_grounded_answer")
    assert not hasattr(module, "_stream_chunk_text")
    assert not hasattr(module, "_grounded_answer_rules")
    assert not hasattr(module, "_stream_response_text")
    assert not hasattr(module, "_prepare_citation_nodes")
    assert not hasattr(module, "_extractive_answer")


def test_generation_module_does_not_expose_answer_cleanup_helper(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    module = GenerationIntegrationModule()

    assert not hasattr(module, "_clean_answer_text")
