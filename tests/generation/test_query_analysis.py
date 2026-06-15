from finrag.generation import GenerationIntegrationModule


def test_generation_module_defaults_to_qwen_max(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    module = GenerationIntegrationModule()

    assert module.model_name == "qwen-max"


def test_generation_module_only_configures_llm(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    module = GenerationIntegrationModule()

    assert module.llm is None


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
