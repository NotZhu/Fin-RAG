from finrag.core.config import PROJECT_ROOT


def test_professional_script_and_document_names_are_used():
    assert (PROJECT_ROOT / "scripts" / "evaluate_demo_documents.py").is_file()
    assert (PROJECT_ROOT / "scripts" / "evaluate_demo_documents_ragas.py").is_file()
    assert (PROJECT_ROOT / "scripts" / "generate_demo_documents.py").is_file()
    assert not (PROJECT_ROOT / "scripts" / "evaluate_retrieval.py").exists()
    assert not (PROJECT_ROOT / "scripts" / "evaluate_ragas.py").exists()
    assert not (PROJECT_ROOT / "scripts" / "run_retrieval_eval.py").exists()
    assert not (PROJECT_ROOT / "scripts" / "run_ragas_eval.py").exists()

    assert not (PROJECT_ROOT / "docs" / "superpowers" / "specs" / "pasted-text.txt").exists()


def test_api_app_is_split_into_focused_modules():
    expected_modules = [
        "src/finrag/api/main.py",
        "src/finrag/api/middleware.py",
        "src/finrag/api/handlers.py",
        "src/finrag/api/routes/frontend.py",
        "src/finrag/api/routes/health.py",
        "src/finrag/api/routes/documents.py",
        "src/finrag/api/routes/qa.py",
    ]
    for relative_path in expected_modules:
        assert (PROJECT_ROOT / relative_path).is_file(), relative_path

    main_text = (PROJECT_ROOT / "src" / "finrag" / "api" / "main.py").read_text(encoding="utf-8")
    assert "@app.get" not in main_text
    assert "@app.post" not in main_text
    assert "@app.delete" not in main_text
    assert "@app.middleware" not in main_text
