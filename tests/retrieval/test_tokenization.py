from finrag.retrieval.tokenization import tokenize_chinese_text


def test_tokenize_chinese_text_normalizes_and_removes_blank_tokens():
    tokens = tokenize_chinese_text("客户 风险等级\nKYC")

    assert "客户" in tokens
    assert "kyc" in tokens
    assert "" not in tokens
    assert all(token == token.strip().lower() for token in tokens)


def test_search_and_milvus_reuse_shared_tokenizer():
    import finrag.retrieval as retrieval_package
    import finrag.indexing.milvus as milvus_module
    import finrag.retrieval.search as search_module

    assert retrieval_package.tokenize_chinese_text is tokenize_chinese_text
    assert not hasattr(milvus_module, "_get_jieba_cut_for_search")
    assert not hasattr(milvus_module, "_tokenize_for_bm25")
    assert not hasattr(search_module, "tokenize_chinese_text")
    assert not hasattr(search_module, "_get_jieba_cut_for_search")
