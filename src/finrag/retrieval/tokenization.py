"""检索文本共享分词工具"""

from __future__ import annotations

import logging
import warnings
from functools import lru_cache
from typing import List

logger = logging.getLogger(__name__)


def tokenize_chinese_text(text: str) -> List[str]:
    """
    对中文检索文本分词，jieba 不可用时回退为逐字切分
    Args:
        text: 需要分词的中文文本
    Returns:
        分词后的字符串列表，已去除空白并转换为小写
    """
    cut_for_search = _get_jieba_cut_for_search()
    tokens = cut_for_search(text or "") if cut_for_search else list(text or "")
    return [token.strip().lower() for token in tokens if token.strip()]


@lru_cache(maxsize=1)
def _get_jieba_cut_for_search():
    """
    延迟加载 jieba 搜索分词器
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import jieba

        jieba.setLogLevel(logging.WARNING)
        return jieba.cut_for_search
    except Exception as exc:
        logger.warning("jieba 分词器不可用，回退为逐字切分: %s", exc)
        return None
