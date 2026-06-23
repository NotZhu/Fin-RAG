"""FinRAG 金融资料库包"""

import warnings

try:
    from pydantic.warnings import UnsupportedFieldAttributeWarning

    warnings.filterwarnings(
        "ignore",
        message="The 'validate_default' attribute with value True was provided to the `Field\\(\\)` function.*",
        category=UnsupportedFieldAttributeWarning,
    )
except Exception:
    pass

from .application.system import FinRAGSystem

__all__ = ["FinRAGSystem"]

__version__ = "1.0.0"
