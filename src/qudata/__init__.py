"""QuData - Data processing pipeline for LLM training datasets.

Lightweight package initialization to avoid pulling heavy optional
dependencies at import time (e.g., OCR stack). Import submodules
directly where needed, e.g.:

    from qudata.config import ConfigManager
    from qudata.pipeline import QuDataPipeline

This keeps `import qudata` and `from qudata import ...` fast and robust
in minimal environments.
"""

__version__ = "1.0.0"
__author__ = "Qubase Team"

__all__ = [
    "__version__",
    "__author__",
]