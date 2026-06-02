"""
AuditorAI — Universal AI Prediction Auditor.

Wraps any AI model — sklearn, PyTorch, HuggingFace, or LLM APIs —
with a second model that learns when the first one is likely wrong,
and suppresses those predictions before they reach the human.
"""

from auditorai.core.system import AuditorSystem, audit
from auditorai.adapters.base import wrap, ModelAdapter
from auditorai.core.evaluate import run_full_evaluation

# Lazy imports for optional adapters
def __getattr__(name):
    if name == "SklearnAdapter":
        from auditorai.adapters.sklearn_adapter import SklearnAdapter
        return SklearnAdapter
    if name == "PyTorchAdapter":
        from auditorai.adapters.pytorch_adapter import PyTorchAdapter
        return PyTorchAdapter
    if name == "HuggingFaceAdapter":
        from auditorai.adapters.huggingface_adapter import HuggingFaceAdapter
        return HuggingFaceAdapter
    if name == "APIAdapter":
        from auditorai.adapters.api_adapter import APIAdapter
        return APIAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.2.0"
__all__ = [
    "AuditorSystem",
    "audit",
    "wrap",
    "ModelAdapter",
    "SklearnAdapter",
    "PyTorchAdapter",
    "HuggingFaceAdapter",
    "APIAdapter",
    "run_full_evaluation",
]
