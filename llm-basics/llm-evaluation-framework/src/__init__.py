"""
LLM Evaluation Framework
A simple Python framework to evaluate prompts and models automatically.
"""

from .config import Config
from .models import ModelClient, NVIDIAClient, ZAIClient
from .metrics import MetricsCalculator
from .dataset import Dataset
from .evaluation import Evaluator, EvaluationResult
from .comparison import ModelComparator

__all__ = [
    "Config",
    "ModelClient",
    "NVIDIAClient",
    "ZAIClient",
    "MetricsCalculator",
    "Dataset",
    "Evaluator",
    "EvaluationResult",
    "ModelComparator",
]

__version__ = "0.1.0"
