"""Early Time Series Classification with ROCKET variants"""

from .classifier import EarlyTimeSeriesClassifier
from .evaluator import EarlyClassificationEvaluator
from .transformer_classifier import EarlyTransformerClassifier
from .utils import normalize_input, generate_synthetic_ts_data

__version__ = "0.1.0"

__all__ = [
    "EarlyTimeSeriesClassifier",
    "EarlyClassificationEvaluator",
    "EarlyTransformerClassifier",
    "normalize_input",
    "generate_synthetic_ts_data",
]
