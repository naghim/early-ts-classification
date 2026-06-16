"""Early Time Series Classification with ROCKET variants"""

from ._debug import enable_debug, debug_print

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
    "enable_debug",
    "debug_print",
]
