"""Rolling-origin, leakage-safe evaluation harness for environmental Valley Fever forecasting."""

from .config import FeatureBlock, HarnessConfig, RateSpec
from .data import load_panel, mask_history, panel_from_frame
from .features import Design, build_design
from .harness import PREDICTION_COLUMNS, run_panel, training_rows
from .metrics import diebold_mariano, paired_contrasts, summarize
from .models import MODEL_BUILDERS, ForecastModel, TabularModel, build_models
from .tasks import ForecastTask

__all__ = [
    "FeatureBlock", "HarnessConfig", "RateSpec", "load_panel", "mask_history", "panel_from_frame", "Design",
    "build_design", "PREDICTION_COLUMNS", "run_panel", "training_rows", "diebold_mariano", "paired_contrasts",
    "summarize", "MODEL_BUILDERS", "ForecastModel", "TabularModel", "build_models", "ForecastTask",
]
