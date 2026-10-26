"""
Model adapters for the harness.

Every model implements the `ForecastModel` protocol: a `name`, a `uses_features` flag, and
`fit_predict(task) -> float` returning a rate-scale prediction for task.target_month. A model
with uses_features=False runs once per forecast, labelled with its optional
`feature_set_label` attribute (default "none", e.g. "R" for SARIMA on case history). The
harness calls it once per (county, origin, horizon, feature set); a model never sees data
outside the ForecastTask it is given.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from .config import HarnessConfig
from .tasks import ForecastTask


@runtime_checkable
class ForecastModel(Protocol):
    name: str
    uses_features: bool

    def fit_predict(self, task: ForecastTask) -> float: ...


@dataclass
class SeasonalNaive:
    """Rate in the same calendar month one year before the target (needs h + case lag <= 12)."""

    name: str = "seasonal_naive"
    uses_features: bool = False

    def fit_predict(self, task: ForecastTask) -> float:
        return float(task.history[task.target_col].get(task.target_month - 12, np.nan))


@dataclass
class Persistence:
    """Most recent published rate."""

    name: str = "persistence"
    uses_features: bool = False

    def fit_predict(self, task: ForecastTask) -> float:
        rate = task.history[task.target_col].dropna()
        return float(rate.iloc[-1]) if len(rate) else np.nan


@dataclass
class Climatology:
    """Mean rate of the target's calendar month over the training window's labels."""

    name: str = "climatology"
    uses_features: bool = False

    def fit_predict(self, task: ForecastTask) -> float:
        same_month = task.y_train[task.label_months.month == task.target_month.month]
        return float(same_month.mean()) if len(same_month) else np.nan


@dataclass
class TabularModel:
    """
    Direct model on the harness design matrix.

    `factory(seed)` must return an unfitted sklearn-style estimator. Any preprocessing
    (imputation, scaling) belongs inside that estimator, typically a Pipeline, so it is fitted
    on X_train only and merely applied to X_forecast. With log_target the estimator is trained
    on log1p(rate) and predictions are mapped back with expm1 (a median-type estimate).
    """

    name: str
    factory: Callable[[int], Any]
    log_target: bool = False
    clip_min: float | None = 0.0
    uses_features: bool = True

    def fit_predict(self, task: ForecastTask) -> float:
        estimator = self.factory(task.seed)
        if self.log_target:
            estimator = TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1, check_inverse=False)
        estimator.fit(task.X_train, task.y_train)
        prediction = float(estimator.predict(task.X_forecast)[0])
        return max(prediction, self.clip_min) if self.clip_min is not None else prediction


@dataclass
class Sarima:
    """
    Seasonal ARIMA on case-rate history only (no environmental inputs), reported as feature set "R".

    Fitted on the `window_months` most recent published rates (ending at origin - case lag), then
    forecast h + case lag steps ahead. One fit serves every horizon: forecast paths are cached by
    the exact training values, so a cached path can only be reused for identical inputs. Missing
    rates inside the window are handled by the Kalman filter. Failed fits return NaN and are counted.
    """

    order: tuple[int, int, int] = (1, 0, 1)
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 12)
    trend: str = "c"
    log_target: bool = True
    name: str = "sarima"
    uses_features: bool = False
    feature_set_label: str = "R"
    failures: int = 0
    _cache: dict = field(default_factory=dict, repr=False)

    def fit_predict(self, task: ForecastTask) -> float:
        case_lag, window = task.config.rate.availability_lag, task.config.window_months
        end = task.origin - case_lag
        values = task.history[task.target_col].loc[end - window + 1:end].to_numpy(float)
        steps = max(task.config.horizons) + case_lag
        key = (hashlib.sha256(values.tobytes()).hexdigest(), steps)
        if key not in self._cache:
            self._cache[key] = self._forecast(values, steps)
        return float(self._cache[key][task.horizon + case_lag - 1])

    def _forecast(self, values: np.ndarray, steps: int) -> np.ndarray:
        endog = np.log1p(values) if self.log_target else values
        try:
            # one BLAS thread: the likelihood optimum otherwise depends on thread count (serial vs parallel runs)
            with warnings.catch_warnings(), threadpool_limits(limits=1):
                warnings.simplefilter("ignore")
                fitted = SARIMAX(endog, order=self.order, seasonal_order=self.seasonal_order, trend=self.trend).fit(disp=False)
            path = fitted.forecast(steps)
        except (ValueError, np.linalg.LinAlgError):
            self.failures += 1
            return np.full(steps, np.nan)
        return np.clip(np.expm1(path) if self.log_target else path, 0.0, None)


def linear_pipeline(regressor: Any) -> Any:
    """Median imputation and standardisation fitted on training rows, then the regressor."""
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), regressor)


XGBOOST_DEFAULTS = {
    "n_estimators": 300, "learning_rate": 0.05, "max_depth": 2, "min_child_weight": 5,
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "tree_method": "hist",
}


def xgboost_factory(params: dict[str, Any]) -> Callable[[int], Any]:
    """XGBoost handles missing values natively and needs no scaling; n_jobs=1 keeps fits deterministic."""
    settings = {**XGBOOST_DEFAULTS, **params}
    return lambda seed: XGBRegressor(objective="reg:squarederror", random_state=seed, n_jobs=1, verbosity=0, **settings)


MODEL_BUILDERS: dict[str, Callable[[dict[str, Any]], ForecastModel]] = {
    "seasonal_naive": lambda params: SeasonalNaive(),
    "persistence": lambda params: Persistence(),
    "climatology": lambda params: Climatology(),
    "linear": lambda params: TabularModel(
        "linear", lambda seed: linear_pipeline(LinearRegression()), log_target=params.get("log_target", False)
    ),
    "ridge": lambda params: TabularModel(
        "ridge", lambda seed: linear_pipeline(Ridge(alpha=params.get("alpha", 1.0))), log_target=params.get("log_target", False)
    ),
    "sarima": lambda params: Sarima(
        order=tuple(params.get("order", (1, 0, 1))), seasonal_order=tuple(params.get("seasonal_order", (1, 0, 1, 12))),
        trend=params.get("trend", "c"), log_target=params.get("log_target", True),
    ),
    "xgboost": lambda params: TabularModel(
        "xgboost", xgboost_factory(params.get("params", {})), log_target=params.get("log_target", False)
    ),
}


def build_models(config: HarnessConfig) -> list[ForecastModel]:
    """Instantiate the models named in the config, passing their `model_params` entries."""
    unknown = [name for name in config.models if name not in MODEL_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown models {unknown}; registered: {sorted(MODEL_BUILDERS)}")
    return [MODEL_BUILDERS[name](config.model_params.get(name, {})) for name in config.models]
