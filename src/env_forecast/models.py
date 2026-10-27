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
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from .config import HarnessConfig
from .features import window_mean
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
    columns_for: Callable[[ForecastTask], list[str]] | None = None

    def fit_predict(self, task: ForecastTask) -> float:
        columns = self.columns_for(task) if self.columns_for else list(task.X_train.columns)
        estimator = self.factory(task.seed)
        if self.log_target:
            estimator = TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1, check_inverse=False)
        estimator.fit(task.X_train[columns], task.y_train)
        prediction = float(estimator.predict(task.X_forecast[columns])[0])
        return max(prediction, self.clip_min) if self.clip_min is not None else prediction


def ar_columns(task: ForecastTask) -> list[str]:
    """With feature set R, keep only the rate lags (an AR(p) model); other feature sets use every column."""
    if task.feature_set != "R":
        return list(task.X_train.columns)
    return [c for c in task.X_train.columns if c.startswith("rate_lag")]


def lr_factory(alphas: tuple[float, ...], cv_splits: int) -> Callable[[int], Any]:
    """
    Ridge-regularised linear regression; the penalty is chosen inside each training window by
    chronological cross-validation (TimeSeriesSplit), so no evaluation target influences it.
    """
    return lambda seed: linear_pipeline(RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=cv_splits)))


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
    county_orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    failures: int = 0
    _cache: dict = field(default_factory=dict, repr=False)

    def orders(self, county: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """County-specific (order, seasonal_order) if selected on the development period, else the defaults."""
        chosen = self.county_orders.get(county, {})
        return tuple(chosen.get("order", self.order)), tuple(chosen.get("seasonal_order", self.seasonal_order))

    def fit_predict(self, task: ForecastTask) -> float:
        case_lag, window = task.config.rate.availability_lag, task.config.window_months
        end = task.origin - case_lag
        values = task.history[task.target_col].loc[end - window + 1:end].to_numpy(float)
        steps = max(task.config.horizons) + case_lag
        orders = self.orders(task.county)
        key = (hashlib.sha256(values.tobytes()).hexdigest(), steps, orders)
        if key not in self._cache:
            self._cache[key] = self._forecast(values, steps, orders=orders)
        return float(self._cache[key][task.horizon + case_lag - 1])

    def _forecast(self, values: np.ndarray, steps: int, exog: np.ndarray | None = None,
                  exog_future: np.ndarray | None = None, orders: tuple | None = None) -> np.ndarray:
        order, seasonal_order = orders or (self.order, self.seasonal_order)
        endog = np.log1p(values) if self.log_target else values
        try:
            # one BLAS thread: the likelihood optimum otherwise depends on thread count (serial vs parallel runs)
            with warnings.catch_warnings(), threadpool_limits(limits=1):
                warnings.simplefilter("ignore")
                model = SARIMAX(endog, exog=exog, order=order, seasonal_order=seasonal_order, trend=self.trend)
                path = model.fit(disp=False).forecast(steps, exog=exog_future)
        except (ValueError, np.linalg.LinAlgError):
            self.failures += 1
            return np.full(steps, np.nan)
        return np.clip(np.expm1(path) if self.log_target else path, 0.0, None)


DEFAULT_SARIMAX_EXOG = (
    {"column": "Precipitation", "rolling_mean": 12},
    {"column": "Avg Soil Temp (F)"},
    {"column": "Avg Wind Speed (mph)"},
    {"column": "AQI_PM10"},
    {"column": "FIRE_Acres_Burned", "log1p": True},
)


@dataclass
class Sarimax(Sarima):
    """
    SARIMA plus lagged environmental regressors, reported as feature set "RE".

    The model is fitted on the last W published rates and forecast h + case lag steps, so it needs
    regressor values for every month m up to the target t + h. Each regressor enters as
    z[m] = x[m - d] with d = h + availability lag of x, so even the target month's regressor is a
    value published by the origin; no realised future environment is ever used. Because d depends on
    h, each horizon has its own fit. Regressors are standardised with training-window statistics and
    missing values are set to the training mean.
    """

    exog: tuple[dict[str, Any], ...] = DEFAULT_SARIMAX_EXOG
    name: str = "sarimax"
    feature_set_label: str = "RE"

    def _regressors(self, task: ForecastTask, months: pd.PeriodIndex) -> np.ndarray:
        columns = []
        for spec in self.exog:
            col = spec["column"]
            lag = task.horizon + task.config.env_cutoffs[col]
            if (months - lag).max() > task.origin - task.config.env_cutoffs[col]:
                raise RuntimeError(f"SARIMAX regressor {col} would use unpublished data at origin {task.origin}.")
            series = np.log1p(task.history[col]) if spec.get("log1p") else task.history[col]
            if spec.get("rolling_mean"):
                series = window_mean(series, spec["rolling_mean"])
            columns.append(series.reindex(months - lag).to_numpy(float))
        return np.column_stack(columns)

    def fit_predict(self, task: ForecastTask) -> float:
        case_lag, window = task.config.rate.availability_lag, task.config.window_months
        end = task.origin - case_lag
        train_months = pd.period_range(end - window + 1, end, freq="M")
        future_months = pd.period_range(end + 1, task.target_month, freq="M")
        values = task.history[task.target_col].reindex(train_months).to_numpy(float)
        exog, exog_future = self._regressors(task, train_months), self._regressors(task, future_months)
        mean, std = np.nanmean(exog, axis=0), np.nanstd(exog, axis=0)
        std[~(std > 0)] = 1.0
        exog, exog_future = (np.nan_to_num((x - mean) / std) for x in (exog, exog_future))
        orders = self.orders(task.county)
        key = (hashlib.sha256(b"".join(a.tobytes() for a in (values, exog, exog_future))).hexdigest(), orders)
        if key not in self._cache:
            self._cache[key] = self._forecast(values, len(future_months), exog, exog_future, orders=orders)
        return float(self._cache[key][-1])


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
        trend=params.get("trend", "c"), log_target=params.get("log_target", True), county_orders=params.get("county_orders", {}),
    ),
    "xgboost": lambda params: TabularModel(
        "xgboost", xgboost_factory(params.get("params", {})), log_target=params.get("log_target", False)
    ),
    "lr": lambda params: TabularModel(
        "lr", lr_factory(tuple(params.get("alphas", np.logspace(-2, 3, 11))), params.get("cv_splits", 5)),
        log_target=params.get("log_target", False), columns_for=ar_columns,
    ),
    "sarimax": lambda params: Sarimax(
        order=tuple(params.get("order", (1, 0, 1))), seasonal_order=tuple(params.get("seasonal_order", (1, 0, 1, 12))),
        trend=params.get("trend", "c"), log_target=params.get("log_target", True),
        exog=tuple(params.get("exog", DEFAULT_SARIMAX_EXOG)), county_orders=params.get("county_orders", {}),
    ),
    "lstm": lambda params: _lstm(params),
}


def _lstm(params: dict[str, Any]) -> ForecastModel:
    from .lstm import LSTMModel  # torch is imported only when an LSTM is requested

    return LSTMModel(**params)


def build_models(config: HarnessConfig) -> list[ForecastModel]:
    """Instantiate the models named in the config, passing their `model_params` entries."""
    unknown = [name for name in config.models if name not in MODEL_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown models {unknown}; registered: {sorted(MODEL_BUILDERS)}")
    return [MODEL_BUILDERS[name](config.model_params.get(name, {})) for name in config.models]
