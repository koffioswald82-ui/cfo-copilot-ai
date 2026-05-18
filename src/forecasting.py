"""
forecasting.py — Revenue and cash flow forecasting.

Two strategies are available:
  1. ARIMA  — classical time-series model (statsmodels)
  2. Ridge  — sklearn linear regression with lag features

The public interface always returns a ForecastResult regardless of strategy.
"""

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

logger = logging.getLogger(__name__)

# Suppress noisy statsmodels convergence warnings during auto-tuning
warnings.filterwarnings("ignore", module="statsmodels")


@dataclass
class ForecastResult:
    metric: str
    model_type: str
    historical: pd.Series
    forecast: pd.Series          # index = future PeriodIndex
    confidence_lower: pd.Series
    confidence_upper: pd.Series
    mape_in_sample: float        # in-sample MAPE on last 4 periods (hold-out proxy)


# ─── ARIMA ───────────────────────────────────────────────────────────────────

def _arima_forecast(series: pd.Series, horizon: int) -> ForecastResult:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise ImportError("statsmodels is required for ARIMA forecasting.") from exc

    train = series.iloc[:-4]
    test = series.iloc[-4:]

    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 4),
                    enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)

    # In-sample MAPE on hold-out
    hold_out_pred = fit.predict(start=len(train), end=len(train) + len(test) - 1)
    mape = mean_absolute_percentage_error(test.values, hold_out_pred.values)

    # Re-fit on full series for forecast
    full_model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 0, 4),
                         enforce_stationarity=False, enforce_invertibility=False)
    full_fit = full_model.fit(disp=False)
    forecast_obj = full_fit.get_forecast(steps=horizon)
    mean_fc = forecast_obj.predicted_mean
    ci = forecast_obj.conf_int(alpha=0.20)   # 80% CI

    future_index = _future_period_index(series.index[-1], horizon)
    fc_series = pd.Series(mean_fc.values, index=future_index, name=series.name)
    lower = pd.Series(ci.iloc[:, 0].values, index=future_index)
    upper = pd.Series(ci.iloc[:, 1].values, index=future_index)

    return ForecastResult(
        metric=str(series.name),
        model_type="ARIMA(1,1,1)(1,1,0)[4]",
        historical=series,
        forecast=fc_series,
        confidence_lower=lower,
        confidence_upper=upper,
        mape_in_sample=round(mape * 100, 2),
    )


# ─── Ridge Regression ────────────────────────────────────────────────────────

def _ridge_forecast(series: pd.Series, horizon: int, n_lags: int = 4) -> ForecastResult:
    """Autoregressive Ridge regression with lag features."""
    values = series.values.astype(float)
    n = len(values)

    # Build lag matrix
    X, y = [], []
    for i in range(n_lags, n):
        X.append(values[i - n_lags:i])
        y.append(values[i])
    X = np.array(X)
    y = np.array(y)

    # Hold-out MAPE on last 4 observations
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[:-4])
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y[:-4])
    preds_ho = model.predict(scaler.transform(X[-4:]))
    mape = mean_absolute_percentage_error(y[-4:], preds_ho)

    # Re-fit on full data
    X_full_scaled = scaler.fit_transform(X)
    model.fit(X_full_scaled, y)

    # Iterative multi-step forecast
    window = list(values[-n_lags:])
    fc_values, lower_values, upper_values = [], [], []
    residuals = y - model.predict(X_full_scaled)
    std_resid = np.std(residuals)

    for step in range(horizon):
        feat = np.array(window[-n_lags:]).reshape(1, -1)
        feat_scaled = scaler.transform(feat)
        pred = model.predict(feat_scaled)[0]
        margin = 1.28 * std_resid * np.sqrt(step + 1)   # grows with horizon
        fc_values.append(pred)
        lower_values.append(pred - margin)
        upper_values.append(pred + margin)
        window.append(pred)

    future_index = _future_period_index(series.index[-1], horizon)
    fc_series = pd.Series(fc_values, index=future_index, name=series.name)
    lower = pd.Series(lower_values, index=future_index)
    upper = pd.Series(upper_values, index=future_index)

    return ForecastResult(
        metric=str(series.name),
        model_type="Ridge Regression (AR lags)",
        historical=series,
        forecast=fc_series,
        confidence_lower=lower,
        confidence_upper=upper,
        mape_in_sample=round(mape * 100, 2),
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def forecast_metric(
    series: pd.Series,
    horizon: int = 4,
    strategy: str = "auto",
) -> ForecastResult:
    """
    Forecast a single financial metric forward `horizon` quarters.

    Parameters
    ----------
    series   : quarterly PeriodIndex Series (e.g. revenue).
    horizon  : number of quarters to forecast (default 4 = 1 year).
    strategy : 'arima', 'ridge', or 'auto' (tries ARIMA, falls back to Ridge).

    Returns
    -------
    ForecastResult dataclass.
    """
    if len(series) < 8:
        raise ValueError("Need at least 8 historical periods for forecasting.")

    if strategy == "arima":
        return _arima_forecast(series, horizon)
    elif strategy == "ridge":
        return _ridge_forecast(series, horizon)
    else:  # auto — prefer ARIMA, fall back to Ridge on exception or poor fit
        _MAPE_CEILING = 50.0  # switch to Ridge if hold-out MAPE exceeds 50%
        arima_result = None
        try:
            arima_result = _arima_forecast(series, horizon)
        except Exception as exc:
            logger.warning("[%s] ARIMA failed (%s) — falling back to Ridge.", series.name, exc)

        if arima_result is not None and arima_result.mape_in_sample <= _MAPE_CEILING:
            logger.info("[%s] ARIMA chosen. MAPE: %.1f%%", series.name, arima_result.mape_in_sample)
            return arima_result

        ridge_result = _ridge_forecast(series, horizon)
        logger.info(
            "[%s] Ridge chosen (ARIMA MAPE %.1f%% > ceiling). Ridge MAPE: %.1f%%",
            series.name,
            arima_result.mape_in_sample if arima_result else float("nan"),
            ridge_result.mape_in_sample,
        )
        return ridge_result


def run_all_forecasts(
    income: pd.DataFrame,
    cash_flow: pd.DataFrame,
    horizon: int = 4,
    strategy: str = "auto",
) -> dict[str, ForecastResult]:
    """Forecast revenue, EBITDA, net income, and operating cash flow."""
    targets = {
        "revenue": income["revenue"],
        "ebitda": income["ebitda"],
        "net_income": income["net_income"],
        "operating_cash_flow": cash_flow["operating_cash_flow"],
    }
    return {
        name: forecast_metric(series, horizon=horizon, strategy=strategy)
        for name, series in targets.items()
    }


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _future_period_index(last_period, horizon: int) -> pd.PeriodIndex:
    return pd.period_range(start=last_period + 1, periods=horizon, freq="Q")


def forecast_summary(forecasts: dict[str, ForecastResult]) -> pd.DataFrame:
    """Return a tidy DataFrame of forecast values for all metrics."""
    rows = []
    for name, fc in forecasts.items():
        for period, val in fc.forecast.items():
            rows.append({
                "metric": name,
                "period": str(period),
                "forecast": round(val),
                "lower_80": round(fc.confidence_lower[period]),
                "upper_80": round(fc.confidence_upper[period]),
                "model": fc.model_type,
                "mape_pct": fc.mape_in_sample,
            })
    return pd.DataFrame(rows)
