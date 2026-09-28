"""Detectors. Each takes one sensor's series and returns a boolean Series of flagged points.

All of them only look at past data, so they could run on a live stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .sensors import Sensor
from .simulate import POINTS_PER_DAY


def dropout(x: pd.Series) -> pd.Series:
    return x.isna()


def stuck(x: pd.Series, window: int = 12) -> pd.Series:
    """Flag runs where the value has not changed for `window` points (1 hour at 5-minute steps).

    Real readings always have some noise, so a perfectly flat hour means a frozen sensor or logger.
    """
    flat_end = x.diff().abs().rolling(window - 1).max() < 1e-9  # True at the last point of a flat window
    flagged = flat_end.copy()
    for lag in range(1, window):  # extend the flag back over the whole flat window
        flagged |= flat_end.shift(-lag, fill_value=False)
    return flagged & x.notna()


def daily_profile(x: pd.Series, baseline_points: int, harmonics: int = 2) -> tuple[np.ndarray, float]:
    """Fit the normal daily cycle on the baseline period with a few sine/cosine terms.

    Returns (expected value at every point, noise standard deviation). A smooth curve with
    2 * harmonics + 1 parameters needs far less data than a separate mean for each time of day,
    which with 3 days of data would be mostly noise.
    """
    t = 2 * np.pi * (np.arange(len(x)) % POINTS_PER_DAY) / POINTS_PER_DAY
    design = np.column_stack([np.ones(len(x))] + [f(h * t) for h in range(1, harmonics + 1) for f in (np.sin, np.cos)])
    base = x.iloc[:baseline_points].to_numpy()
    ok = ~np.isnan(base)
    coef, *_ = np.linalg.lstsq(design[:baseline_points][ok], base[ok], rcond=None)
    expected = design @ coef
    sigma = float(np.std(base[ok] - expected[:baseline_points][ok]))
    return expected, sigma


def spike(residual: pd.Series, window: int = 48, threshold: float = 6.0) -> pd.Series:
    """Robust z-score of the de-seasonalised reading against the previous `window` points.

    Uses median and MAD rather than mean and SD, so earlier spikes in the window don't hide new ones.
    """
    past = residual.shift(1)
    median = past.rolling(window, min_periods=window // 2).median()
    mad = (past - median).abs().rolling(window, min_periods=window // 2).median()
    z = (residual - median).abs() / (1.4826 * mad.replace(0, np.nan))
    return z > threshold


def drift(residual: pd.Series, sigma: float, start: int, k: float = 0.5, h: float = 20.0,
          skip: pd.Series | None = None) -> pd.Series:
    """Two-sided CUSUM on the de-seasonalised reading, in units of noise SD.

    A slow drift adds a little every step, so the running sum grows until it crosses h. After each
    alarm the sums reset, so a continuing drift re-alarms every few hours and a finished one stops
    promptly instead of leaving a long tail. Points in `skip` (already explained as dropout, stuck
    or spike) are ignored so they cannot trigger it.
    """
    r = (residual / sigma).to_numpy()
    ignore = residual.isna().to_numpy() if skip is None else (skip | residual.isna()).to_numpy()
    up = down = 0.0
    out = np.zeros(len(r), dtype=bool)
    for i in range(start, len(r)):
        if ignore[i]:
            continue
        up = max(0.0, up + r[i] - k)
        down = max(0.0, down - r[i] - k)
        if up > h or down > h:
            out[i] = True
            up = down = 0.0
    return pd.Series(out, index=residual.index)


def limit(x: pd.Series, sensor: Sensor) -> pd.Series:
    flagged = pd.Series(False, index=x.index)
    if sensor.high is not None:
        flagged |= x > sensor.high
    if sensor.low is not None:
        flagged |= x < sensor.low
    return flagged


DETECTORS = ("limit", "dropout", "stuck", "spike", "drift")


def run_all(df: pd.DataFrame, sensors: dict[str, Sensor], baseline_points: int) -> dict[str, pd.DataFrame]:
    """Return {detector: boolean frame with one column per sensor}."""
    results = {d: pd.DataFrame(False, index=df.index, columns=df.columns) for d in DETECTORS}
    for name in df.columns:
        x = df[name]
        results["limit"][name] = limit(x, sensors[name])
        results["dropout"][name] = dropout(x)
        results["stuck"][name] = stuck(x)
        expected, sigma = daily_profile(x, baseline_points)
        residual = x - expected
        results["spike"][name] = spike(residual)
        explained = results["dropout"][name] | results["stuck"][name] | results["spike"][name]
        results["drift"][name] = drift(residual, sigma, start=baseline_points, skip=explained)
    return results


@dataclass(frozen=True)
class Alert:
    sensor: str
    detector: str
    start: pd.Timestamp
    end: pd.Timestamp
    points: int
    extreme: float | None  # reading furthest from the sensor's normal level (None if all missing)


# Flags closer than this many points become one alert. Drift re-alarms every few hours while it
# continues (the CUSUM resets), so it needs a wider gap to read as one incident.
MERGE_GAP = {"drift": 72}


def to_alerts(results: dict[str, pd.DataFrame], df: pd.DataFrame, merge_gap: int = 3) -> list[Alert]:
    """Merge flagged points into alert episodes, so a 3-hour stuck sensor is one alert, not 36."""
    alerts = []
    for detector, frame in results.items():
        gap = MERGE_GAP.get(detector, merge_gap)
        for sensor in frame.columns:
            idx = np.flatnonzero(frame[sensor].to_numpy())
            if not len(idx):
                continue
            normal = df[sensor].median()
            for g in np.split(idx, np.flatnonzero(np.diff(idx) > gap) + 1):
                values = df[sensor].iloc[g[0]:g[-1] + 1].dropna()
                # "Furthest from normal" works for both directions: high methane and low oxygen.
                extreme = float(values.loc[(values - normal).abs().idxmax()]) if len(values) else None
                alerts.append(Alert(sensor, detector, df.index[g[0]], df.index[g[-1]], len(g), extreme))
    return sorted(alerts, key=lambda a: (a.start, a.sensor, a.detector))
