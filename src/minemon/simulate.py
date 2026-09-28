"""Synthetic sensor data with labelled faults and events, so detectors can be scored."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .sensors import SENSORS, Sensor

STEP_MINUTES = 5
POINTS_PER_DAY = 24 * 60 // STEP_MINUTES  # 288

# Sensor faults (the reading is wrong) and one real event (the air really is dangerous).
FAULT_TYPES = ("spike", "stuck", "dropout", "drift")
EVENT_TYPES = ("gas_event",)


@dataclass(frozen=True)
class Injected:
    sensor: str
    kind: str
    start: int  # index of first affected point
    end: int    # index after the last affected point


def normal_signal(sensor: Sensor, n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n)
    # Peak around mid-day (index 144 of 288), when production and traffic are highest.
    daily = sensor.daily_amp * np.sin(2 * np.pi * (t - POINTS_PER_DAY / 4) / POINTS_PER_DAY)
    return sensor.base + daily + rng.normal(0, sensor.noise, n)


def _free_slot(taken: list[tuple[int, int]], length: int, lo: int, hi: int, rng: np.random.Generator) -> int:
    """Pick a start so [start, start+length) does not touch any taken range (with a 1-hour gap)."""
    gap = 12
    for _ in range(1000):
        start = int(rng.integers(lo, hi - length))
        if all(start + length + gap <= a or start >= b + gap for a, b in taken):
            return start
    raise RuntimeError("could not place anomaly; use more days")


def simulate(days: int = 14, seed: int = 7, warmup_days: int = 3) -> tuple[pd.DataFrame, list[Injected]]:
    """Return (wide DataFrame indexed by timestamp, list of injected anomalies).

    The first `warmup_days` are kept clean: detectors learn their baseline from them.
    """
    if days < warmup_days + 5:
        raise ValueError("need at least warmup_days + 5 days of data")
    rng = np.random.default_rng(seed)
    n = days * POINTS_PER_DAY
    index = pd.date_range("2026-01-05", periods=n, freq=f"{STEP_MINUTES}min")
    data, injected = {}, []
    lo = warmup_days * POINTS_PER_DAY

    for name, sensor in SENSORS.items():
        x = normal_signal(sensor, n, rng)
        taken: list[tuple[int, int]] = []
        drift_len = 2 * POINTS_PER_DAY
        # (kind, length): two single-point glitches, 3 hours frozen, 2 hours missing, 2 days of drift,
        # and for methane one real gas build-up.
        plan = [("spike", 1), ("spike", 1), ("stuck", 36), ("dropout", 24), ("drift", drift_len)]
        if name == "ch4_pct":
            plan.append(("gas_event", 36))

        for kind, length in plan:
            s = _free_slot(taken, length, lo, n, rng)
            taken.append((s, s + length))
            injected.append(Injected(name, kind, s, s + length))
            if kind == "spike":  # 8-12 noise SDs away, either direction
                x[s] += rng.choice([-1, 1]) * rng.uniform(8, 12) * sensor.noise
            elif kind == "stuck":
                x[s:s + length] = x[s - 1]
            elif kind == "dropout":
                x[s:s + length] = np.nan
            elif kind == "drift":  # calibration drift ramping to 6 SDs
                x[s:s + length] += np.linspace(0, 6 * sensor.noise, length)
            elif kind == "gas_event":  # rises past the 1.0% limit and clears over 3 hours
                x[s:s + length] += 0.9 * np.sin(np.linspace(0, np.pi, length))

        data[name] = x

    return pd.DataFrame(data, index=index), sorted(injected, key=lambda a: (a.sensor, a.start))


def labels(df: pd.DataFrame, injected: list[Injected]) -> pd.DataFrame:
    """Boolean frame, True where a point is inside any injected anomaly."""
    out = pd.DataFrame(False, index=df.index, columns=df.columns)
    for a in injected:
        out.iloc[a.start:a.end, out.columns.get_loc(a.sensor)] = True
    return out
