"""Score detectors against the injected anomalies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .simulate import STEP_MINUTES, Injected

TOLERANCE = 12  # a flag up to 1 hour after an anomaly ends still counts as catching it


@dataclass(frozen=True)
class EventResult:
    anomaly: Injected
    detected: bool
    detected_by: tuple[str, ...]
    delay_minutes: float | None  # time from anomaly start to first flag


@dataclass(frozen=True)
class Summary:
    events: list[EventResult]
    precision: float          # share of flagged points that belong to an injected anomaly
    false_alarm_points: int

    def recall_by_kind(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for e in self.events:
            hit, total = out.get(e.anomaly.kind, (0, 0))
            out[e.anomaly.kind] = (hit + e.detected, total + 1)
        return out


def score(results: dict[str, pd.DataFrame], injected: list[Injected], truth: pd.DataFrame) -> Summary:
    any_flag = sum(frame.astype(int) for frame in results.values()) > 0
    events = []
    for a in injected:
        window = slice(a.start, min(a.end + TOLERANCE, len(truth)))
        by = tuple(d for d, frame in results.items() if frame[a.sensor].iloc[window].any())
        flagged = np.flatnonzero(any_flag[a.sensor].iloc[window].to_numpy())
        delay = float(flagged[0] * STEP_MINUTES) if len(flagged) else None
        events.append(EventResult(a, bool(by), by, delay))

    # Widen each true anomaly by the tolerance before judging flags as true or false.
    near_truth = truth.copy()
    for a in injected:
        near_truth.iloc[a.start:min(a.end + TOLERANCE, len(truth)), truth.columns.get_loc(a.sensor)] = True
    flagged_total = int(any_flag.to_numpy().sum())
    false_points = int((any_flag & ~near_truth).to_numpy().sum())
    precision = 1 - false_points / flagged_total if flagged_total else 1.0
    return Summary(events, precision, false_points)
