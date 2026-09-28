"""Plots and a Markdown summary of one run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (CI, servers)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .detect import Alert  # noqa: E402
from .evaluate import Summary  # noqa: E402
from .sensors import SENSORS  # noqa: E402
from .simulate import Injected  # noqa: E402

COLORS = {"spike": "tab:red", "stuck": "tab:purple", "dropout": "tab:gray", "drift": "tab:orange",
          "gas_event": "tab:green"}
MARKERS = {"limit": "v", "dropout": "x", "stuck": "s", "spike": "o", "drift": "^"}


def plot_sensor(df: pd.DataFrame, sensor: str, injected: list[Injected],
                results: dict[str, pd.DataFrame], path: Path) -> None:
    s = SENSORS[sensor]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.plot(df.index, df[sensor], lw=0.6, color="tab:blue", label="reading")
    for a in (a for a in injected if a.sensor == sensor):
        ax.axvspan(df.index[a.start], df.index[min(a.end, len(df) - 1)], color=COLORS[a.kind], alpha=0.18,
                   label=f"injected: {a.kind}")
    for detector, frame in results.items():
        hit = frame[sensor]
        if hit.any():
            y = df[sensor].where(hit).fillna(df[sensor].median())  # dropouts have no value; draw at the median
            ax.scatter(df.index[hit], y[hit], s=14, marker=MARKERS[detector], label=f"flagged: {detector}", zorder=3)
    for lim in (s.low, s.high):
        if lim is not None:
            ax.axhline(lim, ls="--", color="black", lw=0.8)
    ax.set_title(f"{sensor} ({s.unit})")
    handles, names = ax.get_legend_handles_labels()
    unique = dict(zip(names, handles, strict=True))
    ax.legend(unique.values(), unique.keys(), fontsize=7, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def markdown(summary: Summary, alerts: list[Alert], days: int, seed: int) -> str:
    lines = [f"# Run summary (simulated {days} days, seed {seed})", "",
             "| anomaly | caught | total |", "|---|---|---|"]
    for kind, (hit, total) in sorted(summary.recall_by_kind().items()):
        lines.append(f"| {kind} | {hit} | {total} |")
    delays = [e.delay_minutes for e in summary.events if e.anomaly.kind == "drift" and e.delay_minutes is not None]
    lines += ["", f"Point precision: **{summary.precision:.2f}** ({summary.false_alarm_points} flagged points "
              "outside any injected anomaly).", ""]
    if delays:
        lines.append(f"Drift detection delay: median {pd.Series(delays).median() / 60:.1f} h after the drift began.")
    missed = [e.anomaly for e in summary.events if not e.detected]
    if missed:
        lines += ["", "Missed:"] + [f"- {a.sensor} {a.kind} at index {a.start}" for a in missed]
    lines += ["", f"{len(alerts)} alert episodes raised. First 15:", "",
              "| start | sensor | detector | points | extreme |", "|---|---|---|---|---|"]
    for a in alerts[:15]:
        ext = "" if a.extreme is None else f"{a.extreme:.2f}"
        lines.append(f"| {a.start:%m-%d %H:%M} | {a.sensor} | {a.detector} | {a.points} | {ext} |")
    return "\n".join(lines) + "\n"
