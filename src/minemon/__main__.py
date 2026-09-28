"""python -m minemon --days 14 --seed 7 --out output"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .detect import run_all, to_alerts
from .evaluate import score
from .report import markdown, plot_sensor
from .sensors import SENSORS
from .simulate import POINTS_PER_DAY, labels, simulate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="minemon", description="Simulate mine sensors, detect faults, score detectors.")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--warmup-days", type=int, default=3, help="clean days used to learn the normal daily profile")
    p.add_argument("--out", type=Path, default=Path("output"))
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)

    df, injected = simulate(args.days, args.seed, args.warmup_days)
    results = run_all(df, SENSORS, baseline_points=args.warmup_days * POINTS_PER_DAY)
    summary = score(results, injected, labels(df, injected))
    alerts = to_alerts(results, df)

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "readings.csv", index_label="timestamp")
    pd.DataFrame([asdict(a) for a in alerts]).to_csv(args.out / "alerts.csv", index=False)
    report = markdown(summary, alerts, args.days, args.seed)
    (args.out / "summary.md").write_text(report, encoding="utf-8")
    if not args.no_plots:
        for sensor in df.columns:
            plot_sensor(df, sensor, injected, results, args.out / f"{sensor}.png")
    print(report)
    print(f"files written to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
