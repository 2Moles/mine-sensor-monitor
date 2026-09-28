"""Sensor definitions: normal behaviour for the simulator and alarm limits for the detector.

The limits are illustrative values in the range used by mine safety and discharge rules.
They are not taken from any specific regulation or permit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Sensor:
    name: str
    unit: str
    base: float          # typical level
    daily_amp: float     # size of the day/night cycle (shift work, haul-road traffic)
    noise: float         # standard deviation of measurement noise
    low: float | None    # alarm below this (None = no lower limit)
    high: float | None   # alarm above this (None = no upper limit)


SENSORS: dict[str, Sensor] = {
    s.name: s
    for s in [
        Sensor("ch4_pct", "% vol", base=0.30, daily_amp=0.10, noise=0.03, low=None, high=1.0),
        Sensor("co_ppm", "ppm", base=8.0, daily_amp=3.0, noise=1.0, low=None, high=25.0),
        Sensor("o2_pct", "% vol", base=20.8, daily_amp=0.05, noise=0.03, low=19.5, high=None),
        Sensor("pm10_ugm3", "µg/m³", base=60.0, daily_amp=25.0, noise=8.0, low=None, high=150.0),
        Sensor("water_ph", "pH", base=7.4, daily_amp=0.10, noise=0.05, low=6.5, high=8.5),
    ]
}
