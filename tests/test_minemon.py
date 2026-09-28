import numpy as np
import pandas as pd
import pytest

from minemon.detect import daily_profile, drift, limit, run_all, spike, stuck, to_alerts
from minemon.evaluate import score
from minemon.sensors import SENSORS, Sensor
from minemon.simulate import POINTS_PER_DAY, labels, simulate

IDX = pd.date_range("2026-01-01", periods=POINTS_PER_DAY * 6, freq="5min")


def noisy(seed=0, amp=1.0, noise=0.1):
    rng = np.random.default_rng(seed)
    t = np.arange(len(IDX))
    return pd.Series(amp * np.sin(2 * np.pi * t / POINTS_PER_DAY) + rng.normal(0, noise, len(t)), index=IDX)


def test_stuck_flags_exactly_the_flat_run():
    x = noisy()
    x.iloc[500:540] = x.iloc[499]
    flagged = np.flatnonzero(stuck(x).to_numpy())
    assert flagged.min() == 499 and flagged.max() == 539  # the last good value is part of the flat run


def test_stuck_ignores_normal_noise():
    assert not stuck(noisy()).any()


def test_daily_profile_learns_the_cycle():
    x = noisy(amp=2.0, noise=0.1)
    expected, sigma = daily_profile(x, baseline_points=3 * POINTS_PER_DAY)
    assert np.abs(expected - 2.0 * np.sin(2 * np.pi * np.arange(len(x)) / POINTS_PER_DAY)).max() < 0.05
    assert sigma == pytest.approx(0.1, rel=0.15)


def test_spike_found_on_residual_despite_daily_cycle():
    x = noisy(amp=1.0, noise=0.1)
    x.iloc[1000] += 1.0  # 10 noise SDs, but small next to the daily swing
    expected, _ = daily_profile(x, 3 * POINTS_PER_DAY)
    flagged = spike(x - expected)
    assert flagged.iloc[1000] and flagged.sum() <= 2


def test_drift_detected_and_clean_data_quiet():
    x = noisy(noise=0.1)
    expected, sigma = daily_profile(x, 3 * POINTS_PER_DAY)
    assert not drift(x - expected, sigma, start=3 * POINTS_PER_DAY).any()

    start = 4 * POINTS_PER_DAY
    x.iloc[start:] += np.linspace(0, 0.6, len(x) - start)  # ramp to 6 SDs
    flagged = np.flatnonzero(drift(x - expected, sigma, start=3 * POINTS_PER_DAY).to_numpy())
    assert len(flagged) and flagged[0] > start
    assert (flagged[0] - start) * 5 / 60 < 12  # caught within 12 hours


def test_limit_checks_both_directions():
    s = Sensor("o2", "%", 20.8, 0, 0, low=19.5, high=23.0)
    flagged = limit(pd.Series([20.8, 19.4, 23.1, 21.0]), s)
    assert flagged.tolist() == [False, True, True, False]


def test_simulation_is_reproducible_and_labelled():
    a, inj_a = simulate(seed=3)
    b, inj_b = simulate(seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert inj_a == inj_b
    # 5 sensors x (2 spikes + stuck + dropout + drift), plus one methane gas event
    assert len(inj_a) == 5 * 5 + 1
    assert labels(a, inj_a).to_numpy().sum() > 0


def test_alerts_merge_consecutive_points():
    df = pd.DataFrame({"s": np.arange(20.0)}, index=IDX[:20])
    flags = pd.DataFrame({"s": [False] * 20}, index=df.index)
    flags.iloc[2:6, 0] = True
    flags.iloc[15, 0] = True
    alerts = to_alerts({"stuck": flags}, df)
    assert [(a.points, a.start) for a in alerts] == [(4, df.index[2]), (1, df.index[15])]


@pytest.mark.parametrize("seed", [1, 7])
def test_end_to_end_quality(seed):
    df, injected = simulate(seed=seed)
    results = run_all(df, SENSORS, baseline_points=3 * POINTS_PER_DAY)
    summary = score(results, injected, labels(df, injected))
    recall = summary.recall_by_kind()
    for kind in ("stuck", "dropout", "drift", "gas_event"):
        hit, total = recall[kind]
        assert hit == total, kind
    assert summary.precision >= 0.95


def test_gas_event_warned_before_limit():
    df, injected = simulate(seed=7)
    results = run_all(df, SENSORS, baseline_points=3 * POINTS_PER_DAY)
    event = next(a for a in injected if a.kind == "gas_event")
    window = slice(event.start, event.end)
    first_any = min(np.flatnonzero(r["ch4_pct"].iloc[window].to_numpy())[0]
                    for r in results.values() if r["ch4_pct"].iloc[window].any())
    first_limit = np.flatnonzero(results["limit"]["ch4_pct"].iloc[window].to_numpy())[0]
    assert first_any < first_limit
