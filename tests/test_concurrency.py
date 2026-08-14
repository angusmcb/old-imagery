from __future__ import annotations

import inspect
import threading
import time

import pytest

import old_imagery
from old_imagery import _concurrency


@pytest.fixture(autouse=True)
def clear_learned_state():
    _concurrency._reset_learned_for_tests()
    yield
    _concurrency._reset_learned_for_tests()


def test_current_metadata_limits_remain_conservative() -> None:
    # Retained only for the fixed outer coordinator and geometry payloads.
    assert _concurrency.workers_for("google", 10_000) == 16
    assert _concurrency.workers_for("esri", 10_000) == 10
    assert _concurrency._METADATA_WINDOWS == {
        "google-quadtree": (8, 12, 16, 24, 32),
        "esri-point": (16, 24, 32, 40, 48, 64, 80, 96),
        "esri-tilemap": (16, 24, 32, 40, 48, 64, 80, 96),
        "esri-feature": (10, 16, 24, 32),
    }
    assert _concurrency.RAW_TILE_CONNECTION_LIMIT == 104


@pytest.mark.parametrize(
    ("workload", "expected"),
    [
        ("google-quadtree", 8),
        ("esri-point", 16),
        ("esri-tilemap", 16),
        ("esri-feature", 10),
    ],
)
def test_small_metadata_calls_use_the_endpoint_start(
    monkeypatch, workload: str, expected: int
) -> None:
    seen = []

    def fixed(function, items, workers):
        seen.append(workers)
        return [function(item) for item in items]

    monkeypatch.setattr(_concurrency, "_fixed_map", fixed)
    values = list(range(8))
    assert _concurrency.adaptive_metadata_map(workload, lambda value: value, values) == values
    assert seen == [expected]


def test_metadata_workloads_learn_independently() -> None:
    point = _concurrency._METADATA_WINDOWS["esri-point"]
    tilemap = _concurrency._METADATA_WINDOWS["esri-tilemap"]
    _concurrency._remember("metadata:esri-point", 64)
    _concurrency._remember("metadata:esri-tilemap", 32)
    assert _concurrency._get_learned("metadata:esri-point", point) == 64
    assert _concurrency._get_learned("metadata:esri-tilemap", tilemap) == 32


def test_metadata_calibration_uses_its_endpoint_windows(monkeypatch) -> None:
    throughputs = {16: 100.0, 24: 140.0, 32: 130.0}
    seen = []

    def run_batch(pool, function, items, workers):
        seen.append(workers)
        return [function(item) for item in items], _concurrency._Observation(
            workers, throughputs[workers], 0.2
        )

    monkeypatch.setattr(_concurrency, "_run_observed_batch", run_batch)
    values = list(range(500))
    assert _concurrency.adaptive_metadata_map("esri-point", lambda value: value, values) == values
    assert seen[:6] == [16, 16, 24, 24, 32, 32]
    windows = _concurrency._METADATA_WINDOWS["esri-point"]
    assert _concurrency._get_learned("metadata:esri-point", windows) == 24


def test_incomplete_feature_calibration_rejects_the_wider_window(monkeypatch) -> None:
    seen = []

    def run_batch(pool, function, items, workers):
        seen.append(workers)
        complete = workers == 10
        return [complete for _item in items], _concurrency._Observation(workers, 200.0, 0.2)

    monkeypatch.setattr(_concurrency, "_run_observed_batch", run_batch)
    _concurrency.adaptive_metadata_map(
        "esri-feature", lambda _value: True, list(range(320)), is_acceptable=bool
    )
    assert seen[:4] == [10, 10, 16, 16]
    windows = _concurrency._METADATA_WINDOWS["esri-feature"]
    assert _concurrency._get_learned("metadata:esri-feature", windows) == 10


def test_incomplete_monitored_feature_batch_immediately_backs_off(monkeypatch) -> None:
    windows = _concurrency._METADATA_WINDOWS["esri-feature"]
    _concurrency._remember("metadata:esri-feature", 24, throughput=100.0, p95_latency=1.0)

    def run_batch(pool, function, items, workers):
        return [False for _item in items], _concurrency._Observation(workers, 200.0, 0.5)

    monkeypatch.setattr(_concurrency, "_run_observed_batch", run_batch)
    _concurrency.adaptive_metadata_map(
        "esri-feature", lambda _value: True, list(range(64)), is_acceptable=bool
    )
    assert _concurrency._get_learned("metadata:esri-feature", windows) == 16


@pytest.mark.parametrize(("provider", "expected"), [("google", 8), ("esri", 32)])
def test_small_raw_tile_calls_use_the_conservative_start(
    monkeypatch, provider: str, expected: int
) -> None:
    seen = []

    def fixed(function, items, workers):
        seen.append(workers)
        return [function(item) for item in items]

    monkeypatch.setattr(_concurrency, "_fixed_map", fixed)
    assert _concurrency.adaptive_tile_map(provider, lambda value: value * 2, list(range(8))) == [
        value * 2 for value in range(8)
    ]
    assert seen == [expected]


def test_calibration_uses_two_observations_stops_at_knee_and_remembers(monkeypatch) -> None:
    throughputs = {8: 100.0, 12: 125.0, 16: 110.0}
    observed = []

    def run_batch(pool, function, items, workers):
        observed.append((workers, list(items)))
        return [function(item) for item in items], _concurrency._Observation(
            workers, throughputs[workers], 0.1
        )

    monkeypatch.setattr(_concurrency, "_run_observed_batch", run_batch)
    values = list(range(400))
    assert _concurrency.adaptive_tile_map("google", lambda value: value + 1, values) == [
        value + 1 for value in values
    ]
    assert [workers for workers, _items in observed[:6]] == [8, 8, 12, 12, 16, 16]
    assert all(workers == 12 for workers, _items in observed[6:])
    assert sorted(item for _workers, items in observed for item in items) == values
    assert _concurrency._get_learned("google", (8, 12, 16)) == 12


def test_two_observations_prevent_one_noisy_sample_deciding_the_width(monkeypatch) -> None:
    samples = {
        8: [150.0, 100.0],
        12: [140.0, 145.0],
        16: [130.0, 135.0],
    }

    def run_batch(pool, function, items, workers):
        throughput = samples[workers].pop(0) if samples[workers] else 142.0
        return [function(item) for item in items], _concurrency._Observation(
            workers, throughput, 0.1
        )

    monkeypatch.setattr(_concurrency, "_run_observed_batch", run_batch)
    _concurrency.adaptive_tile_map("google", lambda value: value, list(range(256)))
    assert _concurrency._get_learned("google", (8, 12, 16)) == 12


def test_esri_calibration_stops_before_a_slower_wider_window(monkeypatch) -> None:
    throughputs = {32: 100.0, 40: 130.0, 48: 128.0}
    observed = []

    def run_batch(pool, function, items, workers):
        observed.append(workers)
        return [function(item) for item in items], _concurrency._Observation(
            workers, throughputs[workers], 0.2
        )

    monkeypatch.setattr(_concurrency, "_run_observed_batch", run_batch)
    _concurrency.adaptive_tile_map("esri", lambda value: value, list(range(500)))
    assert observed[:6] == [32, 32, 40, 40, 48, 48]
    assert all(workers == 40 for workers in observed[6:])
    assert _concurrency._get_learned("esri", (32, 40, 48, 64, 80, 96, 104)) == 40


def test_medium_call_does_not_remember_an_untested_starting_width(monkeypatch) -> None:
    seen = []

    def fixed(function, items, workers):
        seen.append((workers, len(items)))
        return [function(item) for item in items]

    monkeypatch.setattr(_concurrency, "_fixed_map", fixed)
    _concurrency.adaptive_tile_map("esri", lambda value: value, list(range(80)))
    assert seen == [(32, 80)]
    assert _concurrency._get_learned("esri", (32, 40, 48, 64, 80, 96, 104)) is None


def test_three_consecutive_material_drops_back_off_one_window(monkeypatch) -> None:
    workers_seen = []

    def run_batch(pool, function, items, workers):
        workers_seen.append(workers)
        throughput = 150.0 if workers == 104 else 120.0
        return [function(item) for item in items], _concurrency._Observation(
            workers, throughput, 0.5
        )

    monkeypatch.setattr(_concurrency, "_run_observed_batch", run_batch)
    _concurrency._remember("esri", 104, throughput=200.0, p95_latency=0.5)
    values = list(range(832))
    assert _concurrency.adaptive_tile_map("esri", lambda value: value, values) == values
    assert workers_seen[:4] == [104, 104, 104, 96]
    assert _concurrency._get_learned("esri", (32, 40, 48, 64, 80, 96, 104)) == 96


def test_nonconsecutive_drops_do_not_back_off(monkeypatch) -> None:
    throughputs = iter((70.0, 100.0, 70.0, 100.0))

    def run_batch(pool, function, items, workers):
        return [function(item) for item in items], _concurrency._Observation(
            workers, next(throughputs), 0.1
        )

    monkeypatch.setattr(_concurrency, "_run_observed_batch", run_batch)
    _concurrency._remember("google", 16, throughput=100.0, p95_latency=0.1)
    _concurrency.adaptive_tile_map("google", lambda value: value, list(range(256)))
    assert _concurrency._get_learned("google", (8, 12, 16)) == 16


def test_stable_cooldown_accepts_a_useful_recovery_probe(monkeypatch) -> None:
    state = _concurrency._LearnedState(
        workers=8,
        learned_at=time.monotonic(),
        baseline_throughput=100.0,
        baseline_p95_latency=0.1,
        stable_batches=_concurrency._STABLE_BATCHES_BEFORE_RECOVERY,
    )
    _concurrency._store_state("google", state)
    seen = []

    def run_batch(pool, function, items, workers):
        seen.append(workers)
        return [function(item) for item in items], _concurrency._Observation(workers, 110.0, 0.1)

    monkeypatch.setattr(_concurrency, "_run_observed_batch", run_batch)
    _concurrency.adaptive_tile_map("google", lambda value: value, list(range(64)))
    assert seen == [12]
    assert _concurrency._get_learned("google", (8, 12, 16)) == 12


def test_stable_cooldown_rejects_an_unhelpful_recovery_probe(monkeypatch) -> None:
    state = _concurrency._LearnedState(
        workers=8,
        learned_at=time.monotonic(),
        baseline_throughput=100.0,
        baseline_p95_latency=0.1,
        stable_batches=_concurrency._STABLE_BATCHES_BEFORE_RECOVERY,
    )
    _concurrency._store_state("google", state)

    def run_batch(pool, function, items, workers):
        return [function(item) for item in items], _concurrency._Observation(workers, 103.0, 0.1)

    monkeypatch.setattr(_concurrency, "_run_observed_batch", run_batch)
    _concurrency.adaptive_tile_map("google", lambda value: value, list(range(64)))
    learned = _concurrency._get_learned_state("google", (8, 12, 16))
    assert learned is not None
    assert learned.workers == 8
    assert learned.stable_batches == 0


def test_learned_width_expires_without_disk_state(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr(_concurrency.time, "monotonic", lambda: now[0])
    _concurrency._remember("google", 12)
    assert _concurrency._get_learned("google", (8, 12, 16)) == 12
    now[0] += _concurrency._LEARNED_TTL_SECONDS + 1
    assert _concurrency._get_learned("google", (8, 12, 16)) is None


def test_parallel_map_never_exceeds_its_active_window() -> None:
    lock = threading.Lock()
    active = 0
    peak = 0

    def work(value):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.002)
        with lock:
            active -= 1
        return value

    assert _concurrency.adaptive_tile_map("google", work, list(range(64))) == list(range(64))
    assert 1 < peak <= 8


def test_public_download_functions_do_not_expose_a_concurrency_override() -> None:
    for function in (old_imagery.download, old_imagery.download_tiles):
        parameters = inspect.signature(function).parameters
        assert "concurrency" not in parameters
        assert "max_workers" not in parameters
