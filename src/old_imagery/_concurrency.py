"""Concurrency policy for metadata discovery and resolved image payloads.

There are several materially different network workloads in this package:

* Google quadtree packets, Esri tilemap hops, centre-point capture metadata,
  and feature-service queries each calibrate independently;
* resolved immutable image payloads use the caller's real tiles to calibrate
  and continuously monitor the useful width of the current network path.

Measurements made in August 2026 found Google payloads peaking at 8 workers on
GCE and 16 on UNIGE Baobab, while Esri Wayback ranged from 32 on GCE to 104 on
Baobab. Sustained testing also found that Esri at 104 could oscillate between
roughly 123 and 241 tiles/s and lose 27% between early and late medians. A
static value, or one provider-wide value shared by unlike endpoints, is
therefore not robust across places or over time. Metadata measurements found
the same split: Esri point/tilemap requests peaked at 32 workers on GCE but
80-96 on Baobab, while the much slower feature service peaked near 24 and
occasionally returned incomplete HTTP-200 answers at wider settings.

Adaptation never issues a duplicate probe request. Calibration takes two
independent observations at each candidate width. Later requested tiles remain
chunked and measured: three consecutive material drops back off one window,
and a wider window is tried again only after a stable cooldown. State lives in
process memory for 30 minutes; nothing is written to disk and there is
deliberately no public or environment override.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import TypeVar

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


# Fixed pools remain only where adaptation would measure the wrong thing: the
# outer multi-zoom coordinator and the handful of large geometry payloads.
_METADATA_MAX_WORKERS = {"google": 16, "esri": 10}
_DEFAULT_METADATA_MAX_WORKERS = 10

# Endpoint-specific metadata candidates measured on GCE and UNIGE Baobab in
# August 2026. State keys are deliberately endpoint-specific: a fast tilemap
# service must never teach the slow ArcGIS feature service to use 80 workers.
_METADATA_WINDOWS = {
    "google-quadtree": (8, 12, 16, 24, 32),
    "esri-point": (16, 24, 32, 40, 48, 64, 80, 96),
    "esri-tilemap": (16, 24, 32, 40, 48, 64, 80, 96),
    "esri-feature": (10, 16, 24, 32),
}
_DEFAULT_METADATA_WINDOWS = (8, 12, 16)

# Raw image payload candidates, ordered from the conservative starting width
# to the largest empirically useful width. Explicit sequences make each step
# reviewable and keep Google from inheriting Esri's much larger ceiling.
_RAW_TILE_WINDOWS = {
    "google": (8, 12, 16),
    "esri": (32, 40, 48, 64, 80, 96, 104),
}
_DEFAULT_RAW_TILE_WINDOWS = (8, 12, 16)

# The HTTP transport must not bind Esri before its adaptive scheduler does.
# A high limit does not open connections by itself; only submitted work does.
RAW_TILE_CONNECTION_LIMIT = max(window[-1] for window in _RAW_TILE_WINDOWS.values())

_CALIBRATION_OBSERVATIONS = 2
_MIN_OBSERVATION_TASKS = 32
_MIN_MONITORING_TASKS = 64
_MONITORING_WAVES = 2
_MIN_THROUGHPUT_GAIN = 0.05
_MAX_LATENCY_RATIO = 1.25
_MATERIAL_DROP_RATIO = 0.80
_DROPS_BEFORE_BACKOFF = 3
_STABLE_BATCHES_BEFORE_RECOVERY = 8
_BASELINE_WEIGHT = 0.80
_LEARNED_TTL_SECONDS = 30 * 60


@dataclass(frozen=True)
class _Observation:
    workers: int
    throughput: float
    p95_latency: float
    acceptable: bool = True


@dataclass(frozen=True)
class _LearnedState:
    workers: int
    learned_at: float
    baseline_throughput: float | None = None
    baseline_p95_latency: float | None = None
    drop_streak: int = 0
    stable_batches: int = 0


_learned: dict[str, _LearnedState] = {}
_learned_lock = threading.Lock()


def workers_for(provider: str, n_tasks: int) -> int:
    """Fixed pool size for coordination and geometry payload calls.

    Never more threads than there is work: a three-tile AOI has no use for
    sixteen workers, and a pool of zero workers is not constructible.
    """
    cap = _METADATA_MAX_WORKERS.get(provider, _DEFAULT_METADATA_MAX_WORKERS)
    return max(1, min(cap, n_tasks))


def adaptive_metadata_map(
    workload: str,
    function: Callable[[_Input], _Output],
    items: Sequence[_Input],
    *,
    is_acceptable: Callable[[_Output], bool] | None = None,
) -> list[_Output]:
    """Map one homogeneous metadata workload with private adaptive state.

    ``is_acceptable`` lets the Esri feature path reject a width when ArcGIS
    returns an incomplete HTTP-200 answer. Results are still returned to the
    caller; completeness changes only what the controller learns.
    """
    windows = _METADATA_WINDOWS.get(workload, _DEFAULT_METADATA_WINDOWS)
    return _adaptive_map(
        f"metadata:{workload}", function, items, windows, is_acceptable=is_acceptable
    )


def adaptive_tile_map(
    provider: str,
    function: Callable[[_Input], _Output],
    items: Sequence[_Input],
) -> list[_Output]:
    """Map resolved raw-tile work with continuously adaptive concurrency.

    Order is preserved, matching :meth:`concurrent.futures.Executor.map`.
    Small calls use the conservative starting width. Calibration uses two
    observations per width over real requested work. Learned operation remains
    measured so sustained degradation can back off and later recover.
    """
    windows = _RAW_TILE_WINDOWS.get(provider, _DEFAULT_RAW_TILE_WINDOWS)
    return _adaptive_map(provider, function, items, windows)


def _adaptive_map(
    state_key: str,
    function: Callable[[_Input], _Output],
    items: Sequence[_Input],
    windows: tuple[int, ...],
    *,
    is_acceptable: Callable[[_Output], bool] | None = None,
) -> list[_Output]:
    if not items:
        return []

    state = _get_learned_state(state_key, windows)
    if state is not None:
        if len(items) < _MIN_MONITORING_TASKS:
            return _fixed_map(function, items, state.workers)
        return _map_with_feedback(
            state_key, function, items, windows, state, is_acceptable=is_acceptable
        )

    if len(windows) < 2 or len(items) < _minimum_calibration_tasks(windows):
        return _fixed_map(function, items, windows[0])

    outputs: list[_Output] = []
    next_item = 0
    best: _Observation | None = None
    widths_observed = 0

    with ThreadPoolExecutor(max_workers=min(windows[-1], len(items))) as pool:
        for workers in windows:
            observation_size = max(_MIN_OBSERVATION_TASKS, workers)
            required = observation_size * _CALIBRATION_OBSERVATIONS
            if len(items) - next_item < required:
                break

            observations = []
            for _ in range(_CALIBRATION_OBSERVATIONS):
                batch = items[next_item : next_item + observation_size]
                results, observation = _run_observed_batch(pool, function, batch, workers)
                observation = _with_acceptability(observation, results, is_acceptable)
                outputs.extend(results)
                observations.append(observation)
                next_item += observation_size

            combined = _median_observation(workers, observations)
            widths_observed += 1
            if best is None:
                best = combined
                if not combined.acceptable:
                    break
                continue

            throughput_gain = combined.throughput / best.throughput - 1.0
            latency_ratio = combined.p95_latency / max(best.p95_latency, 1e-9)
            if (
                combined.acceptable
                and throughput_gain >= _MIN_THROUGHPUT_GAIN
                and latency_ratio < _MAX_LATENCY_RATIO
            ):
                best = combined
                continue
            break

    assert best is not None
    state = _LearnedState(
        workers=best.workers,
        learned_at=time.monotonic(),
        baseline_throughput=best.throughput,
        baseline_p95_latency=best.p95_latency,
    )

    # One width says nothing about whether widening helped. Do not let such a
    # request prevent a later, larger call from calibrating properly.
    if widths_observed >= 2:
        _store_state(state_key, state)

    if next_item < len(items):
        rest = items[next_item:]
        if widths_observed >= 2 and len(rest) >= _MIN_MONITORING_TASKS:
            outputs.extend(
                _map_with_feedback(
                    state_key,
                    function,
                    rest,
                    windows,
                    state,
                    is_acceptable=is_acceptable,
                )
            )
        else:
            outputs.extend(_fixed_map(function, rest, best.workers))
    return outputs


def _map_with_feedback(
    state_key: str,
    function: Callable[[_Input], _Output],
    items: Sequence[_Input],
    windows: tuple[int, ...],
    state: _LearnedState,
    *,
    is_acceptable: Callable[[_Output], bool] | None = None,
) -> list[_Output]:
    outputs: list[_Output] = []
    next_item = 0

    with ThreadPoolExecutor(max_workers=min(windows[-1], len(items))) as pool:
        while next_item < len(items):
            remaining = len(items) - next_item
            recovery_workers = (
                _recovery_workers(state, windows) if remaining >= _MIN_MONITORING_TASKS else None
            )
            active_workers = recovery_workers or state.workers
            sample_size = min(
                remaining,
                max(_MIN_MONITORING_TASKS, active_workers * _MONITORING_WAVES),
            )
            batch = items[next_item : next_item + sample_size]
            results, observation = _run_observed_batch(pool, function, batch, active_workers)
            observation = _with_acceptability(observation, results, is_acceptable)
            outputs.extend(results)
            next_item += sample_size

            # A short tail completes at the selected width but is too noisy to
            # affect process-wide state.
            if sample_size < _MIN_MONITORING_TASKS:
                continue

            if recovery_workers is not None:
                state = _apply_recovery_observation(state, observation)
            else:
                state = _apply_regular_observation(state, observation, windows)
            _store_state(state_key, state)

    return outputs


def _apply_regular_observation(
    state: _LearnedState,
    observation: _Observation,
    windows: tuple[int, ...],
) -> _LearnedState:
    if not observation.acceptable:
        index = windows.index(state.workers)
        if index > 0:
            return _LearnedState(
                workers=windows[index - 1],
                learned_at=state.learned_at,
            )
        return replace(state, drop_streak=0, stable_batches=0)

    baseline = state.baseline_throughput
    if baseline is None:
        return replace(
            state,
            baseline_throughput=observation.throughput,
            baseline_p95_latency=observation.p95_latency,
            drop_streak=0,
            stable_batches=1,
        )

    if observation.throughput < baseline * _MATERIAL_DROP_RATIO:
        drop_streak = state.drop_streak + 1
        if drop_streak >= _DROPS_BEFORE_BACKOFF:
            index = windows.index(state.workers)
            if index > 0:
                return _LearnedState(
                    workers=windows[index - 1],
                    learned_at=state.learned_at,
                )
        return replace(state, drop_streak=drop_streak, stable_batches=0)

    updated_throughput = (
        _BASELINE_WEIGHT * baseline + (1.0 - _BASELINE_WEIGHT) * observation.throughput
    )
    prior_latency = state.baseline_p95_latency or observation.p95_latency
    updated_latency = (
        _BASELINE_WEIGHT * prior_latency + (1.0 - _BASELINE_WEIGHT) * observation.p95_latency
    )
    return replace(
        state,
        baseline_throughput=updated_throughput,
        baseline_p95_latency=updated_latency,
        drop_streak=0,
        stable_batches=state.stable_batches + 1,
    )


def _recovery_workers(state: _LearnedState, windows: tuple[int, ...]) -> int | None:
    index = windows.index(state.workers)
    if state.stable_batches < _STABLE_BATCHES_BEFORE_RECOVERY or index == len(windows) - 1:
        return None
    return windows[index + 1]


def _apply_recovery_observation(state: _LearnedState, observation: _Observation) -> _LearnedState:
    baseline = state.baseline_throughput
    baseline_latency = state.baseline_p95_latency
    if observation.acceptable and baseline is not None:
        throughput_gain = observation.throughput / baseline - 1.0
        latency_ratio = observation.p95_latency / max(baseline_latency or 1e-9, 1e-9)
        if throughput_gain >= _MIN_THROUGHPUT_GAIN and latency_ratio < _MAX_LATENCY_RATIO:
            return _LearnedState(
                workers=observation.workers,
                learned_at=state.learned_at,
                baseline_throughput=observation.throughput,
                baseline_p95_latency=observation.p95_latency,
            )

    # A rejected recovery probe has still completed useful requested work. It
    # simply restarts the stable cooldown at the prior width.
    return replace(state, drop_streak=0, stable_batches=0)


def _minimum_calibration_tasks(windows: tuple[int, ...]) -> int:
    return _CALIBRATION_OBSERVATIONS * sum(
        max(_MIN_OBSERVATION_TASKS, workers) for workers in windows[:2]
    )


def _median_observation(workers: int, observations: Sequence[_Observation]) -> _Observation:
    return _Observation(
        workers=workers,
        throughput=statistics.median(item.throughput for item in observations),
        p95_latency=statistics.median(item.p95_latency for item in observations),
        acceptable=all(item.acceptable for item in observations),
    )


def _with_acceptability(
    observation: _Observation,
    results: Sequence[_Output],
    is_acceptable: Callable[[_Output], bool] | None,
) -> _Observation:
    if is_acceptable is None:
        return observation
    return replace(observation, acceptable=all(is_acceptable(result) for result in results))


def _fixed_map(
    function: Callable[[_Input], _Output], items: Sequence[_Input], workers: int
) -> list[_Output]:
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(items)))) as pool:
        return list(pool.map(function, items))


def _run_observed_batch(
    pool: ThreadPoolExecutor,
    function: Callable[[_Input], _Output],
    items: Sequence[_Input],
    workers: int,
) -> tuple[list[_Output], _Observation]:
    limiter = threading.Semaphore(workers)

    def timed(item: _Input) -> tuple[_Output, float]:
        with limiter:
            started = time.perf_counter()
            result = function(item)
            return result, time.perf_counter() - started

    started = time.perf_counter()
    futures = [pool.submit(timed, item) for item in items]
    completed = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    results = [result for result, _latency in completed]
    latencies = sorted(latency for _result, latency in completed)
    p95 = latencies[int(0.95 * (len(latencies) - 1))]
    return results, _Observation(
        workers=workers,
        throughput=len(items) / max(elapsed, 1e-9),
        p95_latency=p95,
    )


def _get_learned_state(provider: str, windows: tuple[int, ...]) -> _LearnedState | None:
    now = time.monotonic()
    with _learned_lock:
        state = _learned.get(provider)
        if state is None:
            return None
        if now - state.learned_at > _LEARNED_TTL_SECONDS or state.workers not in windows:
            _learned.pop(provider, None)
            return None
        return state


def _get_learned(provider: str, windows: tuple[int, ...]) -> int | None:
    """Return the learned width; retained as a small private test helper."""
    state = _get_learned_state(provider, windows)
    return None if state is None else state.workers


def _store_state(provider: str, state: _LearnedState) -> None:
    with _learned_lock:
        _learned[provider] = state


def _remember(
    provider: str,
    workers: int,
    *,
    throughput: float | None = None,
    p95_latency: float | None = None,
) -> None:
    """Seed private process state, primarily for deterministic tests."""
    _store_state(
        provider,
        _LearnedState(
            workers=workers,
            learned_at=time.monotonic(),
            baseline_throughput=throughput,
            baseline_p95_latency=p95_latency,
        ),
    )


def _reset_learned_for_tests() -> None:
    """Clear process-local tuning state; intentionally private."""
    with _learned_lock:
        _learned.clear()
