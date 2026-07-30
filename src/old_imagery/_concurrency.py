"""How many requests to keep in flight, and why it is not a caller's choice.

Every parallel section in this package is **network-bound**, so concurrency is
sized by what the remote service tolerates rather than by the local machine.
That is the opposite of upstream's ``--concurrency``, which defaults to
ALL_CPUS.

CPU count is the wrong basis here, by about two orders of magnitude. The only
appreciable CPU work is JPEG decode in :func:`old_imagery.api._decode_image`.
Measured on a 10-CPU machine, decoding a 256x256 RGB tile of noise:

    threads    1     2     4     6     8    16    32
    tiles/s  2347  4291  7373  7116  7054  7052  6800

Two things follow. Decode does release the GIL, but it saturates at four
threads and degrades slightly beyond -- and at ~7,000 tiles/s it costs ~0.4 ms
per tile against tens of milliseconds to fetch one. Decode is roughly 1% of the
per-tile cost, and any width chosen for the network already exceeds the width
decode can use. So one pool sized for the network is right, and a separate
CPU-sized pool would be machinery for nothing.

The per-provider caps below are what the *services* tolerate:

``esri``
    Upstream caps ``--provider=Wayback`` at 10, having determined empirically
    that going wider made Wayback slower. See the ``--concurrency`` option in
    Mbucari/GEHistoricalImagery.

``google``
    Measured here against the live service, 30 tiles at zoom 17, cache
    disabled: 4 workers 20.1 tiles/s, 8 workers 22.9, 16 workers 28.8, 32
    workers 28.6. Throughput is flat from 16 onward. The 32-worker figure is
    soft -- with only 30 tiles a 32-wide pool never fully engages -- so treat
    this as "16 is at or past the knee", not as a sharp optimum.

Both network figures are softer than the decode table above, which was
measured locally and is reproducible. A run taken on a degraded link measures
the *link*, not the service: an attempt to re-measure Esri's two availability
paths at 10 vs 16 wide returned 1.8x slower for one path and 1.9x faster for
the other on the same area -- opposite conclusions of equal size, which is what
noise looks like. Esri's 10 therefore rests on upstream's independent
measurement rather than on anything reproduced here. Both caps are worth
revisiting with many repeats on a stable connection; neither is load-bearing
enough for correctness that a wrong value does more than cost throughput.

Both numbers sit well inside the HTTP client's own ``max_connections=64``
(see :mod:`old_imagery._http`), so the pool is what binds, not the transport.

Deliberately not adaptive. A ramp-up probe would have to *find* the service's
limit by exceeding it, and Google's terms prohibit bulk feeds -- the polite
client is the one that never goes looking for the ceiling. These caps also
travel: a slower link is bounded by bandwidth long before it is bounded here.
"""

from __future__ import annotations

# Requests in flight per provider. Not exported: callers cannot usefully pick a
# value the service has already been measured to dislike.
_MAX_WORKERS = {"google": 16, "esri": 10}

# Used when the provider is unknown -- the lower of the two, so an unrecognised
# service is treated as the more delicate one.
_DEFAULT_MAX_WORKERS = 10


def workers_for(provider: str, n_tasks: int) -> int:
    """Pool size for ``n_tasks`` network calls against ``provider``.

    Never more threads than there is work: a three-tile AOI has no use for
    sixteen workers, and the idle ones still cost a thread each.
    """
    cap = _MAX_WORKERS.get(provider, _DEFAULT_MAX_WORKERS)
    return max(1, min(cap, n_tasks))
