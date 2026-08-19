"""Small caching HTTP client shared by the Google Earth and Esri backends."""

from __future__ import annotations

import hashlib
import os
import queue
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import httpx

from ._concurrency import RAW_TILE_CONNECTION_LIMIT


def _default_cache_dir() -> Path:
    """Where to cache responses, following each platform's own convention.

    ``~/.cache`` is an XDG convention and belongs to Linux only; using it
    everywhere would drop a stray dot-directory in the user's home on Windows
    and bypass the location macOS expects to be able to purge.

    ``OLD_IMAGERY_CACHE_DIR`` overrides all of it. It is read once, at import,
    because the result is a default argument value.
    """
    override = os.environ.get("OLD_IMAGERY_CACHE_DIR")
    if override:  # an empty value would otherwise resolve to the process cwd
        return Path(override)

    # Read into a plain str first: comparing sys.platform directly lets mypy
    # narrow to the platform it is checking on and call the other branches
    # unreachable, which is exactly the code that has to keep working elsewhere.
    platform = str(sys.platform)

    if platform == "win32":
        # LOCALAPPDATA rather than APPDATA: a cache is machine-local and must
        # not follow a roaming profile across machines.
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
        return root / "old-imagery" / "Cache"
    if platform == "darwin":
        return Path.home() / "Library" / "Caches" / "old-imagery"

    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "old-imagery"


DEFAULT_CACHE_DIR = _default_cache_dir()

_USER_AGENT = "old-imagery/0.1 (+https://github.com/angusmcb/old-imagery)"
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_BACKOFF = 0.5
_CACHE_BACKEND_ENV = "OLD_IMAGERY_CACHE_BACKEND"
_DEFAULT_CACHE_BACKEND = "sqlite"
_SQLITE_FILENAME = "responses.sqlite3"
_SQLITE_BUSY_TIMEOUT_MS = 5_000
_SQLITE_BATCH_MAX_ITEMS = 64
_SQLITE_BATCH_MAX_BYTES = 8 * 1024 * 1024
_SQLITE_BATCH_WAIT = 0.05
_SQLITE_MAX_QUEUE_BYTES = 64 * 1024 * 1024


class _CacheBackend(Protocol):
    def read(self, key: str, max_age: float | None) -> bytes | None: ...

    def write(self, key: str, data: bytes) -> None: ...

    def close(self) -> None: ...


class _FileCache:
    """The original one-response-per-file cache backend."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def _path_for(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / digest[:2] / digest[2:]

    def read(self, key: str, max_age: float | None) -> bytes | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            if max_age is not None and (time.time() - path.stat().st_mtime) > max_age:
                return None
            return path.read_bytes()
        except OSError:  # pragma: no cover - unreadable cache entry
            return None

    def write(self, key: str, data: bytes) -> None:
        path = self._path_for(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        except OSError:  # pragma: no cover - cache is best-effort
            pass

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class _PendingWrite:
    key: bytes
    data: bytes
    fetched_at: float


class _SqliteCache:
    """SQLite response cache with a bounded, write-behind worker."""

    def __init__(self, cache_dir: Path):
        self.path = cache_dir / _SQLITE_FILENAME
        self._queue: queue.Queue[_PendingWrite | None] = queue.Queue()
        self._queue_condition = threading.Condition()
        self._queued_bytes = 0
        self._pending: dict[bytes, _PendingWrite] = {}
        self._pending_lock = threading.Lock()
        self._read_connections: list[sqlite3.Connection] = []
        self._read_connections_lock = threading.Lock()
        self._local = threading.local()
        self._closing = False
        self._close_lock = threading.Lock()

        self._initialize()
        self._writer = threading.Thread(
            target=self._writer_loop,
            name="old-imagery-cache-writer",
            daemon=True,
        )
        self._writer.start()

    @staticmethod
    def _configure(connection: sqlite3.Connection) -> None:
        connection.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA synchronous=NORMAL")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=_SQLITE_BUSY_TIMEOUT_MS / 1000,
            check_same_thread=False,
        )
        self._configure(connection)
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    key BLOB PRIMARY KEY,
                    body BLOB NOT NULL,
                    fetched_at REAL NOT NULL
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _digest(key: str) -> bytes:
        return hashlib.sha256(key.encode()).digest()

    def _read_connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._connect()
            self._local.connection = connection
            with self._read_connections_lock:
                self._read_connections.append(connection)
        return connection

    def read(self, key: str, max_age: float | None) -> bytes | None:
        digest = self._digest(key)
        with self._pending_lock:
            pending = self._pending.get(digest)
        if pending is not None:
            if max_age is None or (time.time() - pending.fetched_at) <= max_age:
                return pending.data
            return None

        try:
            row = self._read_connection().execute(
                "SELECT body, fetched_at FROM responses WHERE key = ?",
                (sqlite3.Binary(digest),),
            ).fetchone()
        except sqlite3.Error:  # pragma: no cover - best-effort cache
            return None
        if row is None:
            return None
        if max_age is not None and (time.time() - float(row[1])) > max_age:
            return None
        return bytes(row[0])

    def write(self, key: str, data: bytes) -> None:
        digest = self._digest(key)
        pending = _PendingWrite(digest, data, time.time())
        with self._queue_condition:
            while (
                not self._closing
                and self._queued_bytes > 0
                and self._queued_bytes + len(data) > _SQLITE_MAX_QUEUE_BYTES
            ):
                self._queue_condition.wait()
            if self._closing:
                return
            self._queued_bytes += len(data)
            with self._pending_lock:
                self._pending[digest] = pending
            self._queue.put(pending)
            self._queue_condition.notify_all()

    def _finish_pending(self, batch: list[_PendingWrite]) -> None:
        with self._queue_condition:
            self._queued_bytes -= sum(len(item.data) for item in batch)
            self._queue_condition.notify_all()
        with self._pending_lock:
            for item in batch:
                if self._pending.get(item.key) is item:
                    self._pending.pop(item.key, None)

    def _writer_loop(self) -> None:
        connection = self._connect()
        try:
            while True:
                first = self._queue.get()
                if first is None:
                    self._queue.task_done()
                    return
                batch = [first]
                batch_bytes = len(first.data)
                deadline = time.monotonic() + _SQLITE_BATCH_WAIT
                while (
                    len(batch) < _SQLITE_BATCH_MAX_ITEMS
                    and batch_bytes < _SQLITE_BATCH_MAX_BYTES
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item = self._queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if item is None:
                        # close() only sends the sentinel after queue.join(),
                        # so this branch is defensive rather than expected.
                        self._queue.task_done()
                        break
                    batch.append(item)
                    batch_bytes += len(item.data)

                try:
                    connection.executemany(
                        """
                        INSERT INTO responses(key, body, fetched_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(key) DO UPDATE SET
                            body=excluded.body,
                            fetched_at=excluded.fetched_at
                        """,
                        [(sqlite3.Binary(item.key), item.data, item.fetched_at) for item in batch],
                    )
                    connection.commit()
                except sqlite3.Error:  # pragma: no cover - best-effort cache
                    connection.rollback()
                finally:
                    for _ in batch:
                        self._queue.task_done()
                    self._finish_pending(batch)
        finally:
            connection.close()

    def close(self) -> None:
        with self._close_lock:
            if self._closing:
                return
            with self._queue_condition:
                self._closing = True
        self._queue.join()
        self._queue.put(None)
        self._writer.join()
        with self._read_connections_lock:
            for connection in self._read_connections:
                connection.close()
            self._read_connections.clear()


def _cache_backend_name(value: str | None) -> Literal["file", "sqlite"]:
    name = (value or os.environ.get(_CACHE_BACKEND_ENV, _DEFAULT_CACHE_BACKEND)).lower()
    if name not in {"file", "sqlite"}:
        raise ValueError(f"Unknown cache backend {name!r}; expected 'file' or 'sqlite'")
    return name  # type: ignore[return-value]


class RequestFailed(Exception):
    """A request failed and could not be retried into success.

    Callers catch this rather than httpx's exception hierarchy, so a single
    slow or broken endpoint degrades one tile or one layer instead of aborting
    an entire availability or download call.
    """


class NotFound(RequestFailed):
    """The server returned 404 for an asset."""


class CachedHttpClient:
    """Thread-safe HTTP client with an on-disk response cache.

    Keyhole flatfile assets are addressed by epoch and are therefore immutable,
    so cached entries never need revalidating.  ``max_age`` exists for the
    handful of mutable endpoints (dbRoot, WMTS capabilities).
    """

    def __init__(
        self,
        cache_dir: str | os.PathLike | None = DEFAULT_CACHE_DIR,
        # Esri's metadata feature service can take tens of seconds to answer a
        # region query, so the default is generous rather than snappy.
        timeout: float = 90.0,
        retries: int = 3,
        *,
        cache_backend: str | None = None,
    ):
        self.retries = retries
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cache_backend = (
            _cache_backend_name(cache_backend) if self.cache_dir is not None else None
        )
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache: _CacheBackend = (
                _SqliteCache(self.cache_dir)
                if self.cache_backend == "sqlite"
                else _FileCache(self.cache_dir)
            )
        else:
            self._cache = _FileCache(Path())
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=RAW_TILE_CONNECTION_LIMIT,
                max_keepalive_connections=RAW_TILE_CONNECTION_LIMIT,
            ),
        )
    # -- cache helpers -----------------------------------------------------
    def _read_cache(self, key: str, max_age: float | None) -> bytes | None:
        if self.cache_dir is None:
            return None
        return self._cache.read(key, max_age)

    def _write_cache(self, key: str, data: bytes) -> None:
        if self.cache_dir is None:
            return
        self._cache.write(key, data)

    # -- requests ----------------------------------------------------------
    def _send(self, method: str, url: str, data: dict[str, str] | None) -> bytes:
        """Issue a request, retrying transient transport and 5xx/429 failures.

        Esri's tile servers routinely drop pooled keep-alive connections when
        several threads are in flight, which surfaces as RemoteProtocolError.
        """
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                if method == "GET":
                    response = self._client.get(url)
                else:
                    response = self._client.post(url, data=data)
                if response.status_code == 404:
                    raise NotFound(url)
                if response.status_code in _RETRY_STATUS and attempt < self.retries:
                    last = httpx.HTTPStatusError(
                        f"HTTP {response.status_code}", request=response.request, response=response
                    )
                    time.sleep(_BACKOFF * (2**attempt))
                    continue
                response.raise_for_status()
                return response.content
            except httpx.TransportError as exc:
                last = exc
                if attempt >= self.retries:
                    break
                time.sleep(_BACKOFF * (2**attempt))
            except httpx.HTTPStatusError as exc:
                raise RequestFailed(f"{method} {url} failed: {exc}") from exc
        raise RequestFailed(f"{method} {url} failed after {self.retries} retries: {last}") from last

    def get(self, url: str, *, max_age: float | None = None) -> bytes:
        cached = self._read_cache(url, max_age)
        if cached is not None:
            return cached
        data = self._send("GET", url, None)
        self._write_cache(url, data)
        return data

    def post(self, url: str, data: dict[str, str], *, max_age: float | None = None) -> bytes:
        key = url + "\0" + repr(sorted(data.items()))
        cached = self._read_cache(key, max_age)
        if cached is not None:
            return cached
        body = self._send("POST", url, data)
        self._write_cache(key, body)
        return body

    def close(self) -> None:
        try:
            self._client.close()
        finally:
            self._cache.close()

    def __enter__(self) -> CachedHttpClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
