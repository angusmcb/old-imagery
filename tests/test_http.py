"""Offline tests for the caching HTTP client's retry and failure contract."""

from __future__ import annotations

import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from old_imagery._concurrency import RAW_TILE_CONNECTION_LIMIT
from old_imagery._http import CachedHttpClient, NotFound, RequestFailed


class FakeTransport(httpx.BaseTransport):
    """Replays a scripted sequence of responses / exceptions."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def handle_request(self, request):
        self.calls += 1
        item = self.script.pop(0) if self.script else self.script_default()
        if isinstance(item, Exception):
            raise item
        return httpx.Response(item, content=b"payload", request=request)

    @staticmethod
    def script_default():
        return 200


@pytest.fixture
def client(tmp_path, monkeypatch):
    clients = []

    def build(script, retries=3, backend=None):
        c = CachedHttpClient(tmp_path, retries=retries, cache_backend=backend)
        transport = FakeTransport(script)
        c._client = httpx.Client(transport=transport)
        c._transport = transport
        clients.append(c)
        monkeypatch.setattr("old_imagery._http.time.sleep", lambda _s: None)
        return c

    yield build
    for c in clients:
        c.close()


URL = "https://example.invalid/asset"


def test_successful_request_is_cached(client) -> None:
    c = client([200])
    assert c.get(URL) == b"payload"
    assert c.get(URL) == b"payload"
    assert c._transport.calls == 1  # second call served from disk


@pytest.mark.parametrize("backend", ["file", "sqlite"])
def test_successful_request_is_cached_for_each_backend(client, backend) -> None:
    c = client([200], backend=backend)
    assert c.get(URL) == b"payload"
    assert c.get(URL) == b"payload"
    assert c._transport.calls == 1


def test_transport_error_is_retried_then_succeeds(client) -> None:
    c = client([httpx.ReadTimeout("slow"), httpx.ConnectError("nope"), 200])
    assert c.get(URL) == b"payload"
    assert c._transport.calls == 3


def test_read_timeout_becomes_request_failed(client) -> None:
    """A timeout must surface as RequestFailed, not an httpx exception.

    Callers catch RequestFailed to degrade one tile or layer; an httpx error
    escaping here would abort an entire availability or download call.
    """
    c = client([httpx.ReadTimeout("slow")] * 4)
    with pytest.raises(RequestFailed):
        c.get(URL)
    assert c._transport.calls == 4  # initial attempt plus three retries


def test_retryable_status_is_retried(client) -> None:
    c = client([503, 502, 200])
    assert c.get(URL) == b"payload"
    assert c._transport.calls == 3


def test_persistent_server_error_becomes_request_failed(client) -> None:
    c = client([500] * 4)
    with pytest.raises(RequestFailed):
        c.get(URL)


def test_404_is_not_found_and_not_retried(client) -> None:
    c = client([404, 200])
    with pytest.raises(NotFound):
        c.get(URL)
    assert c._transport.calls == 1


def test_not_found_is_a_request_failed(client) -> None:
    """So a single `except RequestFailed` covers every failure mode."""
    c = client([404])
    with pytest.raises(RequestFailed):
        c.get(URL)


def test_client_errors_are_not_retried(client) -> None:
    c = client([400, 200])
    with pytest.raises(RequestFailed):
        c.get(URL)
    assert c._transport.calls == 1


def test_failures_are_not_cached(client) -> None:
    c = client([500] * 4 + [200])
    with pytest.raises(RequestFailed):
        c.get(URL)
    assert c.get(URL) == b"payload"


def test_post_bodies_key_the_cache_separately(client) -> None:
    c = client([200, 200])
    c.post(URL, {"a": "1"})
    c.post(URL, {"a": "2"})
    assert c._transport.calls == 2
    c.post(URL, {"a": "1"})
    assert c._transport.calls == 2  # first body now cached


def test_cache_can_be_disabled(client, tmp_path) -> None:
    c = CachedHttpClient(None)
    c._client = httpx.Client(transport=FakeTransport([200, 200]))
    assert c.get(URL) == b"payload"
    assert c.get(URL) == b"payload"


def test_sqlite_cache_is_one_container_file(tmp_path) -> None:
    c = CachedHttpClient(tmp_path, cache_backend="sqlite")
    c._write_cache(URL, b"payload")
    c.close()

    database = tmp_path / "responses.sqlite3"
    assert database.exists()
    assert not list(tmp_path.glob("[0-9a-f][0-9a-f]"))
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT body FROM responses").fetchone() == (b"payload",)


def test_sqlite_cache_close_flushes_queued_writes(tmp_path) -> None:
    c = CachedHttpClient(tmp_path, cache_backend="sqlite")
    c._write_cache(URL, b"payload")
    c.close()

    reopened = CachedHttpClient(tmp_path, cache_backend="sqlite")
    try:
        assert reopened._read_cache(URL, None) == b"payload"
    finally:
        reopened.close()


def test_sqlite_cache_reads_pending_write_before_commit(tmp_path) -> None:
    c = CachedHttpClient(tmp_path, cache_backend="sqlite")
    try:
        c._write_cache(URL, b"payload")
        assert c._read_cache(URL, None) == b"payload"
    finally:
        c.close()


def test_sqlite_cache_concurrent_writes_are_durable(tmp_path) -> None:
    c = CachedHttpClient(tmp_path, cache_backend="sqlite")
    entries = [(f"{URL}/{index}", f"payload-{index}".encode()) for index in range(256)]
    try:
        with ThreadPoolExecutor(max_workers=32) as pool:
            list(pool.map(lambda item: c._write_cache(*item), entries))
    finally:
        c.close()

    reopened = CachedHttpClient(tmp_path, cache_backend="sqlite")
    try:
        assert [reopened._read_cache(key, None) for key, _ in entries] == [
            body for _, body in entries
        ]
    finally:
        reopened.close()


def test_cache_backend_can_be_selected_from_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OLD_IMAGERY_CACHE_BACKEND", "file")
    file_client = CachedHttpClient(tmp_path)
    try:
        assert file_client.cache_backend == "file"
    finally:
        file_client.close()

    monkeypatch.setenv("OLD_IMAGERY_CACHE_BACKEND", "sqlite")
    sqlite_client = CachedHttpClient(tmp_path / "sqlite")
    try:
        assert sqlite_client.cache_backend == "sqlite"
    finally:
        sqlite_client.close()


def test_unknown_cache_backend_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="expected 'file' or 'sqlite'"):
        CachedHttpClient(tmp_path, cache_backend="unknown")


def test_transport_capacity_does_not_bind_the_adaptive_raw_tile_ceiling(monkeypatch) -> None:
    seen = {}

    class DummyClient:
        def close(self):
            pass

    def build_client(**kwargs):
        seen.update(kwargs)
        return DummyClient()

    monkeypatch.setattr("old_imagery._http.httpx.Client", build_client)
    CachedHttpClient(None).close()
    limits = seen["limits"]
    assert limits.max_connections == RAW_TILE_CONNECTION_LIMIT
    assert limits.max_keepalive_connections == RAW_TILE_CONNECTION_LIMIT


# --------------------------------------------------------------------------
# default cache directory
# --------------------------------------------------------------------------
def _resolve(monkeypatch, platform, env):
    from old_imagery import _http

    for name in ("XDG_CACHE_HOME", "LOCALAPPDATA", "OLD_IMAGERY_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "platform", platform)
    return _http._default_cache_dir()


def test_cache_dir_follows_each_platform_convention(monkeypatch) -> None:
    """~/.cache is an XDG convention; it is wrong on macOS and on Windows."""
    linux = _resolve(monkeypatch, "linux", {})
    assert linux.parts[-2:] == (".cache", "old-imagery")

    mac = _resolve(monkeypatch, "darwin", {})
    assert mac.parts[-3:] == ("Library", "Caches", "old-imagery")

    win = _resolve(monkeypatch, "win32", {"LOCALAPPDATA": str(Path.home() / "AppData" / "Local")})
    assert win.parts[-3:] == ("Local", "old-imagery", "Cache")


def test_cache_dir_honours_xdg_and_localappdata(monkeypatch) -> None:
    assert _resolve(monkeypatch, "linux", {"XDG_CACHE_HOME": "/xdg"}) == Path("/xdg/old-imagery")
    win = _resolve(monkeypatch, "win32", {"LOCALAPPDATA": "/appdata"})
    assert win == Path("/appdata/old-imagery/Cache")


def test_cache_dir_falls_back_when_localappdata_is_unset(monkeypatch) -> None:
    win = _resolve(monkeypatch, "win32", {})
    assert win.parts[-4:] == ("AppData", "Local", "old-imagery", "Cache")


def test_explicit_override_wins_on_every_platform(monkeypatch) -> None:
    for platform in ("linux", "darwin", "win32"):
        assert _resolve(monkeypatch, platform, {"OLD_IMAGERY_CACHE_DIR": "/custom"}) == Path(
            "/custom"
        )


def test_empty_override_does_not_put_the_cache_in_the_cwd(monkeypatch) -> None:
    """Path("") resolves to ".", which would scatter caches wherever you ran from."""
    resolved = _resolve(monkeypatch, "linux", {"OLD_IMAGERY_CACHE_DIR": ""})
    assert resolved != Path()
    assert resolved.is_absolute()
