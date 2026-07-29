"""Offline tests for the caching HTTP client's retry and failure contract."""

from __future__ import annotations

import httpx
import pytest

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
    def build(script, retries=3):
        c = CachedHttpClient(tmp_path, retries=retries)
        transport = FakeTransport(script)
        c._client = httpx.Client(transport=transport)
        c._transport = transport
        monkeypatch.setattr("old_imagery._http.time.sleep", lambda _s: None)
        return c

    return build


URL = "https://example.invalid/asset"


def test_successful_request_is_cached(client) -> None:
    c = client([200])
    assert c.get(URL) == b"payload"
    assert c.get(URL) == b"payload"
    assert c._transport.calls == 1  # second call served from disk


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
