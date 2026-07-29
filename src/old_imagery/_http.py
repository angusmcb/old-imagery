"""Small caching HTTP client shared by the Google Earth and Esri backends."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

import httpx

DEFAULT_CACHE_DIR = Path(
    os.environ.get("OLD_IMAGERY_CACHE_DIR", Path.home() / ".cache" / "old-imagery")
)

_USER_AGENT = "old-imagery/0.1 (+https://github.com/angusmcb/old-imagery)"
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
_BACKOFF = 0.5


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
    ):
        self.retries = retries
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
        )
        self._lock = threading.Lock()

    # -- cache helpers -----------------------------------------------------
    def _path_for(self, key: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / digest[:2] / digest[2:]

    def _read_cache(self, key: str, max_age: float | None) -> bytes | None:
        path = self._path_for(key)
        if path is None or not path.exists():
            return None
        if max_age is not None and (time.time() - path.stat().st_mtime) > max_age:
            return None
        try:
            return path.read_bytes()
        except OSError:  # pragma: no cover - unreadable cache entry
            return None

    def _write_cache(self, key: str, data: bytes) -> None:
        path = self._path_for(key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        except OSError:  # pragma: no cover - cache is best-effort
            pass

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
        self._client.close()

    def __enter__(self) -> CachedHttpClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
