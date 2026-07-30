"""Google Earth database (dbRoot) client.

Ported from ``LibGoogleEarth.DbRoot`` in Mbucari/GEHistoricalImagery.
"""

from __future__ import annotations

import datetime as _dt
import threading
from dataclasses import dataclass
from enum import Enum

from . import _proto
from ._http import CachedHttpClient, RequestFailed
from ._keyhole import (
    MIN_JPEG_DATE,
    ROOT,
    BinaryNode,
    BinaryPacket,
    KeyholeTile,
    decode_date,
    decompress,
    decrypt,
    parse_binary_packet,
)

_DBROOT_MAX_AGE = 24 * 3600  # dbRoot advertises the current quadtree epoch


class Database(str, Enum):
    """A Google Earth database this port can actually read.

    Keyhole also serves ``sky``, ``moon`` and ``mars``, and the dbRoot and
    quadtree wire formats are the same for all of them. They are absent here
    because none is a time machine: they carry no imagery-history layer, so
    :func:`~old_imagery.availability` has no dates to report and
    :func:`~old_imagery.download` has nothing to select between. Listing them
    would advertise support this port does not implement -- see the "Notes and
    limitations" section of README.md.
    """

    DEFAULT = "default"
    TIME_MACHINE = "tm"


@dataclass(frozen=True)
class DatedTile:
    """A single Keyhole image tile from a particular capture date."""

    tile: KeyholeTile
    date: _dt.date | None
    provider: int
    epoch: int
    asset_url: str


class DbRoot:
    """A Google Earth database instance."""

    @property
    def grid(self):
        from ._region import KeyholeGrid

        return KeyholeGrid()

    def __init__(self, database: Database, client: CachedHttpClient):
        self.database = database
        self._client = client
        self._packet_cache: dict[str, object] = {}
        self._key_locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

        raw = client.get(self._dbroot_url(database), max_age=_DBROOT_MAX_AGE)
        enc = _proto.EncryptedDbRootProto.FromString(raw)
        self._key = enc.encryption_data
        self.proto = _proto.DbRootProto.FromString(decompress(decrypt(enc.dbroot_data, self._key)))

    # -- construction ------------------------------------------------------
    @staticmethod
    def _dbroot_url(database: Database) -> str:
        if database is Database.DEFAULT:
            return "https://khmdb.google.com/dbRoot.v5?&hl=en&gl=us&output=proto"
        return f"https://khmdb.google.com/dbRoot.v5?db={database.value}&hl=en&gl=us&output=proto"

    @property
    def quadtree_version(self) -> int:
        return self.proto.database_version.quadtree_version

    # -- transport ---------------------------------------------------------
    def _download(self, url: str) -> bytes:
        return bytes(decrypt(self._client.get(url), self._key))

    def _packet_url(self, tile: KeyholeTile, epoch: int) -> str:
        if self.database is Database.DEFAULT:
            return f"https://kh.google.com/flatfile?q2-{tile.path}-q.{epoch}"
        return (
            f"https://khmdb.google.com/flatfile?db={self.database.value}&qp-{tile.path}-q.{epoch}"
        )

    def _get_packet(self, tile: KeyholeTile, epoch: int):
        data = decompress(self._download(self._packet_url(tile, epoch)))
        if self.database is Database.DEFAULT:
            return parse_binary_packet(data, tile.is_root)
        return _proto.QuadtreePacket.FromString(data)

    # -- quadtree walk -----------------------------------------------------
    @staticmethod
    def _node_at(packet, subindex: int):
        if isinstance(packet, BinaryPacket):
            return packet.node_at(subindex)
        for sparse in packet.sparsequadtreenode:
            if sparse.index == subindex:
                return sparse.Node
        return None

    def _cached_packet(self, tile: KeyholeTile, epoch: int):
        key = tile.path
        with self._lock:
            if key in self._packet_cache:
                return self._packet_cache[key]
            key_lock = self._key_locks.setdefault(key, threading.Lock())

        # Hold a per-packet lock so concurrent walkers that need the same
        # packet wait for one fetch instead of all issuing their own.
        with key_lock:
            with self._lock:
                if key in self._packet_cache:
                    return self._packet_cache[key]
            try:
                packet = self._get_packet(tile, epoch)
            except (RequestFailed, ValueError):
                packet = None
            with self._lock:
                self._packet_cache[key] = packet
                self._key_locks.pop(key, None)
            return packet

    def _packet_for(self, tile: KeyholeTile):
        """Walk root -> ... -> the packet that contains ``tile``."""
        packet = self._cached_packet(ROOT, self.quadtree_version)
        if packet is None:
            return None

        for path in tile.index_paths():
            child = self._node_at(packet, path.subindex)
            epoch = getattr(child, "cache_node_epoch", 0) if child is not None else 0
            if not epoch:
                return None
            packet = self._cached_packet(path, epoch)
            if packet is None:
                return None
        return packet

    def get_node(self, tile: KeyholeTile):
        packet = self._packet_for(tile)
        if packet is None:
            return None
        return self._node_at(packet, tile.subindex)

    def clear_cache(self) -> None:
        with self._lock:
            self._packet_cache.clear()

    # -- imagery -----------------------------------------------------------
    def dated_tiles(self, tile: KeyholeTile) -> list[DatedTile]:
        """All historical image tiles available for ``tile``."""
        node = self.get_node(tile)
        if node is None:
            return []
        return list(self._dated_tiles_for_node(tile, node))

    def _dated_tiles_for_node(self, tile: KeyholeTile, node):
        # Binary (default-database) nodes carry a single undated imagery layer.
        if isinstance(node, BinaryNode):
            if node.has_image:
                for layer in node.layers:
                    if layer.type == _proto.LAYER_TYPE_IMAGERY:
                        yield self._default_imagery(tile, layer, None)
                        return
            return

        history = None
        imagery = None
        for layer in node.layer:
            if layer.type == _proto.LAYER_TYPE_IMAGERY_HISTORY:
                history = layer
            elif layer.type == _proto.LAYER_TYPE_IMAGERY:
                imagery = layer

        if history is None or not history.dates_layer.dated_tile:
            if imagery is not None:
                yield self._default_imagery(tile, imagery, None)
            return

        for dated in history.dates_layer.dated_tile:
            if dated.date <= MIN_JPEG_DATE:
                continue
            date = decode_date(dated.date)
            if date is None:
                continue
            if dated.provider != 0:
                yield DatedTile(
                    tile=tile,
                    date=date,
                    provider=dated.provider,
                    epoch=dated.dated_tile_epoch,
                    asset_url=(
                        f"https://khmdb.google.com/flatfile?db={self.database.value}"
                        f"&f1-{tile.path}-i.{dated.dated_tile_epoch}-{dated.date:x}"
                    ),
                )
            elif imagery is not None:
                # provider == 0 means the imagery for that date is the default
                # imagery, which lives in the Imagery layer instead.
                yield self._default_imagery(tile, imagery, date)

    @staticmethod
    def _default_imagery(tile: KeyholeTile, layer, date: _dt.date | None) -> DatedTile:
        return DatedTile(
            tile=tile,
            date=date,
            provider=0,
            epoch=layer.layer_epoch,
            asset_url=f"https://kh.google.com/flatfile?f1-{tile.path}-i.{layer.layer_epoch}",
        )

    def download_tile_image(self, dated: DatedTile) -> bytes:
        """Fetch and decrypt the JPEG bytes for a dated tile."""
        return self._download(dated.asset_url)

    def provider_copyright(self, provider_id: int) -> str | None:
        for info in self.proto.provider_info:
            if info.provider_id == provider_id:
                value = info.copyright_string.value
                return value or None
        return None
