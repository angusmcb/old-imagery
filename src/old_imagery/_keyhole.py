"""Keyhole quadtree addressing, packet decryption and binary packet parsing.

Ported from ``LibGoogleEarth`` in Mbucari/GEHistoricalImagery.
"""

from __future__ import annotations

import datetime as _dt
import struct
import zlib
from dataclasses import dataclass
from typing import Iterator

MAX_LEVEL = 30
TILE_SIZE = 256
_SUBINDEX_MAX_SZ = 4

# The Keyhole grid is square in degrees: the globe spans 360 degrees in *both*
# axes, so only the middle half of the Y range covers real latitudes.
EQUATOR_DEGREES = 360.0


# --------------------------------------------------------------------------
# quadtree path <-> subindex
# --------------------------------------------------------------------------
def validate_path(path: str) -> None:
    if not path:
        raise ValueError("Quadtree path must not be empty")
    if any(c not in "0123" for c in path):
        raise ValueError("Quadtree path may only contain the characters '0'-'3'")


def validate_rooted_path(path: str) -> None:
    validate_path(path)
    if path[0] != "0":
        raise ValueError("All quadtree paths must begin with a '0'")


def root_subindex(path: str) -> int:
    """Subindex numbering used by the root packet (plain left-to-right)."""
    validate_path(path)
    if len(path) > _SUBINDEX_MAX_SZ:
        raise ValueError(f"Path {path!r} is longer than {_SUBINDEX_MAX_SZ}")
    subindex = 0
    for c in path[1:]:
        subindex = subindex * _SUBINDEX_MAX_SZ + (ord(c) - 0x30) + 1
    return subindex


def tree_subindex(path: str) -> int:
    """Subindex numbering used by non-root packets (mangled second row)."""
    return root_subindex(path) + (ord(path[0]) - 0x30) * 85 + 1


def _subindex(path: str) -> int:
    if len(path) <= _SUBINDEX_MAX_SZ:
        return root_subindex(path)
    # The subindex is relative to the packet that contains the node, which
    # starts every 4 levels.
    return tree_subindex(path[(len(path) - 1) // _SUBINDEX_MAX_SZ * _SUBINDEX_MAX_SZ :])


# --------------------------------------------------------------------------
# tiles
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class KeyholeTile:
    """A square Keyhole tile, addressed by its rooted quadtree path.

    ``row`` counts north from the bottom (south) edge of the square globe,
    ``column`` counts east from the left (west) edge.
    """

    path: str

    def __post_init__(self) -> None:
        validate_rooted_path(self.path)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_row_col(cls, row: int, col: int, level: int) -> "KeyholeTile":
        num_tiles = validate_level(level)
        if not (0 <= row < num_tiles) or not (0 <= col < num_tiles):
            raise ValueError(f"row/col out of range for level {level}")
        chars = [""] * (level + 1)
        for i in range(level, -1, -1):
            r = row & 1
            c = col & 1
            row >>= 1
            col >>= 1
            chars[i] = chr((r << 1 | (r ^ c)) | 0x30)
        return cls("".join(chars))

    @classmethod
    def from_lat_lon(cls, lat: float, lon: float, level: int) -> "KeyholeTile":
        return cls.from_row_col(_deg_to_row_col(lat, level), _deg_to_row_col(lon, level), level)

    # -- addressing --------------------------------------------------------
    @property
    def level(self) -> int:
        return len(self.path) - 1

    @property
    def subindex(self) -> int:
        return _subindex(self.path)

    @property
    def is_root(self) -> bool:
        return self.path == "0"

    @property
    def row_col(self) -> tuple[int, int]:
        row = col = 0
        for c in self.path:
            cell = ord(c) & 3
            r = cell >> 1
            cl = r ^ (cell & 1)
            row = (row << 1) | r
            col = (col << 1) | cl
        return row, col

    @property
    def row(self) -> int:
        return self.row_col[0]

    @property
    def column(self) -> int:
        return self.row_col[1]

    def parent(self) -> "KeyholeTile":
        if self.level == 0:
            raise ValueError("The root tile has no parent")
        return KeyholeTile(self.path[:-1])

    def index_paths(self) -> Iterator["KeyholeTile"]:
        """The intermediate packet paths between the root and this tile."""
        for end in range(_SUBINDEX_MAX_SZ, len(self.path), _SUBINDEX_MAX_SZ):
            yield KeyholeTile(self.path[:end])

    # -- geometry ----------------------------------------------------------
    def _to_deg(self, row_col: float) -> float:
        return row_col * EQUATOR_DEGREES / (1 << self.level) - 180.0

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(west, south, east, north)`` in degrees."""
        row, col = self.row_col
        return (
            self._to_deg(col),
            self._to_deg(row),
            self._to_deg(col + 1),
            self._to_deg(row + 1),
        )

    @property
    def bounds_wgs84(self) -> tuple[float, float, float, float]:
        """Alias of :attr:`bounds`; the Keyhole grid is already lon/lat."""
        return self.bounds

    @property
    def center(self) -> tuple[float, float]:
        """``(lon, lat)`` of the tile centre, in degrees."""
        row, col = self.row_col
        return self._to_deg(col + 0.5), self._to_deg(row + 0.5)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.path


ROOT = KeyholeTile("0")


def validate_level(level: int) -> int:
    if not (0 <= level <= MAX_LEVEL):
        raise ValueError(f"level must be in [0, {MAX_LEVEL}], got {level}")
    return 1 << level


def _deg_to_row_col(degrees: float, level: int) -> int:
    num_tiles = validate_level(level)
    return min(int((degrees + 180.0) / 360.0 * num_tiles), num_tiles - 1)


def degrees_per_pixel(level: int) -> float:
    return EQUATOR_DEGREES / (TILE_SIZE * (1 << level))


# --------------------------------------------------------------------------
# obfuscation / compression
# --------------------------------------------------------------------------
def decrypt(cipher: bytes | bytearray, key: bytes) -> bytearray:
    """Undo Keyhole's XOR obfuscation (``ENCRYPTION_XOR``).

    The key is walked with an irregular stride: after every 8th byte it jumps
    forward 16, and it wraps to ``(off + 8) % 24`` rather than to 0.
    """
    out = bytearray(cipher)
    key_len = len(key)
    if key_len == 0:
        raise ValueError("Empty encryption key")
    off = 16
    for j in range(len(out)):
        out[j] ^= key[off]
        off += 1
        if (off & 7) == 0:
            off += 16
        if off >= key_len:
            off = (off + 8) % 24
    return out


_PKT_MAGIC = 0x7468DEAD
_PKT_MAGIC_SWAP = 0xADDE6874
_COMPRESS_HDR_SIZE = 8


def decompress(buffer: bytes | bytearray) -> bytes:
    """Strip the 8-byte Keyhole compression header and inflate the payload."""
    if len(buffer) < _COMPRESS_HDR_SIZE:
        raise ValueError("Buffer too small to contain a compression header")
    magic, size = struct.unpack_from("<II", buffer, 0)
    if magic == _PKT_MAGIC:
        expected = size
    elif magic == _PKT_MAGIC_SWAP:
        expected = struct.unpack_from(">I", buffer, 4)[0]
    else:
        raise ValueError(f"Failed to determine packet size (bad magic 0x{magic:08x})")

    data = zlib.decompress(bytes(buffer[_COMPRESS_HDR_SIZE:]))
    if len(data) != expected:
        raise ValueError(f"Decompressed size {len(data)} != declared {expected}")
    return data


# --------------------------------------------------------------------------
# binary (non-protobuf) quadtree packets, used by the default database
# --------------------------------------------------------------------------
_KEYHOLE_MAGIC_ID = 32301
_HEADER = struct.Struct("<IIiiiiii")  # 32 bytes
# children:u16, cnode/image/terrain version:i16 x3, num_channels:i16, pad:u16,
# type_offset:i32, version_offset:i32, image_neighbors:i64,
# image_provider:u8, terrain_provider:u8, pad:u16   => 32 bytes
_QUANTUM = struct.Struct("<HhhhhHiiqBBH")

_BIT_CACHE = 4
_BIT_DRAWABLE = 5
_BIT_IMAGE = 6
_BIT_TERRAIN = 7


@dataclass(frozen=True)
class BinaryLayer:
    type: int
    layer_epoch: int
    provider: int


@dataclass(frozen=True)
class BinaryNode:
    """Mimics the subset of ``keyhole.QuadtreeNode`` that callers rely on."""

    children: int
    cache_node_epoch: int
    layers: tuple[BinaryLayer, ...]

    def has_bit(self, bit: int) -> bool:
        return bool(self.children & (1 << bit))

    @property
    def has_image(self) -> bool:
        return self.has_bit(_BIT_IMAGE)


@dataclass(frozen=True)
class BinaryPacket:
    packet_epoch: int
    nodes: dict[int, BinaryNode]

    def node_at(self, subindex: int) -> BinaryNode | None:
        return self.nodes.get(subindex)


def parse_binary_packet(data: bytes, is_root: bool) -> BinaryPacket:
    """Parse a ``KhQuadTreePacket16`` buffer (decrypted and decompressed)."""
    if len(data) < _HEADER.size:
        raise ValueError("Buffer too small for a quadtree packet header")
    (
        magic_id,
        _data_type_id,
        version,
        num_instances,
        instance_size,
        data_buffer_offset,
        _data_buffer_size,
        _meta_buffer_size,
    ) = _HEADER.unpack_from(data, 0)

    if magic_id != _KEYHOLE_MAGIC_ID:
        raise ValueError(f"Invalid quadtree packet magic_id: {magic_id}")
    if num_instances and data_buffer_offset != _HEADER.size + num_instances * instance_size:
        raise ValueError("Invalid data_buffer_offset")

    quanta = [
        _QUANTUM.unpack_from(data, _HEADER.size + i * _QUANTUM.size) for i in range(num_instances)
    ]
    channels = memoryview(data)[data_buffer_offset:].cast("h")

    nodes: dict[int, BinaryNode] = {}
    if num_instances:
        _traverse(quanta, channels, nodes, 0, "", is_root)
    return BinaryPacket(version, nodes)


def _traverse(quanta, channels, nodes, node_index: int, qt_path: str, is_root: bool) -> int:
    if node_index >= len(quanta):
        return node_index

    q = quanta[node_index]
    (
        children,
        cnode_version,
        image_version,
        terrain_version,
        num_channels,
        _pad,
        type_offset,
        version_offset,
        _neighbors,
        image_provider,
        terrain_provider,
        _pad2,
    ) = q

    if is_root:
        subindex = root_subindex("0" + qt_path)
    elif node_index > 0:
        subindex = tree_subindex(qt_path)
    else:
        subindex = 0

    channel_types = channels[type_offset // 2 : type_offset // 2 + num_channels]
    channel_versions = channels[version_offset // 2 : version_offset // 2 + num_channels]
    del channel_types, channel_versions  # parsed for parity; unused by this port

    layers: list[BinaryLayer] = []
    if children & (1 << _BIT_IMAGE):
        layers.append(BinaryLayer(0, image_version, image_provider))  # imagery
    if children & (1 << _BIT_TERRAIN):
        layers.append(BinaryLayer(1, terrain_version, terrain_provider))  # terrain
    if children & (1 << _BIT_DRAWABLE):
        layers.append(BinaryLayer(2, 0, 0))  # vector

    nodes[subindex] = BinaryNode(children, cnode_version, tuple(layers))

    for i in range(4):
        if children & (1 << i):
            node_index = _traverse(quanta, channels, nodes, node_index + 1, qt_path + str(i), is_root)
    return node_index


# --------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------
MIN_JPEG_DATE = 545


def decode_date(packed: int) -> _dt.date | None:
    """Decode a Keyhole packed date. Returns ``None`` when unrepresentable."""
    year = packed >> 9
    month = (packed >> 5) & 0xF
    day = packed & 0x1F
    try:
        return _dt.date(year, month, day)
    except ValueError:
        return None
