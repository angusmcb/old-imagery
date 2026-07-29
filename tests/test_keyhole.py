"""Offline tests for the Keyhole addressing, crypto and packet code.

The subindex expectations come from upstream's own pre-computed fixtures
(``test/LibGoogleEarthTest/*IndexDictionary.json``), so a passing run means
this port agrees with the C# implementation node-for-node.
"""

from __future__ import annotations

import datetime as dt
import json
import struct
import zlib
from pathlib import Path

import pytest

from old_imagery._keyhole import (
    KeyholeTile,
    decode_date,
    decompress,
    decrypt,
    degrees_per_pixel,
    parse_binary_packet,
    root_subindex,
    tree_subindex,
)

DATA = Path(__file__).parent / "data"
SUB_INDEX = json.loads((DATA / "SubIndexDictionary.json").read_text())
ROOT_INDEX = json.loads((DATA / "RootIndexDictionary.json").read_text())


# --------------------------------------------------------------------------
# quadtree paths
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prefix",
    ["0000", "00000000", "000000000000", "0000000000000000", "00000000000000000000"],
)
def test_subindex_matches_upstream_fixture(prefix: str) -> None:
    """Mirrors upstream QtPathTest.SubIndices."""
    for suffix, expected in SUB_INDEX.items():
        tile = KeyholeTile(prefix + suffix)
        assert tile.path == prefix + suffix
        if len(suffix) <= 4:
            assert tile.subindex == expected


def test_root_subindex_matches_upstream_fixture() -> None:
    """Mirrors upstream QtPathTest.RootIndex."""
    for path, expected in ROOT_INDEX.items():
        assert KeyholeTile(path).subindex == expected


def test_root_tile() -> None:
    root = KeyholeTile("0")
    assert root.is_root
    assert root.subindex == 0
    assert root.level == 0


def test_subindex_helpers() -> None:
    assert root_subindex("0") == 0
    assert root_subindex("00") == 1
    assert tree_subindex("0") == 1
    assert tree_subindex("3") == 3 * 85 + 1


@pytest.mark.parametrize(
    "path", ["1", "01234", "012334", "0000134", "00001304", "10001304", " 02322", ""]
)
def test_bad_paths_rejected(path: str) -> None:
    """Mirrors upstream QtPathTest.BadPaths."""
    with pytest.raises(ValueError):
        KeyholeTile(path)


def test_index_paths_step_every_four_levels() -> None:
    tile = KeyholeTile("0123213213012")
    assert [t.path for t in tile.index_paths()] == ["0123", "01232132", "012321321301"]


# --------------------------------------------------------------------------
# row / column <-> path
# --------------------------------------------------------------------------
MAX_ROW_COL = (1 << 30) - 1


@pytest.mark.parametrize(
    "row,col,level,path",
    [
        (0, 0, 0, "0"),
        (0, 0, 1, "00"),
        (0, 1, 1, "01"),
        (1, 1, 1, "02"),
        (1, 0, 1, "03"),
        ((1 << 10) - 1, 0, 10, "03333333333"),
        (0, (1 << 10) - 1, 10, "01111111111"),
        ((1 << 10) - 1, (1 << 10) - 1, 10, "02222222222"),
        (MAX_ROW_COL, 0, 30, "0" + "3" * 30),
        (0, MAX_ROW_COL, 30, "0" + "1" * 30),
        (MAX_ROW_COL, MAX_ROW_COL, 30, "0" + "2" * 30),
        (0b0111011011, 0b1101101101, 10, "01232132132"),
    ],
)
def test_row_col_roundtrip(row: int, col: int, level: int, path: str) -> None:
    """Mirrors upstream KeyholeTileTests.ValidTiles."""
    tile = KeyholeTile.from_row_col(row, col, level)
    assert tile.path == path
    assert tile.level == level
    assert tile.row_col == (row, col)


@pytest.mark.parametrize(
    "row,col,level", [(0, 0, -1), (0, -1, 0), (-1, 0, 0), (1 << 10, 0, 10), (0, 1 << 10, 10), (0, 0, 31)]
)
def test_row_col_out_of_range(row: int, col: int, level: int) -> None:
    """Mirrors upstream KeyholeTileTests.TilesOutOfRange."""
    with pytest.raises(ValueError):
        KeyholeTile.from_row_col(row, col, level)


@pytest.mark.parametrize(
    "lat,lon,level,row,col",
    [
        (0, 0, 0, 0, 0),
        (0, 0, 1, 1, 1),
        (0.00000001, 0, 1, 1, 1),
        (-0.00000001, 0, 1, 0, 1),
        (90, 0, 1, 1, 1),
        (0, 0.00000001, 1, 1, 1),
        (0, -0.00000001, 1, 1, 0),
    ],
)
def test_tile_from_lat_lon(lat: float, lon: float, level: int, row: int, col: int) -> None:
    """Mirrors upstream CoordinateTests.GetTile."""
    tile = KeyholeTile.from_lat_lon(lat, lon, level)
    assert (tile.row, tile.column) == (row, col)
    assert tile.level == level


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def test_bounds_are_square_in_degrees() -> None:
    tile = KeyholeTile.from_lat_lon(37.7955, -122.3937, 18)
    west, south, east, north = tile.bounds
    assert east - west == pytest.approx(north - south)
    assert east - west == pytest.approx(360.0 / (1 << 18))
    assert west < -122.3937 < east
    assert south < 37.7955 < north


def test_root_covers_the_square_globe() -> None:
    assert KeyholeTile("0").bounds == (-180.0, -180.0, 180.0, 180.0)


def test_degrees_per_pixel() -> None:
    assert degrees_per_pixel(0) == 360.0 / 256
    assert degrees_per_pixel(18) == pytest.approx(360.0 / (256 * 2**18))


# --------------------------------------------------------------------------
# crypto / compression
# --------------------------------------------------------------------------
def test_decrypt_is_an_involution() -> None:
    key = bytes(range(256)) * 2
    plain = b"the quick brown fox jumps over the lazy dog" * 13
    assert bytes(decrypt(decrypt(plain, key), key)) == plain


def test_decrypt_key_walk_matches_reference() -> None:
    """Independent re-implementation of the upstream offset walk."""
    key = bytes((i * 7 + 3) & 0xFF for i in range(1024))
    data = bytes(range(200))

    expected = bytearray(data)
    off = 16
    for j in range(len(expected)):
        expected[j] ^= key[off]
        off += 1
        if (off & 7) == 0:
            off += 16
        if off >= len(key):
            off = (off + 8) % 24

    assert decrypt(data, key) == expected


def test_decrypt_rejects_empty_key() -> None:
    with pytest.raises(ValueError):
        decrypt(b"abc", b"")


def _compressed(payload: bytes, magic: int = 0x7468DEAD) -> bytes:
    return struct.pack("<II", magic, len(payload)) + zlib.compress(payload)


def test_decompress_roundtrip() -> None:
    payload = b"keyhole packet payload" * 40
    assert decompress(_compressed(payload)) == payload


def test_decompress_rejects_bad_magic() -> None:
    with pytest.raises(ValueError, match="bad magic"):
        decompress(_compressed(b"x", magic=0xDEADBEEF))


def test_decompress_rejects_short_buffer() -> None:
    with pytest.raises(ValueError, match="too small"):
        decompress(b"\x00\x01")


def test_decompress_detects_size_mismatch() -> None:
    bad = struct.pack("<II", 0x7468DEAD, 9999) + zlib.compress(b"short")
    with pytest.raises(ValueError, match="!="):
        decompress(bad)


# --------------------------------------------------------------------------
# binary quadtree packets
# --------------------------------------------------------------------------
def _quantum(children: int, num_channels: int = 0, image_version: int = 7) -> bytes:
    return struct.pack(
        "<HhhhhHiiqBBH", children, 11, image_version, 0, num_channels, 0, 0, 0, 0, 3, 0, 0
    )


def _packet(quanta: list[bytes], version: int = 42) -> bytes:
    header = struct.pack(
        "<IIiiiiii", 32301, 1, version, len(quanta), 32, 32 + 32 * len(quanta), 0, 0
    )
    return header + b"".join(quanta)


def test_parse_binary_packet_single_node() -> None:
    packet = _packet([_quantum(children=1 << 6)])  # image bit only
    parsed = parse_binary_packet(packet, is_root=True)
    assert parsed.packet_epoch == 42
    node = parsed.node_at(0)
    assert node is not None and node.has_image
    assert node.cache_node_epoch == 11
    assert [(l.type, l.layer_epoch, l.provider) for l in node.layers] == [(0, 7, 3)]


def test_parse_binary_packet_walks_children() -> None:
    # Root has child 0 and child 2 set, plus the image bit.
    root = _quantum(children=(1 << 6) | 0b0101)
    packet = _packet([root, _quantum(1 << 6), _quantum(1 << 6)])
    parsed = parse_binary_packet(packet, is_root=True)
    # Root plus its two children occupy three distinct subindices.
    assert len(parsed.nodes) == 3
    assert parsed.node_at(0) is not None


def test_parse_binary_packet_rejects_bad_magic() -> None:
    bad = struct.pack("<IIiiiiii", 1234, 1, 1, 0, 32, 32, 0, 0)
    with pytest.raises(ValueError, match="magic_id"):
        parse_binary_packet(bad, is_root=True)


def test_parse_binary_packet_rejects_bad_offset() -> None:
    bad = struct.pack("<IIiiiiii", 32301, 1, 1, 2, 32, 999, 0, 0)
    with pytest.raises(ValueError, match="data_buffer_offset"):
        parse_binary_packet(bad, is_root=True)


# --------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "year,month,day", [(1938, 8, 1), (1993, 7, 10), (2023, 4, 30), (2000, 12, 31)]
)
def test_decode_date_roundtrip(year: int, month: int, day: int) -> None:
    packed = ((year & 0x7FF) << 9) | ((month & 0xF) << 5) | (day & 0x1F)
    assert decode_date(packed) == dt.date(year, month, day)


def test_decode_date_rejects_impossible_values() -> None:
    assert decode_date(0) is None  # year 0, month 0, day 0
