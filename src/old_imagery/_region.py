"""Bridge between shapely geometries (EPSG:4326) and provider tile grids.

Google Earth addresses tiles on a *square* plate-carree grid spanning 360
degrees in both axes; Esri Wayback uses standard Web Mercator XYZ tiles.  Both
are expressed here as a :class:`TileGrid` so the download mosaicking code is
provider-agnostic.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Protocol, TypeVar

from affine import Affine
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep

from ._keyhole import EQUATOR_DEGREES, KeyholeTile, validate_level

# Shared request/memory guard for the public API.  A download allocates four
# uint8 values per output pixel (RGB + mask), so 10,000 full 256 px tiles are
# already about 2.4 GiB before the in-memory GeoTIFF and decoder overhead.
MAX_TILES = 10_000
TILE_PX = 256
MERCATOR_EQUATOR = 40075016.68557849
MERCATOR_MAX_LAT = 85.051128779806589

_G = TypeVar("_G", bound=BaseGeometry)


def normalize_aoi(geometry: _G) -> _G:
    """Validate an area of interest supplied by the caller.

    The AOI is interpreted as lon/lat degrees in EPSG:4326.

    Generic in the geometry type so the caller's narrower annotation survives
    the call rather than widening back to :class:`BaseGeometry`.
    """
    if geometry is None or geometry.is_empty:
        raise ValueError("The area of interest is empty")

    # Checked as area rather than by type, so a GeometryCollection is judged on
    # what it actually contains. The public signatures say Polygon |
    # MultiPolygon | GeometryCollection, but annotations bind nothing at
    # runtime and most callers run unchecked -- and a zero-area AOI does not
    # merely degrade, it makes `coverage` undefined and `complete` false even
    # where imagery covers the point.
    if geometry.area <= 0.0:
        raise ValueError(
            f"The area of interest has no area ({geometry.geom_type}). Coverage "
            f"is measured as a fraction of that area, so it would be undefined. "
            f"Give it extent first, for example aoi.buffer(0.0001)."
        )

    minx, miny, maxx, maxy = geometry.bounds
    if not (minx >= -180.0 and maxx <= 180.0):
        raise ValueError(
            "Longitudes must lie within [-180, 180]. Split areas of interest "
            "that cross the antimeridian and query each part separately."
        )
    if not (miny >= -90.0 and maxy <= 90.0):
        raise ValueError("Latitudes must lie within [-90, 90]")
    return geometry


class TileGrid(Protocol):
    """A tiling scheme: tile selection, pixel space, and output georeferencing."""

    crs: str
    tile_scheme: str

    def tile_at_point(self, longitude: float, latitude: float, level: int): ...

    def tiles(self, aoi: BaseGeometry, level: int, max_tiles: int) -> list: ...

    def pixel_bounds(self, aoi: BaseGeometry, level: int) -> tuple[int, int, int, int]: ...

    def tile_origin(self, tile) -> tuple[int, int]: ...

    def transform(self, left: int, top: int, level: int) -> Affine: ...


def _clamp_index(value: float, n: int) -> int:
    return min(max(int(value), 0), n - 1)


def _select(tile_factory, rows: range, cols: range, aoi: BaseGeometry, level: int, max_tiles: int):
    count = len(rows) * len(cols)
    if count > max_tiles:
        raise ValueError(
            f"The area of interest spans {count:,} tiles at zoom {level}, above the limit "
            f"of {max_tiles:,}. Use a lower zoom or a smaller area, or raise max_tiles."
        )
    prepared = prep(aoi)
    out = []
    for row in rows:
        for col in cols:
            tile = tile_factory(row, col, level)
            if prepared.intersects(box(*tile.bounds_wgs84)):
                out.append(tile)
    return out


# --------------------------------------------------------------------------
# Google Earth / Keyhole
# --------------------------------------------------------------------------
class KeyholeGrid:
    """Square plate-carree grid: 360 degrees across *and* down."""

    crs = "EPSG:4326"
    tile_scheme = "GoogleEarthKeyhole"

    def tile_at_point(self, longitude: float, latitude: float, level: int) -> KeyholeTile:
        """Return the one native tile containing a WGS84 point."""
        n = validate_level(level)

        def index(degrees: float) -> int:
            return _clamp_index((degrees + 180.0) / 360.0 * n, n)

        return KeyholeTile.from_row_col(index(latitude), index(longitude), level)

    def tiles(self, aoi: BaseGeometry, level: int, max_tiles: int = MAX_TILES) -> list[KeyholeTile]:
        n = validate_level(level)
        minx, miny, maxx, maxy = aoi.bounds

        def index(deg: float) -> int:
            return _clamp_index((deg + 180.0) / 360.0 * n, n)

        rows = range(index(miny), index(maxy) + 1)
        cols = range(index(minx), index(maxx) + 1)
        return _select(KeyholeTile.from_row_col, rows, cols, aoi, level, max_tiles)

    def pixel_bounds(self, aoi: BaseGeometry, level: int) -> tuple[int, int, int, int]:
        size = TILE_PX * (1 << level)
        minx, miny, maxx, maxy = aoi.bounds
        left = round((minx + 180.0) / 360.0 * size)
        right = round((maxx + 180.0) / 360.0 * size)
        top = round((180.0 - maxy) / 360.0 * size)
        bottom = round((180.0 - miny) / 360.0 * size)
        return left, top, max(right, left + 1), max(bottom, top + 1)

    def tile_origin(self, tile: KeyholeTile) -> tuple[int, int]:
        n = 1 << tile.level
        row, col = tile.row_col
        return col * TILE_PX, (n - 1 - row) * TILE_PX

    def transform(self, left: int, top: int, level: int) -> Affine:
        size = TILE_PX * (1 << level)
        dpp = EQUATOR_DEGREES / size
        return Affine(dpp, 0.0, left * dpp - 180.0, 0.0, -dpp, 180.0 - top * dpp)


# --------------------------------------------------------------------------
# Web Mercator (Esri Wayback)
# --------------------------------------------------------------------------
class MercatorTile:
    """A standard XYZ tile; rows increase southward."""

    __slots__ = ("row", "column", "level")

    def __init__(self, row: int, column: int, level: int):
        self.row = row
        self.column = column
        self.level = level

    @property
    def bounds_wgs84(self) -> tuple[float, float, float, float]:
        n = 1 << self.level
        west = self.column / n * 360.0 - 180.0
        east = (self.column + 1) / n * 360.0 - 180.0
        north = _mercator_to_lat(self.row / n)
        south = _mercator_to_lat((self.row + 1) / n)
        return west, south, east, north

    @property
    def center(self) -> tuple[float, float]:
        west, south, east, north = self.bounds_wgs84
        return (west + east) / 2.0, (south + north) / 2.0

    def __hash__(self) -> int:
        return hash((self.row, self.column, self.level))

    def __eq__(self, other) -> bool:
        return isinstance(other, MercatorTile) and (self.row, self.column, self.level) == (
            other.row,
            other.column,
            other.level,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MercatorTile(z={self.level}, x={self.column}, y={self.row})"


def _mercator_to_lat(y_fraction: float) -> float:
    return math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_fraction))))


def _lat_to_mercator(lat: float) -> float:
    lat = max(min(lat, MERCATOR_MAX_LAT), -MERCATOR_MAX_LAT)
    rad = math.radians(lat)
    return (1.0 - math.asinh(math.tan(rad)) / math.pi) / 2.0


class MercatorGrid:
    crs = "EPSG:3857"
    tile_scheme = "WebMercatorQuad"
    max_level = 23

    def tile_at_point(self, longitude: float, latitude: float, level: int) -> MercatorTile:
        """Return the one XYZ tile containing a WGS84 point."""
        if not (0 <= level <= self.max_level):
            raise ValueError(f"zoom must be in [0, {self.max_level}] for Esri Wayback")
        n = 1 << level
        column = _clamp_index((longitude + 180.0) / 360.0 * n, n)
        row = _clamp_index(_lat_to_mercator(latitude) * n, n)
        return MercatorTile(row, column, level)

    def tiles(
        self, aoi: BaseGeometry, level: int, max_tiles: int = MAX_TILES
    ) -> list[MercatorTile]:
        if not (0 <= level <= self.max_level):
            raise ValueError(f"zoom must be in [0, {self.max_level}] for Esri Wayback")
        n = 1 << level
        minx, miny, maxx, maxy = aoi.bounds
        cols = range(
            _clamp_index((minx + 180.0) / 360.0 * n, n),
            _clamp_index((maxx + 180.0) / 360.0 * n, n) + 1,
        )
        rows = range(
            _clamp_index(_lat_to_mercator(maxy) * n, n),
            _clamp_index(_lat_to_mercator(miny) * n, n) + 1,
        )
        return _select(MercatorTile, rows, cols, aoi, level, max_tiles)

    def pixel_bounds(self, aoi: BaseGeometry, level: int) -> tuple[int, int, int, int]:
        size = TILE_PX * (1 << level)
        minx, miny, maxx, maxy = aoi.bounds
        left = round((minx + 180.0) / 360.0 * size)
        right = round((maxx + 180.0) / 360.0 * size)
        top = round(_lat_to_mercator(maxy) * size)
        bottom = round(_lat_to_mercator(miny) * size)
        return left, top, max(right, left + 1), max(bottom, top + 1)

    def tile_origin(self, tile: MercatorTile) -> tuple[int, int]:
        return tile.column * TILE_PX, tile.row * TILE_PX

    def transform(self, left: int, top: int, level: int) -> Affine:
        size = TILE_PX * (1 << level)
        mpp = MERCATOR_EQUATOR / size
        return Affine(
            mpp,
            0.0,
            left * mpp - MERCATOR_EQUATOR / 2.0,
            0.0,
            -mpp,
            MERCATOR_EQUATOR / 2.0 - top * mpp,
        )


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
def dissolve(tiles: Iterable) -> BaseGeometry:
    """Dissolve tile extents (in EPSG:4326) into a single geometry."""
    return unary_union([box(*t.bounds_wgs84) for t in tiles])


def sort_by_nearest_date(dates: Sequence, target, match: str):
    """Order dated items by distance from ``target`` under a match rule.

    ``match`` is one of ``closest``, ``exact``, ``before`` or ``after``.  Items
    with an unknown (``None``) date sort last; they are the provider's undated
    default imagery and are only used as a fallback.

    Validation happens here rather than in the generator below: a `raise` in a
    generator body does not run until the first `next()`, which would defer a
    bad ``match`` to whichever caller happened to iterate first -- or swallow it
    entirely for a caller that builds the iterator and never consumes it.
    """
    if match not in ("closest", "exact", "before", "after"):
        raise ValueError(f"Unknown date_match {match!r}")
    return _sorted_by_nearest_date(dates, target, match)


def _sorted_by_nearest_date(dates: Sequence, target, match: str):
    known = [d for d in dates if d.date is not None]
    unknown = [d for d in dates if d.date is None]

    if match == "exact":
        known = [d for d in known if d.date == target]
    elif match == "before":
        known = [d for d in known if d.date <= target]
    elif match == "after":
        known = [d for d in known if d.date >= target]

    known.sort(key=lambda d: (abs((d.date - target).days), d.date))
    yield from known
    if match != "exact":
        yield from unknown
