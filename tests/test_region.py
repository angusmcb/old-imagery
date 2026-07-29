"""Offline tests for tile-grid selection, pixel space and date matching."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pytest
from shapely.geometry import Point, box

from old_imagery._keyhole import KeyholeTile
from old_imagery._region import (
    KeyholeGrid,
    MercatorGrid,
    MercatorTile,
    dissolve,
    normalize_aoi,
    sort_by_nearest_date,
)

AOI = box(-122.4020, 37.7900, -122.3880, 37.7990)


# --------------------------------------------------------------------------
# AOI validation
# --------------------------------------------------------------------------
def test_normalize_accepts_a_plain_box() -> None:
    assert normalize_aoi(AOI) is AOI


def test_normalize_rejects_empty() -> None:
    empty = box(0, 0, 1, 1).intersection(box(5, 5, 6, 6))
    assert empty.is_empty
    with pytest.raises(ValueError, match="empty"):
        normalize_aoi(empty)


def test_normalize_rejects_none() -> None:
    with pytest.raises(ValueError, match="empty"):
        normalize_aoi(None)


def test_degenerate_box_is_treated_as_a_point() -> None:
    """A zero-area box is point-like, not empty, and selects one tile."""
    degenerate = box(-122.3937, 37.7955, -122.3937, 37.7955)
    assert normalize_aoi(degenerate) is degenerate
    assert len(KeyholeGrid().tiles(degenerate, 18, 100)) == 1


def test_normalize_rejects_antimeridian_crossing() -> None:
    with pytest.raises(ValueError, match="antimeridian"):
        normalize_aoi(box(-181, 10, -179, 12))


def test_normalize_rejects_out_of_range_latitude() -> None:
    with pytest.raises(ValueError, match="Latitudes"):
        normalize_aoi(box(0, -91, 1, -89))


# --------------------------------------------------------------------------
# tile selection
# --------------------------------------------------------------------------
@pytest.mark.parametrize("grid", [KeyholeGrid(), MercatorGrid()])
def test_selected_tiles_all_intersect_the_aoi(grid) -> None:
    tiles = grid.tiles(AOI, 17, 10_000)
    assert tiles
    for tile in tiles:
        assert box(*tile.bounds_wgs84).intersects(AOI)


@pytest.mark.parametrize("grid", [KeyholeGrid(), MercatorGrid()])
def test_selected_tiles_cover_the_aoi(grid) -> None:
    tiles = grid.tiles(AOI, 17, 10_000)
    assert dissolve(tiles).contains(AOI)


@pytest.mark.parametrize("grid", [KeyholeGrid(), MercatorGrid()])
def test_tile_count_grows_fourfold_per_level(grid) -> None:
    small = box(-122.40, 37.79, -122.39, 37.80)
    counts = [len(grid.tiles(small, z, 100_000)) for z in (14, 15, 16)]
    assert counts[0] < counts[1] < counts[2]


@pytest.mark.parametrize("grid", [KeyholeGrid(), MercatorGrid()])
def test_max_tiles_is_enforced(grid) -> None:
    with pytest.raises(ValueError, match="above the limit"):
        grid.tiles(box(-30, -30, 30, 30), 14, max_tiles=100)


def test_a_point_selects_a_single_tile() -> None:
    tiles = KeyholeGrid().tiles(Point(-122.3937, 37.7955), 18, 100)
    assert len(tiles) == 1
    west, south, east, north = tiles[0].bounds_wgs84
    assert west <= -122.3937 <= east and south <= 37.7955 <= north


# --------------------------------------------------------------------------
# pixel space
# --------------------------------------------------------------------------
def test_keyhole_tile_origin_is_on_a_tile_boundary() -> None:
    tile = KeyholeTile.from_lat_lon(37.7955, -122.3937, 18)
    ox, oy = KeyholeGrid().tile_origin(tile)
    assert ox % 256 == 0 and oy % 256 == 0


@pytest.mark.parametrize("grid", [KeyholeGrid(), MercatorGrid()])
def test_transform_of_a_tile_origin_matches_its_bounds(grid) -> None:
    """The affine transform and the tile's own bounds must agree."""
    zoom = 17
    tile = grid.tiles(AOI, zoom, 10_000)[0]
    ox, oy = grid.tile_origin(tile)
    transform = grid.transform(ox, oy, zoom)
    # Upper-left of the transform is the tile's upper-left corner.
    x, y = transform * (0, 0)
    if grid.crs == "EPSG:4326":
        west, _south, _east, north = tile.bounds_wgs84
        assert x == pytest.approx(west, abs=1e-9)
        assert y == pytest.approx(north, abs=1e-9)
    else:
        from pyproj import Transformer

        lon, lat = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True).transform(x, y)
        west, _south, _east, north = tile.bounds_wgs84
        assert lon == pytest.approx(west, abs=1e-6)
        assert lat == pytest.approx(north, abs=1e-6)


@pytest.mark.parametrize("grid", [KeyholeGrid(), MercatorGrid()])
def test_pixel_bounds_enclose_every_selected_tile(grid) -> None:
    zoom = 16
    left, top, right, bottom = grid.pixel_bounds(AOI, zoom)
    assert right > left and bottom > top
    for tile in grid.tiles(AOI, zoom, 10_000):
        ox, oy = grid.tile_origin(tile)
        assert ox < right and ox + 256 > left
        assert oy < bottom and oy + 256 > top


def test_mercator_tile_bounds_are_standard_xyz() -> None:
    # z=1 quadrants: row 0 is the northern hemisphere.
    north_west = MercatorTile(0, 0, 1)
    assert north_west.bounds_wgs84[0] == pytest.approx(-180.0)
    assert north_west.bounds_wgs84[3] == pytest.approx(85.0511, abs=1e-4)
    assert north_west.bounds_wgs84[1] == pytest.approx(0.0, abs=1e-9)


def test_mercator_tile_equality_and_hash() -> None:
    assert MercatorTile(1, 2, 3) == MercatorTile(1, 2, 3)
    assert len({MercatorTile(1, 2, 3), MercatorTile(1, 2, 3)}) == 1
    assert MercatorTile(1, 2, 3) != MercatorTile(1, 2, 4)


# --------------------------------------------------------------------------
# date matching
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Dated:
    date: dt.date | None


D = lambda s: Dated(dt.date.fromisoformat(s)) if s else Dated(None)  # noqa: E731
CANDIDATES = [D("2001-01-01"), D("2010-06-01"), D("2015-01-01"), D(None)]
TARGET = dt.date(2011, 1, 1)


def test_closest_orders_by_absolute_distance() -> None:
    got = [d.date for d in sort_by_nearest_date(CANDIDATES, TARGET, "closest")]
    assert got == [dt.date(2010, 6, 1), dt.date(2015, 1, 1), dt.date(2001, 1, 1), None]


def test_before_keeps_only_earlier_dates() -> None:
    got = [d.date for d in sort_by_nearest_date(CANDIDATES, TARGET, "before")]
    assert got == [dt.date(2010, 6, 1), dt.date(2001, 1, 1), None]


def test_after_keeps_only_later_dates() -> None:
    got = [d.date for d in sort_by_nearest_date(CANDIDATES, TARGET, "after")]
    assert got == [dt.date(2015, 1, 1), None]


def test_exact_matches_only_that_day_and_excludes_undated() -> None:
    assert [d.date for d in sort_by_nearest_date(CANDIDATES, TARGET, "exact")] == []
    exact = sort_by_nearest_date(CANDIDATES, dt.date(2010, 6, 1), "exact")
    assert [d.date for d in exact] == [dt.date(2010, 6, 1)]


def test_unknown_match_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown date_match"):
        list(sort_by_nearest_date(CANDIDATES, TARGET, "nearest-ish"))
