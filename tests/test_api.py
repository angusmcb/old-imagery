"""Offline tests for the public API, using a stub backend instead of network."""

from __future__ import annotations

import datetime as dt
import inspect
import json
import sqlite3
import threading
import time
from dataclasses import dataclass

import numpy as np
import pytest
import rasterio
from shapely.geometry import LineString, MultiPoint, MultiPolygon, box

import old_imagery
from old_imagery import api
from old_imagery._esri import EsriFootprint, EsriSource
from old_imagery._keyhole import KeyholeTile
from old_imagery._region import KeyholeGrid, MercatorGrid

AOI = box(-122.4000, 37.7920, -122.3960, 37.7950)
ZOOM = 17


@dataclass(frozen=True)
class StubDated:
    tile: KeyholeTile
    date: dt.date | None
    provider: int
    source: EsriSource | None = None


class StubBackend:
    """A DbRoot-shaped backend serving synthetic tiles with no network."""

    grid = KeyholeGrid()

    def __init__(self, dates, colors=None, missing=(), source=None):
        self.dates = list(dates)
        self.colors = colors or {}
        self.missing = set(missing)
        self.source = source
        self.downloads = 0

    def dated_tiles(self, tile):
        if tile in self.missing:
            return []
        return [StubDated(tile, d, 7, self.source) for d in self.dates]

    def download_tile_image(self, dated):
        self.downloads += 1
        value = self.colors.get(dated.date, 128)
        arr = np.full((3, 256, 256), value, dtype=np.uint8)
        return _encode_jpeg(arr)

    def provider_copyright(self, provider_id):
        return "Test Provider" if provider_id == 7 else None


def _encode_jpeg(arr: np.ndarray) -> bytes:
    from rasterio.io import MemoryFile

    with MemoryFile(ext=".jpg") as mem:
        with mem.open(
            driver="JPEG", width=arr.shape[2], height=arr.shape[1], count=3, dtype="uint8"
        ) as dst:
            dst.write(arr)
        return mem.read()


@pytest.fixture
def stub(monkeypatch):
    """Install a stub backend; returns a setter so each test picks its data."""
    holder = {}

    def install(backend):
        holder["backend"] = backend

        class DummyClient:
            def close(self):
                holder["closed"] = True

        monkeypatch.setattr(api, "_backend", lambda provider, cache_dir: (backend, DummyClient()))
        return backend

    install.holder = holder  # type: ignore[attr-defined]
    return install


D1, D2 = dt.date(2005, 5, 5), dt.date(2015, 9, 9)


# --------------------------------------------------------------------------
# availability
# --------------------------------------------------------------------------
def test_availability_returns_a_geodataframe_sorted_newest_first(stub) -> None:
    stub(StubBackend([D1, D2]))
    gdf = old_imagery.availability(AOI, ZOOM)

    assert list(gdf.columns) == api.AVAILABILITY_COLUMNS
    assert gdf.crs == "EPSG:4326"
    assert list(gdf["date"]) == [D2, D1]
    assert (gdf["coverage"] == 1.0).all()
    assert gdf["complete"].all()
    assert (gdf["providers"] == "Test Provider").all()


def test_availability_geometry_is_clipped_to_the_aoi(stub) -> None:
    stub(StubBackend([D1]))
    gdf = old_imagery.availability(AOI, ZOOM)
    geom = gdf.geometry.iloc[0]
    assert geom.within(AOI.buffer(1e-9))
    assert geom.area == pytest.approx(AOI.area, rel=1e-6)


def test_availability_filters_by_date_range(stub) -> None:
    stub(StubBackend([D1, D2]))
    assert list(old_imagery.availability(AOI, ZOOM, min_date="2010-01-01")["date"]) == [D2]
    assert list(old_imagery.availability(AOI, ZOOM, max_date="2010-01-01")["date"]) == [D1]
    assert old_imagery.availability(AOI, ZOOM, min_date=D2, max_date=D2)["date"].tolist() == [D2]


def test_availability_excludes_undated_imagery(stub) -> None:
    stub(StubBackend([D1, None]))
    assert list(old_imagery.availability(AOI, ZOOM)["date"]) == [D1]


def test_availability_reports_partial_coverage_as_area(stub) -> None:
    """Coverage is the fraction of the AOI's *area*, not of its tile count."""
    backend = StubBackend([D1])
    tiles = KeyholeGrid().tiles(AOI, ZOOM, 10_000)
    assert len(tiles) > 1
    backend.missing = {tiles[0]}
    stub(backend)

    gdf = old_imagery.availability(AOI, ZOOM)
    covered = gdf.geometry.iloc[0]
    assert gdf["coverage"].iloc[0] == pytest.approx(covered.area / AOI.area, rel=1e-3)
    assert 0 < gdf["coverage"].iloc[0] < 1
    assert not bool(gdf["complete"].iloc[0])
    # A dropped tile that only clips the AOI corner must not cost a whole
    # 1/len(tiles) of coverage the way a tile count would.
    assert gdf["coverage"].iloc[0] != pytest.approx((len(tiles) - 1) / len(tiles))


def test_availability_empty_when_no_imagery(stub) -> None:
    tiles = KeyholeGrid().tiles(AOI, ZOOM, 10_000)
    stub(StubBackend([D1], missing=tiles))
    gdf = old_imagery.availability(AOI, ZOOM)
    assert len(gdf) == 0
    assert list(gdf.columns) == api.AVAILABILITY_COLUMNS
    assert gdf.crs == "EPSG:4326"
    assert gdf.attrs["n_aoi_tiles"] == len(tiles)
    assert gdf.attrs["method"] == "per-tile"


def test_availability_records_attrs(stub) -> None:
    stub(StubBackend([D1]))
    gdf = old_imagery.availability(AOI, ZOOM)
    assert gdf.attrs["zoom"] == ZOOM
    assert gdf.attrs["provider"] == "google"
    assert gdf.attrs["n_aoi_tiles"] > 0
    assert gdf.attrs["method"] == "per-tile"


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------
def test_download_returns_georeferenced_rgb(stub) -> None:
    stub(StubBackend([D1, D2], colors={D1: 40, D2: 200}))
    ds = old_imagery.download(AOI, ZOOM, D2)

    assert ds.count == 3
    assert ds.dtypes == ("uint8", "uint8", "uint8")
    assert ds.crs == "EPSG:4326"
    assert ds.tags()["selection_mode"] == "capture-date"
    assert ds.tags()["dates"] == D2.isoformat()
    assert ds.tags()["tiles_missing"] == "0"
    assert ds.tags()["tiles_capture_date_unknown"] == "0"
    assert (ds.dataset_mask() > 0).all()
    # JPEG is lossy, but a flat fill must survive round-tripping.
    assert ds.read(1).mean() == pytest.approx(200, abs=2)


def test_download_bounds_track_the_aoi(stub) -> None:
    stub(StubBackend([D1]))
    ds = old_imagery.download(AOI, ZOOM, D1)
    west, south, east, north = ds.bounds
    tolerance = 360.0 / (256 * 2**ZOOM)  # one pixel
    assert west == pytest.approx(AOI.bounds[0], abs=tolerance)
    assert south == pytest.approx(AOI.bounds[1], abs=tolerance)
    assert east == pytest.approx(AOI.bounds[2], abs=tolerance)
    assert north == pytest.approx(AOI.bounds[3], abs=tolerance)


def test_download_picks_the_closest_date(stub) -> None:
    stub(StubBackend([D1, D2], colors={D1: 40, D2: 200}))
    assert old_imagery.download(AOI, ZOOM, "2014-01-01").tags()["dates"] == D2.isoformat()
    assert old_imagery.download(AOI, ZOOM, "2006-01-01").tags()["dates"] == D1.isoformat()


def test_download_honours_before_and_after(stub) -> None:
    stub(StubBackend([D1, D2]))
    mid = "2010-01-01"
    assert (
        old_imagery.download(AOI, ZOOM, mid, date_match="before").tags()["dates"] == D1.isoformat()
    )
    assert (
        old_imagery.download(AOI, ZOOM, mid, date_match="after").tags()["dates"] == D2.isoformat()
    )


def test_download_exact_match_failure_is_an_error(stub) -> None:
    stub(StubBackend([D1, D2]))
    with pytest.raises(ValueError, match="No imagery found"):
        old_imagery.download(AOI, ZOOM, "2010-01-01", date_match="exact")


def test_download_masks_missing_tiles(stub) -> None:
    tiles = KeyholeGrid().tiles(AOI, ZOOM, 10_000)
    backend = StubBackend([D1], missing={tiles[0]})
    stub(backend)

    ds = old_imagery.download(AOI, ZOOM, D1)
    mask = ds.dataset_mask()
    assert ds.tags()["tiles_missing"] == "1"
    assert (mask == 0).any() and (mask > 0).any()


def test_download_rejects_bad_date_match(stub) -> None:
    stub(StubBackend([D1]))
    with pytest.raises(ValueError, match="Unknown date_match"):
        old_imagery.download(AOI, ZOOM, D1, date_match="whenever")


def test_download_requires_a_date(stub) -> None:
    stub(StubBackend([D1]))
    with pytest.raises(ValueError, match="date is required"):
        old_imagery.download(AOI, ZOOM, None)


def test_download_dataset_outlives_the_call(stub) -> None:
    """The MemoryFile must stay alive after download() returns."""
    import gc

    stub(StubBackend([D1]))
    ds = old_imagery.download(AOI, ZOOM, D1)
    gc.collect()
    assert ds.read(1).shape[0] > 0


def test_esri_download_records_source_metadata_without_changing_zoom(stub) -> None:
    source = EsriSource(
        provider="Imagery Co",
        description="Aerial survey",
        resolution_m=0.3,
        accuracy_m=2.0,
        min_map_level=16,
        max_map_level=19,
    )
    stub(StubBackend([D1], source=source))

    ds = old_imagery.download(AOI, ZOOM, D1, provider="esri")

    assert ds.tags()["zoom"] == str(ZOOM)
    assert json.loads(ds.tags()["esri_source_metadata"]) == [
        {
            "provider": "Imagery Co",
            "description": "Aerial survey",
            "resolution_m": 0.3,
            "accuracy_m": 2.0,
            "min_map_level": 16,
            "max_map_level": 19,
        }
    ]


# --------------------------------------------------------------------------
# download_tiles
# --------------------------------------------------------------------------
def test_download_tiles_point_returns_the_unchanged_native_payload(stub) -> None:
    backend = stub(StubBackend([D1], colors={D1: 73}))
    point = AOI.centroid

    tiles = old_imagery.download_tiles(point, ZOOM, D1, cache_dir=None)

    assert len(tiles) == 1
    result = tiles[0]
    native = backend.grid.tile_at_point(point.x, point.y, ZOOM)
    assert result.content == _encode_jpeg(np.full((3, 256, 256), 73, dtype=np.uint8))
    assert result.image_format == "jpeg"
    assert result.media_type == "image/jpeg"
    assert result.provider == "google"
    assert result.tile_scheme == "GoogleEarthKeyhole"
    assert (result.row, result.column) == (native.row, native.column)
    assert result.bounds_wgs84 == native.bounds_wgs84
    assert result.capture_date_at_center == D1
    assert result.source_metadata_at_center == old_imagery.SourceMetadata(provider="Test Provider")


def test_download_tiles_multipoint_deduplicates_and_orders_tiles(stub) -> None:
    stub(StubBackend([D1]))
    grid = KeyholeGrid()
    selected = grid.tiles(AOI, ZOOM, 1_000)
    points = MultiPoint([tile.center for tile in reversed(selected)] + [selected[0].center])

    results = old_imagery.download_tiles(points, ZOOM, D1)

    addresses = [(result.row, result.column) for result in results]
    assert addresses == sorted({(tile.row, tile.column) for tile in selected})


def test_download_tiles_polygon_returns_complete_intersecting_tiles(stub) -> None:
    backend = stub(StubBackend([D1]))
    selected = backend.grid.tiles(AOI, ZOOM, 1_000)

    results = old_imagery.download_tiles(AOI, ZOOM, D1)

    assert len(results) == len(selected)
    assert [(tile.row, tile.column) for tile in results] == [
        (tile.row, tile.column) for tile in selected
    ]
    assert all(tile.bounds_wgs84 != AOI.bounds for tile in results)


def test_download_tiles_is_strict_when_one_selected_tile_is_missing(stub) -> None:
    selected = KeyholeGrid().tiles(AOI, ZOOM, 1_000)
    stub(StubBackend([D1], missing={selected[0]}))

    with pytest.raises(ValueError, match="No imagery found for tile"):
        old_imagery.download_tiles(AOI, ZOOM, D1)


def test_download_tiles_is_strict_when_a_payload_is_invalid(stub) -> None:
    backend = stub(StubBackend([D1]))
    backend.download_tile_image = lambda dated: b"not an image"

    with pytest.raises(ValueError, match="invalid image payload"):
        old_imagery.download_tiles(AOI.centroid, ZOOM, D1)


def test_download_tiles_writes_nothing_when_cache_is_disabled(stub, tmp_path, monkeypatch) -> None:
    stub(StubBackend([D1]))
    monkeypatch.chdir(tmp_path)

    old_imagery.download_tiles(AOI.centroid, ZOOM, D1, cache_dir=None)

    assert list(tmp_path.iterdir()) == []


def test_download_tiles_finishes_metadata_before_adapting_raw_payloads(stub) -> None:
    selected = KeyholeGrid().tiles(AOI, ZOOM, 10_000)

    class PhaseBackend(StubBackend):
        def __init__(self):
            super().__init__([D1])
            self.resolved = 0
            self.lock = threading.Lock()

        def dated_tiles(self, tile):
            result = super().dated_tiles(tile)
            with self.lock:
                self.resolved += 1
            return result

        def download_tile_image(self, dated):
            assert self.resolved == len(selected)
            return super().download_tile_image(dated)

    backend = stub(PhaseBackend())
    old_imagery.download_tiles(AOI, ZOOM, D1, cache_dir=None)
    assert backend.resolved == len(selected)


def test_download_tiles_rejects_lines(stub) -> None:
    stub(StubBackend([D1]))
    with pytest.raises(TypeError, match="Point, MultiPoint, Polygon or MultiPolygon"):
        old_imagery.download_tiles(LineString([(0, 0), (1, 1)]), ZOOM, D1)


def test_download_tiles_date_annotation_matches_download() -> None:
    tile_date = inspect.signature(old_imagery.download_tiles).parameters["date"].annotation
    mosaic_date = inspect.signature(old_imagery.download).parameters["date"].annotation
    assert tile_date == mosaic_date


# --------------------------------------------------------------------------
# download_geopackage
# --------------------------------------------------------------------------
def test_download_geopackage_preserves_google_tiles_in_crs84(stub, tmp_path) -> None:
    backend = stub(StubBackend([D1], colors={D1: 73}))
    output = tmp_path / "google.gpkg"

    result = old_imagery.download_geopackage(AOI, ZOOM, D1, output=output, cache_dir=None)

    assert result == output
    selected = backend.grid.tiles(AOI, ZOOM, 1_000)
    with sqlite3.connect(output) as connection:
        stored = connection.execute(
            "SELECT zoom_level, tile_column, tile_row, tile_data "
            "FROM imagery WHERE zoom_level = ? ORDER BY tile_row, tile_column",
            (ZOOM - 1,),
        ).fetchall()
        metadata = json.loads(
            connection.execute(
                "SELECT metadata FROM gpkg_metadata WHERE md_scope = 'dataset'"
            ).fetchone()[0]
        )
        matrix = connection.execute(
            "SELECT matrix_width, matrix_height FROM gpkg_tile_matrix "
            "WHERE table_name = 'imagery' AND zoom_level = ?",
            (ZOOM - 1,),
        ).fetchone()
        crs_wkt = connection.execute(
            "SELECT definition_12_063 FROM gpkg_spatial_ref_sys WHERE srs_id = 4326"
        ).fetchone()
        crs_extension = connection.execute(
            "SELECT COUNT(*) FROM gpkg_extensions "
            "WHERE table_name = 'gpkg_spatial_ref_sys' "
            "AND column_name = 'definition_12_063'"
        ).fetchone()
        tile_metadata = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT m.metadata FROM gpkg_metadata AS m "
                "JOIN gpkg_metadata_reference AS r ON r.md_file_id = m.id "
                "JOIN imagery AS i ON i.id = r.row_id_value "
                "WHERE m.md_scope = 'tile' ORDER BY i.tile_row, i.tile_column"
            )
        ]
        custom_tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'old_imagery_%'"
        ).fetchall()

    expected_addresses = [
        (
            ZOOM - 1,
            tile.column,
            (1 << ZOOM) - 1 - tile.row - (1 << ZOOM) // 4,
        )
        for tile in sorted(
            selected,
            key=lambda tile: (
                (1 << ZOOM) - 1 - tile.row - (1 << ZOOM) // 4,
                tile.column,
            ),
        )
    ]
    assert [(zoom, column, row) for zoom, column, row, _ in stored] == expected_addresses
    assert all(
        payload == _encode_jpeg(np.full((3, 256, 256), 73, dtype=np.uint8))
        for *_, payload in stored
    )
    assert [
        (
            item["native_zoom"],
            item["native_column"],
            item["native_row"],
            item["capture_date"],
        )
        for item in tile_metadata
    ] == [
        (ZOOM, tile.column, tile.row, D1.isoformat())
        for tile in sorted(
            selected,
            key=lambda tile: (
                (1 << ZOOM) - 1 - tile.row - (1 << ZOOM) // 4,
                tile.column,
            ),
        )
    ]
    assert custom_tables == []
    assert metadata["provider"] == "google"
    assert metadata["native_zoom"] == ZOOM
    assert metadata["geopackage_zoom"] == ZOOM - 1
    assert metadata["overviews"]["resampling"] == "lanczos"
    assert metadata["overviews"]["factors"]
    assert matrix == (1 << ZOOM, 1 << (ZOOM - 1))
    assert crs_wkt is not None and crs_wkt[0].startswith("GEODCRS[")
    assert crs_extension == (1,)

    with rasterio.open(output) as dataset:
        assert dataset.crs == rasterio.CRS.from_epsg(4326)
        assert dataset.width > 0 and dataset.height > 0
        assert dataset.read(1).mean() == pytest.approx(73, abs=2)
        expected_bounds = (
            min(tile.bounds_wgs84[0] for tile in selected),
            min(tile.bounds_wgs84[1] for tile in selected),
            max(tile.bounds_wgs84[2] for tile in selected),
            max(tile.bounds_wgs84[3] for tile in selected),
        )
        assert tuple(dataset.bounds) == pytest.approx(expected_bounds)

    with sqlite3.connect(output) as connection:
        overview_counts = dict(
            connection.execute(
                "SELECT zoom_level, COUNT(*) FROM imagery GROUP BY zoom_level"
            )
        )
    assert overview_counts[ZOOM - 1] == len(selected)
    assert overview_counts[ZOOM - 2] > 0


def test_download_geopackage_preserves_esri_web_mercator_tiles(stub, tmp_path) -> None:
    backend = StubBackend([D1], colors={D1: 121})
    backend.grid = MercatorGrid()
    stub(backend)
    output = tmp_path / "esri.gpkg"

    old_imagery.download_geopackage(AOI, ZOOM, D1, output=output, provider="esri", cache_dir=None)

    selected = backend.grid.tiles(AOI, ZOOM, 1_000)
    with sqlite3.connect(output) as connection:
        stored = connection.execute(
            "SELECT zoom_level, tile_column, tile_row, tile_data "
            "FROM imagery WHERE zoom_level = ? ORDER BY tile_row, tile_column",
            (ZOOM,),
        ).fetchall()
        matrix = connection.execute(
            "SELECT matrix_width, matrix_height FROM gpkg_tile_matrix "
            "WHERE table_name = 'imagery' AND zoom_level = ?",
            (ZOOM,),
        ).fetchone()

    assert [(zoom, column, row) for zoom, column, row, _ in stored] == [
        (ZOOM, tile.column, tile.row) for tile in selected
    ]
    assert all(
        payload == _encode_jpeg(np.full((3, 256, 256), 121, dtype=np.uint8))
        for *_, payload in stored
    )
    assert matrix == (1 << ZOOM, 1 << ZOOM)
    with rasterio.open(output) as dataset:
        assert dataset.crs == rasterio.CRS.from_epsg(3857)
        assert dataset.read(1).mean() == pytest.approx(121, abs=2)

    with sqlite3.connect(output) as connection:
        overview_counts = dict(
            connection.execute(
                "SELECT zoom_level, COUNT(*) FROM imagery GROUP BY zoom_level"
            )
        )
    assert overview_counts[ZOOM] == len(selected)
    assert overview_counts[ZOOM - 1] > 0


def test_download_geopackage_pyramids_sparse_multipolygon(stub, tmp_path) -> None:
    backend = stub(StubBackend([D1], colors={D1: 73}))
    sparse_aoi = MultiPolygon(
        [
            box(-122.3999, 37.7921, -122.3995, 37.7925),
            box(-122.3965, 37.7945, -122.3961, 37.7949),
        ]
    )
    output = tmp_path / "sparse.gpkg"

    old_imagery.download_geopackage(
        sparse_aoi, ZOOM, D1, output=output, cache_dir=None
    )

    selected = backend.grid.tiles(sparse_aoi, ZOOM, 1_000)
    assert len(selected) == 2
    with sqlite3.connect(output) as connection:
        counts = dict(
            connection.execute(
                "SELECT zoom_level, COUNT(*) FROM imagery GROUP BY zoom_level"
            )
        )
        source_zoom = ZOOM - 1
        source_payloads = connection.execute(
            "SELECT tile_data FROM imagery WHERE zoom_level = ?",
            (source_zoom,),
        ).fetchall()
        overview_payload = connection.execute(
            "SELECT tile_data FROM imagery WHERE zoom_level = ? LIMIT 1",
            (source_zoom - 1,),
        ).fetchone()[0]

    assert counts[source_zoom] == len(selected)
    assert 0 < counts[source_zoom - 1] <= counts[source_zoom]
    assert min(count for zoom, count in counts.items() if zoom < source_zoom) < counts[source_zoom]
    expected_payload = _encode_jpeg(np.full((3, 256, 256), 73, dtype=np.uint8))
    assert all(payload[0] == expected_payload for payload in source_payloads)
    with rasterio.io.MemoryFile(overview_payload) as memory, memory.open() as overview:
        assert overview.width == overview.height == 256
        assert overview.count == 4
        assert overview.read(4).min() == 0


def test_download_geopackage_refuses_existing_output_before_downloading(stub, tmp_path) -> None:
    backend = stub(StubBackend([D1]))
    output = tmp_path / "existing.gpkg"
    output.write_bytes(b"keep me")

    with pytest.raises(FileExistsError, match="already exists"):
        old_imagery.download_geopackage(AOI, ZOOM, D1, output=output)

    assert output.read_bytes() == b"keep me"
    assert backend.downloads == 0


def test_download_geopackage_can_atomically_overwrite(stub, tmp_path) -> None:
    stub(StubBackend([D1], colors={D1: 88}))
    output = tmp_path / "existing.gpkg"
    output.write_bytes(b"replace me")

    old_imagery.download_geopackage(AOI, ZOOM, D1, output=output, cache_dir=None, overwrite=True)

    with rasterio.open(output) as dataset:
        assert dataset.read(1).mean() == pytest.approx(88, abs=2)


def test_download_geopackage_failure_leaves_no_output(stub, tmp_path) -> None:
    selected = KeyholeGrid().tiles(AOI, ZOOM, 1_000)
    stub(StubBackend([D1], missing={selected[0]}))
    output = tmp_path / "failed.gpkg"

    with pytest.raises(ValueError, match="No imagery found for tile"):
        old_imagery.download_geopackage(AOI, ZOOM, D1, output=output)

    assert not output.exists()


def test_download_geopackage_write_failure_leaves_no_output(stub, tmp_path) -> None:
    stub(StubBackend([D1]))
    output = tmp_path / "failed-write.gpkg"

    with pytest.raises(ValueError, match="reserved"):
        old_imagery.download_geopackage(
            AOI, ZOOM, D1, output=output, table_name="gpkg_contents", cache_dir=None
        )

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_download_geopackage_rejects_unrepresentable_google_zooms(stub, tmp_path) -> None:
    backend = stub(StubBackend([D1]))

    with pytest.raises(ValueError, match="zoom 2 or greater"):
        old_imagery.download_geopackage(box(-1, -1, 1, 1), 1, D1, output=tmp_path / "low.gpkg")

    assert backend.downloads == 0


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------
# Zoom chosen so the AOI yields comfortably more tiles than the switch threshold.
REGION_ZOOM = 20


class RegionBackend(StubBackend):
    """A backend that also offers region-wide footprint queries."""

    def __init__(self, regions, **kwargs):
        super().__init__(kwargs.pop("dates", [D1]), **kwargs)
        self.regions = regions
        self.region_calls = 0
        self.seen_kwargs = None
        self.seen_tiles = None

    def dated_regions(self, aoi, zoom, *, min_date=None, max_date=None, tiles=None):
        self.region_calls += 1
        self.seen_kwargs = {"min_date": min_date, "max_date": max_date}
        self.seen_tiles = tiles
        return [EsriFootprint(d, g, EsriSource(), "Wayback release") for d, g in self.regions]


class MetadataRegionBackend(RegionBackend):
    def dated_regions(self, aoi, zoom, *, min_date=None, max_date=None, tiles=None):
        self.region_calls += 1
        self.seen_kwargs = {"min_date": min_date, "max_date": max_date}
        self.seen_tiles = tiles
        return list(self.regions)


@dataclass(frozen=True)
class StubRelease:
    id: int
    identifier: str
    date: dt.date
    title: str


class CatalogueBackend:
    def __init__(self):
        self.layers = [
            StubRelease(1, "WB_2025_R12", dt.date(2025, 12, 4), "Older release"),
            StubRelease(2, "WB_2026_R01", dt.date(2026, 1, 8), "Newest release"),
        ]


def test_esri_wayback_releases_returns_the_catalogue_newest_first(stub) -> None:
    stub(CatalogueBackend())

    releases = old_imagery.esri_wayback_releases()

    assert list(releases.columns) == ["release_id", "release_date", "release_title"]
    assert list(releases["release_id"]) == ["WB_2026_R01", "WB_2025_R12"]
    assert list(releases["release_title"]) == ["Newest release", "Older release"]
    assert str(releases["release_id"].dtype) == "string"
    assert str(releases["release_title"].dtype) == "string"
    assert str(releases["release_date"].dtype) == "datetime64[ns]"
    assert stub.holder["closed"] is True


class ReleaseBackend(StubBackend):
    """An Esri-shaped backend serving one exact Wayback release."""

    def __init__(self, release_date, capture_date, **kwargs):
        super().__init__([capture_date], **kwargs)
        self.release = StubRelease(
            id=42,
            identifier="WB_2014_R01",
            date=release_date,
            title=f"World Imagery (Wayback {release_date.isoformat()})",
        )
        self.capture_date = capture_date
        self.dated_tile_calls = 0
        self.release_tile_calls = 0
        self.release_metadata_requests = []

    def release_by_identifier(self, identifier):
        if identifier != self.release.identifier:
            raise ValueError(f"No Esri Wayback release has identifier {identifier}")
        return self.release

    def release_on_or_before(self, visible_date):
        if visible_date < self.release.date:
            raise ValueError(f"No Esri Wayback release was visible on or before {visible_date}")
        return self.release

    def tile_at_release(self, tile, release, *, include_metadata=True):
        assert release is self.release
        self.release_tile_calls += 1
        self.release_metadata_requests.append(include_metadata)
        return StubDated(
            tile,
            self.capture_date if include_metadata else None,
            release.id,
        )

    def dated_tiles(self, tile):
        self.dated_tile_calls += 1
        return super().dated_tiles(tile)


RELEASE_DATE = dt.date(2014, 2, 20)


def test_availability_no_longer_accepts_release_selectors(stub) -> None:
    """Release snapshots moved to esri_mosaic_as_of; availability means capture dates."""
    stub(ReleaseBackend(RELEASE_DATE, D1))
    for kwargs in (
        {"esri_wayback_release_id": "WB_2014_R01"},
        {"esri_wayback_as_of_date": RELEASE_DATE},
    ):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            old_imagery.availability(AOI, ZOOM, provider="esri", **kwargs)


def test_availability_attrs_no_longer_carry_a_selection_mode(stub) -> None:
    """With only one mode left, availability has nothing to disambiguate."""
    stub(StubBackend([D1]))
    gdf = old_imagery.availability(AOI, ZOOM)
    assert "selection_mode" not in gdf.attrs
    assert gdf.attrs["method"] == "per-tile"


def test_download_can_select_one_exact_esri_release_by_identifier(stub) -> None:
    backend = stub(ReleaseBackend(RELEASE_DATE, D1, colors={D1: 77}))
    ds = old_imagery.download(
        AOI,
        ZOOM,
        provider="esri",
        esri_wayback_release_id=backend.release.identifier,
    )

    tags = ds.tags()
    assert tags["selection_mode"] == "esri-wayback-release"
    assert tags["esri_wayback_release_id"] == backend.release.identifier
    assert tags["esri_wayback_catalogue_date"] == RELEASE_DATE.isoformat()
    assert tags["esri_wayback_release_title"] == backend.release.title
    assert "esri_wayback_release_resolution" not in tags  # only one selector now
    assert tags["dates"] == D1.isoformat()
    assert "target_date" not in tags
    assert "date_match" not in tags
    assert ds.read(1).mean() == pytest.approx(77, abs=2)
    assert backend.dated_tile_calls == 0


def test_release_download_keeps_tiles_with_unknown_capture_date(stub) -> None:
    backend = stub(ReleaseBackend(RELEASE_DATE, None))
    ds = old_imagery.download(
        AOI,
        ZOOM,
        provider="esri",
        esri_wayback_release_id=backend.release.identifier,
    )
    tags = ds.tags()
    assert "dates" not in tags
    assert tags["tiles_capture_date_unknown"] == tags["tiles_total"]
    assert (ds.dataset_mask() > 0).all()


def test_download_tiles_exact_release_carries_separate_release_provenance(stub) -> None:
    backend = stub(ReleaseBackend(RELEASE_DATE, D1))

    result = old_imagery.download_tiles(
        AOI.centroid,
        ZOOM,
        provider="esri",
        esri_wayback_release_id=backend.release.identifier,
    )[0]

    assert result.release_id == backend.release.identifier
    assert result.release_date == RELEASE_DATE
    assert result.release_title == backend.release.title
    assert result.capture_date_at_center == D1
    assert backend.release_metadata_requests == [True]


def test_download_tiles_can_skip_exact_release_metadata(stub) -> None:
    backend = stub(ReleaseBackend(RELEASE_DATE, D1))

    result = old_imagery.download_tiles(
        AOI.centroid,
        ZOOM,
        provider="esri",
        esri_wayback_release_id=backend.release.identifier,
        include_metadata=False,
    )[0]

    assert result.release_id == backend.release.identifier
    assert result.capture_date_at_center is None
    assert result.source_metadata_at_center is None
    assert backend.release_metadata_requests == [False]


def test_download_no_longer_accepts_an_as_of_date(stub) -> None:
    """Service dates resolve in esri_mosaic_as_of, which reports the release_id."""
    stub(ReleaseBackend(RELEASE_DATE, D1))
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        old_imagery.download(AOI, ZOOM, provider="esri", esri_wayback_as_of_date=RELEASE_DATE)


def test_release_id_from_a_mosaic_round_trips_into_download(stub) -> None:
    """The documented workflow: map the seams, then fetch that exact release."""
    backend = stub(MosaicBackend({ZOOM: [(D1, AOI)]}))
    seams = old_imagery.esri_mosaic_as_of(AOI, ZOOM, RELEASE_DATE)

    stub(ReleaseBackend(RELEASE_DATE, D1, colors={D1: 66}))
    ds = old_imagery.download(
        AOI, ZOOM, provider="esri", esri_wayback_release_id=seams.attrs["release_id"]
    )
    tags = ds.tags()
    assert tags["esri_wayback_release_id"] == backend.release.identifier
    assert tags["esri_wayback_catalogue_date"] == RELEASE_DATE.isoformat()
    assert ds.read(1).mean() == pytest.approx(66, abs=2)


def test_download_rejects_an_empty_release_id(stub) -> None:
    stub(ReleaseBackend(RELEASE_DATE, D1))
    with pytest.raises(ValueError, match="cannot be empty"):
        old_imagery.download(AOI, ZOOM, provider="esri", esri_wayback_release_id="")


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (
            {"provider": "google", "esri_wayback_release_id": "WB_2014_R01"},
            "requires provider='esri'",
        ),
        (
            {
                "provider": "esri",
                "date": D1,
                "esri_wayback_release_id": "WB_2014_R01",
            },
            "Choose either date",
        ),
        (
            {
                "provider": "esri",
                "esri_wayback_release_id": "WB_2014_R01",
                "date_match": "exact",
            },
            "date_match cannot be set",
        ),
    ],
)
def test_release_download_rejects_mixed_selection_modes(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        old_imagery.download(AOI, ZOOM, **kwargs)


def test_esri_always_uses_footprints(stub) -> None:
    """No `method` option: Esri resolves through footprints, small AOI or large."""
    tiny = box(-122.3999, 37.7929, -122.3997, 37.7931)
    for aoi, zoom in ((tiny, ZOOM), (AOI, REGION_ZOOM)):
        backend = stub(RegionBackend([(D1, aoi)]))
        gdf = old_imagery.availability(aoi, zoom, provider="esri", max_tiles=10_000)
        assert gdf.attrs["method"] == "region-query"
        assert backend.region_calls == 1
        assert list(gdf["date"]) == [D1]


def test_availability_passes_through_distinct_esri_source_metadata(stub) -> None:
    left = box(-122.4000, 37.7920, -122.3980, 37.7950)
    right = box(-122.3980, 37.7920, -122.3960, 37.7950)
    source_a = EsriSource(provider="Provider A", resolution_m=1.0, accuracy_m=5.0)
    source_b = EsriSource(provider="Provider B", resolution_m=0.3, accuracy_m=2.0)
    stub(
        MetadataRegionBackend(
            [
                EsriFootprint(D1, left, source_a, "Wayback release"),
                EsriFootprint(D1, right, source_b, "Wayback release"),
            ]
        )
    )

    gdf = old_imagery.availability(AOI, ZOOM, provider="esri", max_tiles=10_000)

    row = gdf.iloc[0]
    assert row["source_providers"] == ("Provider A", "Provider B")
    assert row["source_resolutions_m"] == (0.3, 1.0)
    assert row["source_accuracies_m"] == (2.0, 5.0)


def test_availability_no_longer_accepts_a_method(stub) -> None:
    """The knob is gone: footprints are strictly better, so there was no trade-off."""
    stub(RegionBackend([(D1, AOI)]))
    for method in ("region", "per-tile", "auto"):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            old_imagery.availability(AOI, ZOOM, provider="esri", method=method)


def test_google_never_uses_the_region_query(stub) -> None:
    stub(StubBackend([D1]))
    gdf = old_imagery.availability(AOI, REGION_ZOOM, max_tiles=10_000)
    assert gdf.attrs["method"] == "per-tile"


def test_region_query_geometry_is_the_footprint_not_whole_tiles(stub) -> None:
    """The whole point of the region path: exact footprints, not tile unions."""
    half = box(-122.4000, 37.7920, -122.3980, 37.7950)  # left half of the AOI
    stub(RegionBackend([(D1, half)]))
    gdf = old_imagery.availability(AOI, REGION_ZOOM, provider="esri", max_tiles=10_000)

    geom = gdf.geometry.iloc[0]
    assert geom.area == pytest.approx(half.intersection(AOI).area, rel=1e-9)
    assert geom.area < AOI.area
    # Partial coverage must be reflected in the tile-based columns too.
    assert 0 < gdf["coverage"].iloc[0] < 1


def test_region_query_clips_to_the_aoi(stub) -> None:
    huge = box(-123.0, 37.0, -122.0, 38.0)  # far larger than the AOI
    stub(RegionBackend([(D1, huge)]))
    gdf = old_imagery.availability(AOI, REGION_ZOOM, provider="esri", max_tiles=10_000)

    geom = gdf.geometry.iloc[0]
    assert geom.within(AOI.buffer(1e-9))
    assert geom.area == pytest.approx(AOI.area, rel=1e-9)
    assert gdf["coverage"].iloc[0] == pytest.approx(1.0)


def test_region_query_unions_footprints_sharing_a_date(stub) -> None:
    left = box(-122.4000, 37.7920, -122.3985, 37.7950)
    right = box(-122.3975, 37.7920, -122.3960, 37.7950)
    stub(RegionBackend([(D1, left), (D1, right)]))
    gdf = old_imagery.availability(AOI, REGION_ZOOM, provider="esri", max_tiles=10_000)

    assert len(gdf) == 1
    expected = left.union(right).intersection(AOI).area
    assert gdf.geometry.iloc[0].area == pytest.approx(expected, rel=1e-9)


def test_region_query_passes_date_bounds_to_the_backend(stub) -> None:
    backend = stub(RegionBackend([(D1, AOI)]))
    old_imagery.availability(
        AOI,
        REGION_ZOOM,
        provider="esri",
        min_date="2000-01-01",
        max_date="2020-01-01",
        max_tiles=10_000,
    )
    assert backend.seen_kwargs == {
        "min_date": dt.date(2000, 1, 1),
        "max_date": dt.date(2020, 1, 1),
    }


def test_region_query_with_no_results_is_empty(stub) -> None:
    stub(RegionBackend([]))
    gdf = old_imagery.availability(AOI, REGION_ZOOM, provider="esri", max_tiles=10_000)
    assert len(gdf) == 0
    assert list(gdf.columns) == api.AVAILABILITY_COLUMNS
    assert gdf.attrs["n_aoi_tiles"] > 0
    assert gdf.attrs["method"] == "region-query"


def test_region_query_drops_footprints_outside_the_aoi(stub) -> None:
    elsewhere = box(-70.0, 40.0, -69.9, 40.1)
    stub(RegionBackend([(D1, elsewhere)]))
    gdf = old_imagery.availability(AOI, REGION_ZOOM, provider="esri", max_tiles=10_000)
    assert len(gdf) == 0


def test_unknown_provider_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        old_imagery.availability(AOI, ZOOM, provider="bing")


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2020-01-02", dt.date(2020, 1, 2)),
        (dt.date(2020, 1, 2), dt.date(2020, 1, 2)),
        (dt.datetime(2020, 1, 2, 3, 4), dt.date(2020, 1, 2)),
        (None, None),
    ],
)
def test_as_date(value, expected) -> None:
    assert api._as_date(value) == expected


def test_decode_image_handles_greyscale() -> None:
    grey = np.full((1, 256, 256), 90, dtype=np.uint8)
    from rasterio.io import MemoryFile

    with MemoryFile(ext=".jpg") as mem:
        with mem.open(driver="JPEG", width=256, height=256, count=1, dtype="uint8") as dst:
            dst.write(grey)
        raw = mem.read()

    arr = api._decode_image(raw)
    assert arr is not None and arr.shape == (3, 256, 256)
    assert np.allclose(arr[0], arr[1]) and np.allclose(arr[1], arr[2])


def test_decode_image_rejects_garbage() -> None:
    assert api._decode_image(b"not an image at all") is None


# --------------------------------------------------------------------------
# zoom caps
#
# The tile schemes address deeper than either service publishes imagery for
# (Keyhole to 30, Web Mercator to 23), so the cap is a separate, shallower
# limit taken from upstream's documented practical maxima.
# --------------------------------------------------------------------------
def test_max_imagery_zoom_is_below_the_addressing_limits() -> None:
    from old_imagery._keyhole import MAX_LEVEL
    from old_imagery._region import MercatorGrid

    assert api.MAX_IMAGERY_ZOOM["google"] < MAX_LEVEL
    assert api.MAX_IMAGERY_ZOOM["esri"] < MercatorGrid.max_level


@pytest.mark.parametrize(
    "provider,zoom",
    [("google", 22), ("google", 30), ("esri", 21), ("esri", 23)],
)
def test_availability_rejects_zoom_beyond_published_imagery(provider, zoom) -> None:
    with pytest.raises(ValueError, match="carry no imagery"):
        old_imagery.availability(AOI, zoom, provider=provider)


@pytest.mark.parametrize("provider,zoom", [("google", 22), ("esri", 21)])
def test_download_rejects_zoom_beyond_published_imagery(provider, zoom) -> None:
    with pytest.raises(ValueError, match="carry no imagery"):
        old_imagery.download(AOI, zoom, D1, provider=provider)


def test_zoom_at_the_cap_is_allowed(stub) -> None:
    """The cap is inclusive — 21 is the deepest usable Google zoom, not the first bad one."""
    stub(StubBackend([D1]))
    gdf = old_imagery.availability(AOI, api.MAX_IMAGERY_ZOOM["google"], max_tiles=1_000_000)
    assert list(gdf["date"]) == [D1]


def test_zoom_cap_is_checked_before_any_network_client_is_built(monkeypatch) -> None:
    """A rejected zoom must not open an HTTP client or touch the cache."""

    def explode(*args, **kwargs):
        raise AssertionError("backend was constructed despite an invalid zoom")

    monkeypatch.setattr(api, "_backend", explode)
    with pytest.raises(ValueError, match="carry no imagery"):
        old_imagery.availability(AOI, 25)


def test_unknown_provider_still_reports_the_provider_error() -> None:
    """Zoom validation must not mask a bad provider name."""
    with pytest.raises(ValueError, match="Unknown provider"):
        old_imagery.availability(AOI, 25, provider="bing")


# --------------------------------------------------------------------------
# esri_mosaic_as_of
# --------------------------------------------------------------------------
class MosaicBackend:
    """An Esri-shaped backend serving one release's capture footprints."""

    from old_imagery._region import MercatorGrid as _Grid

    grid = _Grid()

    def __init__(self, footprints_by_zoom, release_date=RELEASE_DATE, raises=None):
        self.footprints_by_zoom = footprints_by_zoom
        self.raises = raises
        self.release = StubRelease(
            id=42,
            identifier="WB_2014_R01",
            date=release_date,
            title=f"World Imagery (Wayback {release_date.isoformat()})",
        )
        self.asked_zooms: list[int] = []
        self.asked_release = None
        self.asked_identifier = None
        self.seen_max_footprints: list[int] = []
        self.peak_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()

    def release_on_or_before(self, visible_date):
        if visible_date < self.release.date:
            raise ValueError(
                f"No Esri Wayback release was visible on or before {visible_date}; "
                f"the earliest catalogue release is {self.release.date}"
            )
        return self.release

    def release_by_identifier(self, identifier):
        self.asked_identifier = identifier
        return self.release

    def release_footprints(self, layer, aoi, zoom, *, max_footprints=500):
        with self._lock:
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            time.sleep(0.01)  # widen the window so real overlap is observable
            self.asked_release = layer
            self.asked_zooms.append(zoom)
            self.seen_max_footprints.append(max_footprints)
            if self.raises is not None:
                raise self.raises
            return [
                row if isinstance(row, EsriFootprint) else EsriFootprint(*row, EsriSource())
                for row in self.footprints_by_zoom.get(zoom, [])
            ]
        finally:
            with self._lock:
                self._in_flight -= 1


LEFT = box(-122.4000, 37.7920, -122.3980, 37.7950)  # exactly half of AOI
RIGHT = box(-122.3980, 37.7920, -122.3960, 37.7950)  # the other half


def test_mosaic_returns_one_row_per_zoom_and_date(stub) -> None:
    stub(MosaicBackend({18: [(D1, LEFT), (D2, RIGHT)]}))
    gdf = old_imagery.esri_mosaic_as_of(AOI, 18, RELEASE_DATE)

    assert list(gdf.columns) == api.ESRI_MOSAIC_COLUMNS
    assert gdf.crs == "EPSG:4326"
    assert list(gdf["date"]) == [D2, D1]  # newest first within a zoom
    assert set(gdf["zoom"]) == {18}
    assert (gdf["release_id"] == "WB_2014_R01").all()


def test_mosaic_geometry_is_the_footprint_not_a_tile_union(stub) -> None:
    """The whole point: real seams, not the provider's tile grid."""
    stub(MosaicBackend({18: [(D1, LEFT)]}))
    gdf = old_imagery.esri_mosaic_as_of(AOI, 18, RELEASE_DATE)

    geom = gdf.geometry.iloc[0]
    assert geom.area == pytest.approx(LEFT.area, rel=1e-9)
    assert geom.area < AOI.area


def test_mosaic_clips_footprints_to_the_aoi(stub) -> None:
    huge = box(-123.0, 37.0, -122.0, 38.0)
    stub(MosaicBackend({18: [(D1, huge)]}))
    gdf = old_imagery.esri_mosaic_as_of(AOI, 18, RELEASE_DATE)

    geom = gdf.geometry.iloc[0]
    assert geom.within(AOI.buffer(1e-9))
    assert geom.area == pytest.approx(AOI.area, rel=1e-9)
    assert gdf["area_fraction"].iloc[0] == pytest.approx(1.0, rel=1e-6)


def test_mosaic_dissolves_footprints_sharing_a_zoom_and_date(stub) -> None:
    stub(MosaicBackend({18: [(D1, LEFT), (D1, RIGHT)]}))
    gdf = old_imagery.esri_mosaic_as_of(AOI, 18, RELEASE_DATE)

    assert len(gdf) == 1
    assert gdf.geometry.iloc[0].area == pytest.approx(AOI.area, rel=1e-9)


def test_mosaic_preserves_distinct_sources_sharing_a_capture_date(stub) -> None:
    source_a = EsriSource(provider="Provider A", resolution_m=1.0, accuracy_m=5.0)
    source_b = EsriSource(provider="Provider B", resolution_m=0.3, accuracy_m=2.0)
    stub(
        MosaicBackend(
            {
                18: [
                    EsriFootprint(D1, LEFT, source_a),
                    EsriFootprint(D1, RIGHT, source_b),
                ]
            }
        )
    )

    gdf = old_imagery.esri_mosaic_as_of(AOI, 18, RELEASE_DATE)

    assert len(gdf) == 2
    assert set(gdf["source_provider"]) == {"Provider A", "Provider B"}
    assert set(gdf["source_resolution_m"]) == {1.0, 0.3}
    assert set(gdf["source_accuracy_m"]) == {5.0, 2.0}


def test_mosaic_area_fraction_is_partial_for_partial_cover(stub) -> None:
    stub(MosaicBackend({18: [(D1, LEFT)]}))
    gdf = old_imagery.esri_mosaic_as_of(AOI, 18, RELEASE_DATE)
    assert gdf["area_fraction"].iloc[0] == pytest.approx(0.5, rel=1e-3)


def test_mosaic_drops_footprints_outside_the_aoi(stub) -> None:
    elsewhere = box(-70.0, 40.0, -69.9, 40.1)
    stub(MosaicBackend({18: [(D1, elsewhere)]}))
    gdf = old_imagery.esri_mosaic_as_of(AOI, 18, RELEASE_DATE)
    assert len(gdf) == 0
    assert list(gdf.columns) == api.ESRI_MOSAIC_COLUMNS
    assert gdf.attrs["release_id"] == "WB_2014_R01"


def test_mosaic_reports_a_different_date_per_zoom(stub) -> None:
    """Wayback composes per scale, so zoom is a real axis, not a resolution knob."""
    backend = stub(MosaicBackend({13: [(D2, AOI)], 19: [(D1, AOI)]}))
    gdf = old_imagery.esri_mosaic_as_of(AOI, [19, 13], RELEASE_DATE)

    assert list(gdf["zoom"]) == [13, 19]  # sorted ascending
    assert list(gdf["date"]) == [D2, D1]
    assert sorted(backend.asked_zooms) == [13, 19]
    assert gdf.attrs["zooms"] == [13, 19]


def test_mosaic_accepts_a_single_int_zoom(stub) -> None:
    backend = stub(MosaicBackend({18: [(D1, AOI)]}))
    gdf = old_imagery.esri_mosaic_as_of(AOI, 18, RELEASE_DATE)
    assert backend.asked_zooms == [18]
    assert gdf.attrs["zooms"] == [18]


def test_mosaic_deduplicates_and_sorts_requested_zooms(stub) -> None:
    backend = stub(MosaicBackend({18: [(D1, AOI)], 19: [(D1, AOI)]}))
    old_imagery.esri_mosaic_as_of(AOI, [19, 18, 19], RELEASE_DATE)
    assert sorted(backend.asked_zooms) == [18, 19]


def test_mosaic_resolves_the_latest_release_on_or_before_the_date(stub) -> None:
    backend = stub(MosaicBackend({18: [(D1, AOI)]}, release_date=RELEASE_DATE))
    later = RELEASE_DATE + dt.timedelta(days=400)
    gdf = old_imagery.esri_mosaic_as_of(AOI, 18, later)

    assert backend.asked_release is backend.release
    assert gdf.attrs["release_date"] == RELEASE_DATE  # publication date
    assert gdf.attrs["as_of"] == later  # what was asked for
    assert gdf.attrs["release_title"] == backend.release.title


def test_mosaic_rejects_a_date_before_the_archive(stub) -> None:
    stub(MosaicBackend({18: [(D1, AOI)]}))
    with pytest.raises(ValueError, match="earliest catalogue release"):
        old_imagery.esri_mosaic_as_of(AOI, 18, RELEASE_DATE - dt.timedelta(days=1))


def test_mosaic_accepts_an_iso_string_date(stub) -> None:
    stub(MosaicBackend({18: [(D1, AOI)]}))
    gdf = old_imagery.esri_mosaic_as_of(AOI, 18, RELEASE_DATE.isoformat())
    assert gdf.attrs["as_of"] == RELEASE_DATE


def test_mosaic_accepts_an_exact_release_id(stub) -> None:
    backend = stub(MosaicBackend({18: [(D1, AOI)]}))

    gdf = old_imagery.esri_mosaic_as_of(AOI, 18, "WB_2014_R01")

    assert backend.asked_identifier == "WB_2014_R01"
    assert backend.asked_release is backend.release
    assert gdf.attrs["as_of"] == "WB_2014_R01"
    assert gdf.attrs["release_id"] == "WB_2014_R01"


@pytest.mark.parametrize("zoom", [21, 99])
def test_mosaic_rejects_zoom_beyond_published_imagery(stub, zoom) -> None:
    stub(MosaicBackend({}))
    with pytest.raises(ValueError, match="deeper than esri publishes"):
        old_imagery.esri_mosaic_as_of(AOI, zoom, RELEASE_DATE)


def test_mosaic_rejects_a_negative_zoom(stub) -> None:
    stub(MosaicBackend({}))
    with pytest.raises(ValueError, match="zoom must be at least 0"):
        old_imagery.esri_mosaic_as_of(AOI, -1, RELEASE_DATE)


def test_mosaic_rejects_an_empty_zoom_sequence(stub) -> None:
    stub(MosaicBackend({}))
    with pytest.raises(ValueError, match="cannot be an empty sequence"):
        old_imagery.esri_mosaic_as_of(AOI, [], RELEASE_DATE)


def test_mosaic_rejects_a_non_integer_zoom(stub) -> None:
    stub(MosaicBackend({}))
    with pytest.raises(TypeError, match="must be ints"):
        old_imagery.esri_mosaic_as_of(AOI, [18.5], RELEASE_DATE)
    with pytest.raises(TypeError, match="must be an int or a sequence"):
        old_imagery.esri_mosaic_as_of(AOI, "18", RELEASE_DATE)


def test_mosaic_validates_zoom_before_any_network_client_is_built(monkeypatch) -> None:
    def explode(provider, cache_dir):
        raise AssertionError("a client must not be built for an invalid zoom")

    monkeypatch.setattr(api, "_backend", explode)
    with pytest.raises(ValueError, match="deeper than esri publishes"):
        old_imagery.esri_mosaic_as_of(AOI, 21, RELEASE_DATE)


def test_mosaic_passes_max_footprints_through(stub) -> None:
    backend = stub(MosaicBackend({18: [(D1, AOI)]}))
    old_imagery.esri_mosaic_as_of(AOI, 18, RELEASE_DATE, max_footprints=7)
    assert backend.seen_max_footprints == [7]


def test_mosaic_propagates_a_refused_partial_answer(stub) -> None:
    """release_footprints raises rather than returning holes; don't swallow it."""
    stub(MosaicBackend({}, raises=old_imagery.RequestFailed("incomplete")))
    with pytest.raises(old_imagery.RequestFailed):
        old_imagery.esri_mosaic_as_of(AOI, 18, RELEASE_DATE)


def test_mosaic_closes_the_client_even_when_a_zoom_fails(stub) -> None:
    stub(MosaicBackend({}, raises=old_imagery.RequestFailed("incomplete")))
    with pytest.raises(old_imagery.RequestFailed):
        old_imagery.esri_mosaic_as_of(AOI, 18, RELEASE_DATE)
    assert stub.holder.get("closed") is True


def test_mosaic_never_exceeds_the_wayback_concurrency_cap(stub) -> None:
    """The cap is on requests in flight, so it must not multiply per zoom."""
    from old_imagery._concurrency import workers_for

    cap = workers_for("esri", 10_000)  # the policy's ceiling for Esri
    # More zooms than the cap, or the assertion could not fail: with only `cap`
    # tasks a capped pool and an uncapped one are indistinguishable.
    zooms = list(range(0, 21))  # every zoom Esri publishes imagery for
    assert len(zooms) > cap
    backend = stub(MosaicBackend({z: [(D1, AOI)] for z in zooms}))

    old_imagery.esri_mosaic_as_of(AOI, zooms, RELEASE_DATE)
    assert backend.peak_in_flight <= cap
    # And the work really did overlap, so the bound above means something.
    assert backend.peak_in_flight > 1
    assert sorted(backend.asked_zooms) == zooms


def test_every_entry_point_rejects_a_zero_area_aoi(stub) -> None:
    """area_fraction and coverage are fractions of the AOI, so it needs one."""
    from shapely.geometry import LineString, Point

    for geometry in (LineString([(-122.400, 37.792), (-122.396, 37.795)]), Point(-122.4, 37.79)):
        stub(MosaicBackend({18: [(D1, AOI)]}))
        with pytest.raises(ValueError, match="no area"):
            old_imagery.esri_mosaic_as_of(geometry, 18, RELEASE_DATE)

        stub(StubBackend([D1]))
        with pytest.raises(ValueError, match="no area"):
            old_imagery.availability(geometry, ZOOM)
        with pytest.raises(ValueError, match="no area"):
            old_imagery.download(geometry, ZOOM, D1)


# --------------------------------------------------------------------------
# concurrency policy
# --------------------------------------------------------------------------
def test_workers_never_exceed_the_number_of_tasks() -> None:
    """A three-tile AOI has no use for sixteen threads."""
    from old_imagery._concurrency import workers_for

    assert workers_for("google", 3) == 3
    assert workers_for("esri", 1) == 1
    assert workers_for("google", 0) == 1  # a pool of zero is not constructible


def test_workers_are_capped_per_provider() -> None:
    from old_imagery._concurrency import workers_for

    google, esri = workers_for("google", 10_000), workers_for("esri", 10_000)
    assert google > 1 and esri > 1
    # Esri is the more delicate service; upstream measured going wider as slower.
    assert esri < google


def test_unknown_provider_gets_the_more_cautious_cap() -> None:
    from old_imagery._concurrency import workers_for

    assert workers_for("bing", 10_000) == workers_for("esri", 10_000)


def test_download_pool_is_sized_from_the_provider_not_the_cpu_count(stub) -> None:
    """The parallel work is network-bound, so os.cpu_count() is the wrong basis."""
    from old_imagery import _concurrency

    seen = []
    real = _concurrency.adaptive_metadata_map

    def recording(workload, function, items, **kwargs):
        seen.append((workload, len(items)))
        return real(workload, function, items, **kwargs)

    stub(StubBackend([D1]))
    tiles = KeyholeGrid().tiles(AOI, ZOOM, 10_000)
    api.adaptive_metadata_map = recording
    try:
        old_imagery.download(AOI, ZOOM, D1)
    finally:
        api.adaptive_metadata_map = real

    assert seen == [("google-quadtree", len(tiles))]


def test_coverage_is_area_not_touched_tiles(stub) -> None:
    """A footprint clipping a sliver off every tile must not read as full cover.

    This is the defect area-based coverage exists to fix: with a tile count, a
    geometry touching all AOI tiles reported coverage 1.0 over almost no ground.
    """
    minx, miny, maxx, maxy = AOI.bounds
    sliver = box(minx, miny, maxx, miny + (maxy - miny) * 0.02)  # 2% of the AOI
    stub(RegionBackend([(D1, sliver)]))

    gdf = old_imagery.availability(AOI, REGION_ZOOM, provider="esri", max_tiles=10_000)

    assert gdf["coverage"].iloc[0] == pytest.approx(0.02, abs=0.005)
    assert not bool(gdf["complete"].iloc[0])


def test_full_cover_is_complete(stub) -> None:
    stub(RegionBackend([(D1, AOI)]))
    gdf = old_imagery.availability(AOI, REGION_ZOOM, provider="esri", max_tiles=10_000)
    assert gdf["coverage"].iloc[0] == pytest.approx(1.0)
    assert bool(gdf["complete"].iloc[0])


def test_availability_no_longer_reports_a_tile_count_column(stub) -> None:
    """The tile grid is transport, not something the caller chose or can see."""
    stub(StubBackend([D1]))
    gdf = old_imagery.availability(AOI, ZOOM)
    assert "n_tiles" not in gdf.columns
    assert list(gdf.columns) == api.AVAILABILITY_COLUMNS


def test_public_tile_guard_matches_the_internal_default() -> None:
    """The number is inlined in the signatures; keep it from drifting."""
    import inspect

    from old_imagery._region import MAX_TILES

    for fn in (old_imagery.availability, old_imagery.download):
        default = inspect.signature(fn).parameters["max_tiles"].default
        assert default == MAX_TILES == 1_000, fn.__name__


def test_option_values_are_visible_in_the_signature() -> None:
    """No alias to look up: help() and IDE tooltips show the accepted values."""
    import inspect
    import typing

    hints = typing.get_type_hints(old_imagery.availability)
    assert typing.get_args(hints["provider"]) == ("google", "esri")

    hints = typing.get_type_hints(old_imagery.download)
    assert typing.get_args(hints["date_match"]) == ("closest", "exact", "before", "after")

    # And they are rendered literally, not as a name the reader must resolve.
    text = str(inspect.signature(old_imagery.availability))
    assert "Literal['google', 'esri']" in text


def test_aoi_annotation_names_where_the_type_comes_from() -> None:
    """`BaseGeometry` alone does not tell a reader it is shapely's."""
    import inspect

    for fn in (old_imagery.availability, old_imagery.download, old_imagery.esri_mosaic_as_of):
        annotation = inspect.signature(fn).parameters["aoi"].annotation
        # Spelled out rather than BaseGeometry: it says both where the type
        # comes from and that the AOI has to enclose an area.
        assert annotation == (
            "shapely.Polygon | shapely.MultiPolygon | shapely.GeometryCollection"
        ), fn.__name__
