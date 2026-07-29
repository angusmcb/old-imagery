"""Offline tests for the public API, using a stub backend instead of network."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pytest
from shapely.geometry import box

import old_imagery
from old_imagery import api
from old_imagery._keyhole import KeyholeTile
from old_imagery._region import KeyholeGrid

AOI = box(-122.4000, 37.7920, -122.3960, 37.7950)
ZOOM = 17


@dataclass(frozen=True)
class StubDated:
    tile: KeyholeTile
    date: dt.date | None
    provider: int


class StubBackend:
    """A DbRoot-shaped backend serving synthetic tiles with no network."""

    grid = KeyholeGrid()

    def __init__(self, dates, colors=None, missing=()):
        self.dates = list(dates)
        self.colors = colors or {}
        self.missing = set(missing)
        self.downloads = 0

    def dated_tiles(self, tile):
        if tile in self.missing:
            return []
        return [StubDated(tile, d, 7) for d in self.dates]

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


def test_availability_reports_partial_coverage(stub) -> None:
    backend = StubBackend([D1])
    tiles = KeyholeGrid().tiles(AOI, ZOOM, 10_000)
    assert len(tiles) > 1
    backend.missing = {tiles[0]}
    stub(backend)

    gdf = old_imagery.availability(AOI, ZOOM)
    assert gdf["n_tiles"].iloc[0] == len(tiles) - 1
    assert gdf["coverage"].iloc[0] == pytest.approx((len(tiles) - 1) / len(tiles))
    assert not bool(gdf["complete"].iloc[0])


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
    assert gdf.attrs["selection_mode"] == "capture-date"


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

    def dated_regions(self, aoi, zoom, *, min_date=None, max_date=None, max_workers=16):
        self.region_calls += 1
        self.seen_kwargs = {"min_date": min_date, "max_date": max_date}
        return [(d, g, "Wayback release") for d, g in self.regions]


@dataclass(frozen=True)
class StubRelease:
    id: int
    date: dt.date
    title: str


class ReleaseBackend(StubBackend):
    """An Esri-shaped backend serving one exact Wayback release."""

    def __init__(self, release_date, capture_date, **kwargs):
        super().__init__([capture_date], **kwargs)
        self.release = StubRelease(
            id=42,
            date=release_date,
            title=f"World Imagery (Wayback {release_date.isoformat()})",
        )
        self.capture_date = capture_date
        self.dated_tile_calls = 0
        self.release_tile_calls = 0

    def release_at(self, release_date):
        if release_date != self.release.date:
            raise ValueError(f"No Esri Wayback release was published on {release_date}")
        return self.release

    def tile_at_release(self, tile, release):
        assert release is self.release
        self.release_tile_calls += 1
        return StubDated(tile, self.capture_date, release.id)

    def dated_tiles(self, tile):
        self.dated_tile_calls += 1
        return super().dated_tiles(tile)


RELEASE_DATE = dt.date(2014, 2, 20)


def test_availability_can_select_one_exact_esri_release(stub) -> None:
    backend = stub(ReleaseBackend(RELEASE_DATE, D1))
    gdf = old_imagery.availability(
        AOI,
        ZOOM,
        provider="esri",
        esri_wayback_release_date=RELEASE_DATE,
    )

    assert list(gdf["date"]) == [D1]
    assert (gdf["providers"] == backend.release.title).all()
    assert gdf.attrs["selection_mode"] == "esri-wayback-release"
    assert gdf.attrs["method"] == "esri-wayback-release"
    assert gdf.attrs["esri_wayback_release_date"] == RELEASE_DATE
    assert gdf.attrs["esri_wayback_release_title"] == backend.release.title
    assert backend.release_tile_calls == gdf.attrs["n_aoi_tiles"]
    assert backend.dated_tile_calls == 0


def test_release_availability_omits_unknown_capture_dates_but_keeps_metadata(stub) -> None:
    backend = stub(ReleaseBackend(RELEASE_DATE, None))
    gdf = old_imagery.availability(
        AOI,
        ZOOM,
        provider="esri",
        esri_wayback_release_date=RELEASE_DATE,
    )

    assert len(gdf) == 0
    assert gdf.attrs["selection_mode"] == "esri-wayback-release"
    assert gdf.attrs["esri_wayback_release_date"] == RELEASE_DATE
    assert backend.release_tile_calls == gdf.attrs["n_aoi_tiles"]


def test_download_can_select_one_exact_esri_release(stub) -> None:
    backend = stub(ReleaseBackend(RELEASE_DATE, D1, colors={D1: 77}))
    ds = old_imagery.download(
        AOI,
        ZOOM,
        provider="esri",
        esri_wayback_release_date=RELEASE_DATE,
    )

    tags = ds.tags()
    assert tags["selection_mode"] == "esri-wayback-release"
    assert tags["esri_wayback_release_date"] == RELEASE_DATE.isoformat()
    assert tags["esri_wayback_release_title"] == backend.release.title
    assert tags["dates"] == D1.isoformat()
    assert "target_date" not in tags
    assert "date_match" not in tags
    assert ds.read(1).mean() == pytest.approx(77, abs=2)
    assert backend.dated_tile_calls == 0


def test_release_download_keeps_tiles_with_unknown_capture_date(stub) -> None:
    stub(ReleaseBackend(RELEASE_DATE, None))
    ds = old_imagery.download(
        AOI,
        ZOOM,
        provider="esri",
        esri_wayback_release_date=RELEASE_DATE,
    )
    tags = ds.tags()
    assert "dates" not in tags
    assert tags["tiles_capture_date_unknown"] == tags["tiles_total"]
    assert (ds.dataset_mask() > 0).all()


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (
            {"provider": "google", "esri_wayback_release_date": RELEASE_DATE},
            "requires provider='esri'",
        ),
        (
            {
                "provider": "esri",
                "esri_wayback_release_date": RELEASE_DATE,
                "min_date": D1,
            },
            "cannot be combined with min_date or max_date",
        ),
        (
            {
                "provider": "esri",
                "esri_wayback_release_date": RELEASE_DATE,
                "method": "per-tile",
            },
            "method cannot be set",
        ),
    ],
)
def test_release_availability_rejects_mixed_selection_modes(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        old_imagery.availability(AOI, ZOOM, **kwargs)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (
            {"provider": "google", "esri_wayback_release_date": RELEASE_DATE},
            "requires provider='esri'",
        ),
        (
            {
                "provider": "esri",
                "date": D1,
                "esri_wayback_release_date": RELEASE_DATE,
            },
            "Choose either date",
        ),
        (
            {
                "provider": "esri",
                "esri_wayback_release_date": RELEASE_DATE,
                "date_match": "exact",
            },
            "date_match cannot be set",
        ),
    ],
)
def test_release_download_rejects_mixed_selection_modes(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        old_imagery.download(AOI, ZOOM, **kwargs)


def test_region_query_used_above_the_tile_threshold(stub) -> None:
    backend = stub(RegionBackend([(D1, AOI)]))
    gdf = old_imagery.availability(AOI, REGION_ZOOM, provider="esri", max_tiles=10_000)

    assert gdf.attrs["n_aoi_tiles"] >= api.ESRI_REGION_QUERY_MIN_TILES
    assert gdf.attrs["method"] == "region-query"
    assert backend.region_calls == 1
    assert list(gdf["date"]) == [D1]


def test_per_tile_used_below_the_tile_threshold(stub) -> None:
    tiny = box(-122.3999, 37.7929, -122.3997, 37.7931)
    backend = stub(RegionBackend([(D1, tiny)]))
    gdf = old_imagery.availability(tiny, ZOOM, provider="esri")

    assert gdf.attrs["n_aoi_tiles"] < api.ESRI_REGION_QUERY_MIN_TILES
    assert gdf.attrs["method"] == "per-tile"
    assert backend.region_calls == 0


def test_method_region_forces_the_region_query_below_the_threshold(stub) -> None:
    tiny = box(-122.3999, 37.7929, -122.3997, 37.7931)
    backend = stub(RegionBackend([(D1, tiny)]))
    gdf = old_imagery.availability(tiny, ZOOM, provider="esri", method="region")

    assert gdf.attrs["n_aoi_tiles"] < api.ESRI_REGION_QUERY_MIN_TILES
    assert gdf.attrs["method"] == "region-query"
    assert backend.region_calls == 1


def test_method_per_tile_forces_probing_above_the_threshold(stub) -> None:
    backend = stub(RegionBackend([(D1, AOI)]))
    gdf = old_imagery.availability(
        AOI, REGION_ZOOM, provider="esri", method="per-tile", max_tiles=10_000
    )

    assert gdf.attrs["n_aoi_tiles"] >= api.ESRI_REGION_QUERY_MIN_TILES
    assert gdf.attrs["method"] == "per-tile"
    assert backend.region_calls == 0


def test_method_region_rejected_for_google(stub) -> None:
    stub(StubBackend([D1]))
    with pytest.raises(ValueError, match="only available for provider='esri'"):
        old_imagery.availability(AOI, ZOOM, method="region")


def test_unknown_method_rejected(stub) -> None:
    stub(StubBackend([D1]))
    with pytest.raises(ValueError, match="Unknown method"):
        old_imagery.availability(AOI, ZOOM, method="quick")


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
