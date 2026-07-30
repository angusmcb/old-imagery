"""Live integration tests. Run with ``pytest -m network``.

These hit Google's and Esri's public endpoints and are skipped by default.
"""

from __future__ import annotations

import datetime as dt

import pytest
from shapely.geometry import box

import old_imagery
from old_imagery import api

pytestmark = pytest.mark.network

# San Francisco Ferry Building and the Embarcadero: dense, long-running
# historical coverage from both providers.
AOI = box(-122.3965, 37.7940, -122.3910, 37.7970)


@pytest.fixture(scope="module")
def google_availability():
    return old_imagery.availability(AOI, zoom=17)


def test_google_availability_spans_decades(google_availability) -> None:
    gdf = google_availability
    assert len(gdf) > 20
    assert gdf.crs == "EPSG:4326"
    assert gdf["date"].iloc[0] > gdf["date"].iloc[-1]  # newest first
    assert gdf["date"].iloc[-1].year < 1970  # historical imagery reaches back
    assert (gdf["coverage"] > 0).all()
    assert gdf.geometry.is_valid.all()


def test_google_availability_respects_date_bounds() -> None:
    gdf = old_imagery.availability(AOI, zoom=17, min_date="1990-01-01", max_date="2000-01-01")
    assert len(gdf) > 0
    assert gdf["date"].min() >= dt.date(1990, 1, 1)
    assert gdf["date"].max() <= dt.date(2000, 1, 1)


def test_google_download_matches_an_available_date(google_availability) -> None:
    target = google_availability["date"].iloc[-1]  # oldest
    ds = old_imagery.download(AOI, zoom=18, date=target)

    assert ds.count == 3
    assert ds.crs == "EPSG:4326"
    assert ds.tags()["dates"] == target.isoformat()
    assert ds.tags()["tiles_missing"] == "0"
    assert (ds.dataset_mask() > 0).all()
    assert ds.read().std() > 5  # real imagery, not a flat fill


def test_google_download_bounds_match_the_aoi() -> None:
    zoom = 18
    ds = old_imagery.download(AOI, zoom=zoom, date="2020-01-01")
    tolerance = 360.0 / (256 * 2**zoom)
    for got, want in zip(ds.bounds, AOI.bounds, strict=True):
        assert got == pytest.approx(want, abs=tolerance)


def test_google_exact_match_rejects_a_date_with_no_imagery() -> None:
    with pytest.raises(ValueError, match="No imagery found"):
        old_imagery.download(AOI, zoom=18, date="1900-01-01", date_match="exact")


def test_esri_availability_and_download() -> None:
    small = box(-122.3965, 37.7940, -122.3945, 37.7955)
    gdf = old_imagery.availability(small, zoom=17, provider="esri")
    assert len(gdf) > 3
    assert gdf.crs == "EPSG:4326"

    target = gdf["date"].iloc[-1]
    ds = old_imagery.download(small, zoom=17, date=target, provider="esri")
    assert ds.crs == "EPSG:3857"  # Esri's native grid
    assert ds.count == 3
    assert (ds.dataset_mask() > 0).all()


def test_esri_footprints_cover_what_tile_probing_would_have_found() -> None:
    """Footprints replaced per-tile probing, so they must not lose capture dates."""
    from old_imagery._esri import WayBack
    from old_imagery._http import CachedHttpClient
    from old_imagery._region import MercatorGrid

    small = box(-122.3965, 37.7940, -122.3945, 37.7955)
    region = old_imagery.availability(small, zoom=17, provider="esri")
    assert region.attrs["method"] == "region-query"

    # What the removed per-tile path would have reported, computed directly.
    with CachedHttpClient() as client:
        wb = WayBack(client)
        probed = {
            d.date
            for tile in MercatorGrid().tiles(small, 17, 10_000)
            for d in wb.dated_tiles(tile)
            if d.date is not None
        }

    # Footprints can additionally see imagery that only clips the AOI edge, so
    # require a superset rather than equality.
    assert probed <= set(region["date"])
    assert region.geometry.is_valid.all()
    assert region.geometry.apply(lambda g: g.within(small.buffer(1e-9))).all()


def test_esri_region_query_dates_are_capture_not_release() -> None:
    """Capture dates must not coincide with the Wayback release dates."""
    from old_imagery._esri import WayBack
    from old_imagery._http import CachedHttpClient

    small = box(-122.3965, 37.7940, -122.3945, 37.7955)
    with CachedHttpClient() as client:
        release_dates = {layer.date for layer in WayBack(client).layers}

    gdf = old_imagery.availability(small, zoom=17, provider="esri")
    capture_dates = set(gdf["date"])
    assert capture_dates
    # Some capture dates could coincide by chance; most must not.
    assert len(capture_dates - release_dates) > len(capture_dates) / 2


def test_esri_mosaic_as_of_maps_real_footprints() -> None:
    """The seam map must come from footprints and agree with the pixels."""
    # Big enough to straddle a capture seam, so the multi-row case is exercised.
    area = box(-122.52, 37.70, -122.15, 37.90)
    gdf = old_imagery.esri_mosaic_as_of(area, 16, "2020-06-01")

    assert len(gdf) > 1, "this AOI should span several capture dates"
    assert list(gdf.columns) == api.ESRI_MOSAIC_COLUMNS
    assert gdf.crs == "EPSG:4326"
    assert gdf.attrs["release_date"] <= dt.date(2020, 6, 1)
    assert (gdf["release_id"] == gdf.attrs["release_id"]).all()

    # Footprints, not tile unions: a tile-quantised answer at zoom 16 could not
    # sum this close to 1 while still resolving several distinct dates.
    assert gdf["area_fraction"].sum() == pytest.approx(1.0, abs=0.05)
    assert gdf.geometry.within(area.buffer(1e-9)).all()

    # The metadata layer is a flat partition: rows must not claim the same
    # ground. Boundary slivers from geometryPrecision rounding are expected.
    geoms = list(gdf.geometry)
    for i in range(len(geoms)):
        for j in range(i + 1, len(geoms)):
            assert geoms[i].intersection(geoms[j]).area < area.area * 1e-6


def test_esri_mosaic_as_of_resolves_a_different_date_per_zoom() -> None:
    """Wayback composes per scale, so zoom is a real axis of the answer."""
    small = box(-122.404, 37.792, -122.396, 37.798)
    gdf = old_imagery.esri_mosaic_as_of(small, [13, 19], "2020-06-01")

    assert sorted(gdf["zoom"].unique()) == [13, 19]
    by_zoom = {z: set(g["date"]) for z, g in gdf.groupby("zoom")}
    assert by_zoom[13] != by_zoom[19], (
        "this AOI is known to show different capture dates at zoom 13 and 19; "
        "if Esri has recomposed the release, pick another area rather than "
        "weakening the assertion"
    )


def test_esri_mosaic_as_of_agrees_with_the_downloaded_pixels() -> None:
    small = box(-122.404, 37.792, -122.396, 37.798)
    gdf = old_imagery.esri_mosaic_as_of(small, 18, "2020-06-01")
    mapped = set(gdf["date"])

    with old_imagery.download(
        small, zoom=18, provider="esri", esri_wayback_release_id=gdf.attrs["release_id"]
    ) as src:
        tags = src.tags()
    pixel_dates = {dt.date.fromisoformat(d) for d in tags["dates"].split(",") if d}

    assert tags["esri_wayback_release_id"] == gdf.attrs["release_id"]
    assert pixel_dates <= mapped


def test_esri_wayback_release_selection() -> None:
    """The stable identifier targets and clearly labels one exact release."""
    from old_imagery._esri import WayBack
    from old_imagery._http import CachedHttpClient

    small = box(-122.3965, 37.7940, -122.3945, 37.7955)
    with CachedHttpClient() as client:
        wayback = WayBack(client)
        release = wayback.layers[0]
        assert wayback.release_by_identifier(release.identifier) is release
        assert wayback.release_on_or_before(release.date) is release

    ds = old_imagery.download(
        small,
        zoom=17,
        provider="esri",
        esri_wayback_release_id=release.identifier,
    )
    tags = ds.tags()
    assert tags["selection_mode"] == "esri-wayback-release"
    assert tags["esri_wayback_release_id"] == release.identifier
    assert tags["esri_wayback_catalogue_date"] == release.date.isoformat()
    assert tags["esri_wayback_release_title"] == release.title
    assert (ds.dataset_mask() > 0).all()


def test_release_download_beats_capture_date_across_a_seam() -> None:
    """A release is a mosaic of many dates, so no single date= reproduces it."""
    # Straddles a real seam between two capture dates in one release.
    seam = box(-122.3484, 37.89444, -122.3460, 37.89684)
    gdf = old_imagery.esri_mosaic_as_of(seam, 18, "2020-06-01")
    assert len(gdf) > 1, "this AOI should straddle a capture seam"

    with old_imagery.download(
        seam, 18, provider="esri", esri_wayback_release_id=gdf.attrs["release_id"]
    ) as src:
        release_tags = src.tags()
    assert release_tags["tiles_missing"] == "0"

    # The same area asked for by capture date loses everything flown otherwise.
    with old_imagery.download(
        seam, 18, provider="esri", date=gdf["date"].iloc[0], date_match="exact"
    ) as src:
        capture_tags = src.tags()
    assert int(capture_tags["tiles_missing"]) > 0
    assert capture_tags["tiles_total"] == release_tags["tiles_total"]


def test_both_providers_agree_on_extent() -> None:
    """Independent grids must georeference the same AOI to the same place."""
    from rasterio.warp import transform_bounds

    google = old_imagery.download(AOI, zoom=17, date="2016-06-01")
    esri = old_imagery.download(AOI, zoom=17, date="2016-06-01", provider="esri")
    esri_4326 = transform_bounds(esri.crs, "EPSG:4326", *esri.bounds)

    for a, b in zip(google.bounds, esri_4326, strict=True):
        assert a == pytest.approx(b, abs=1e-4)


def test_region_narrowing_gives_the_same_answer_as_querying_every_release() -> None:
    """Narrowing must be a pure speedup: identical dates, identical geometry."""
    from old_imagery import _esri

    small = box(-122.404, 37.792, -122.398, 37.7968)
    original = _esri.NARROW_RELEASES_MAX_TILES
    try:
        _esri.NARROW_RELEASES_MAX_TILES = 0  # disable
        full = old_imagery.availability(small, 17, provider="esri")
        _esri.NARROW_RELEASES_MAX_TILES = original
        narrowed = old_imagery.availability(small, 17, provider="esri")
    finally:
        _esri.NARROW_RELEASES_MAX_TILES = original

    assert len(narrowed) == len(full)
    assert list(narrowed["date"]) == list(full["date"])
    for a, b in zip(narrowed.geometry, full.geometry, strict=True):
        assert a.symmetric_difference(b).area < small.area * 1e-9


def test_candidate_releases_finds_the_same_dates_as_the_whole_catalogue() -> None:
    """The property the narrowing rests on, checked against the live service."""
    from old_imagery._esri import WayBack, _envelope_3857
    from old_imagery._http import CachedHttpClient
    from old_imagery._region import MercatorGrid

    small = box(-122.404, 37.792, -122.398, 37.7968)
    tiles = MercatorGrid().tiles(small, 17, 10_000)
    with CachedHttpClient() as client:
        wb = WayBack(client)
        env = _envelope_3857(small)
        candidates = wb.candidate_releases(tiles)
        assert 0 < len(candidates) < len(wb.layers)

        def dates(layers):
            return {d for layer in layers for d, _oid in wb._query_layer(layer, env, 17)[0]}

        assert dates(candidates) == dates(wb.layers)
