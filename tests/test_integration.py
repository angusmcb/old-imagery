"""Live integration tests. Run with ``pytest -m network``.

These hit Google's and Esri's public endpoints and are skipped by default.
"""

from __future__ import annotations

import datetime as dt

import pytest
from shapely.geometry import box

import old_imagery

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


def test_esri_region_query_agrees_with_per_tile() -> None:
    """The two Esri availability paths must report the same capture dates."""
    small = box(-122.3965, 37.7940, -122.3945, 37.7955)
    per_tile = old_imagery.availability(small, zoom=17, provider="esri", method="per-tile")
    region = old_imagery.availability(small, zoom=17, provider="esri", method="region")

    assert per_tile.attrs["method"] == "per-tile"
    assert region.attrs["method"] == "region-query"
    # The region path can additionally see footprints that only clip the AOI
    # edge, so require it to be a superset rather than an exact match.
    assert set(per_tile["date"]) <= set(region["date"])
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


def test_esri_exact_wayback_release_selection() -> None:
    """Release mode must target one exact published snapshot and label it clearly."""
    from old_imagery._esri import WayBack
    from old_imagery._http import CachedHttpClient

    small = box(-122.3965, 37.7940, -122.3945, 37.7955)
    with CachedHttpClient() as client:
        release = WayBack(client).layers[0]

    gdf = old_imagery.availability(
        small,
        zoom=17,
        provider="esri",
        esri_wayback_release_date=release.date,
    )
    assert gdf.attrs["selection_mode"] == "esri-wayback-release"
    assert gdf.attrs["esri_wayback_release_date"] == release.date
    if len(gdf):
        assert (gdf["providers"] == release.title).all()

    ds = old_imagery.download(
        small,
        zoom=17,
        provider="esri",
        esri_wayback_release_date=release.date,
    )
    tags = ds.tags()
    assert tags["selection_mode"] == "esri-wayback-release"
    assert tags["esri_wayback_release_date"] == release.date.isoformat()
    assert tags["esri_wayback_release_title"] == release.title
    assert (ds.dataset_mask() > 0).all()


def test_both_providers_agree_on_extent() -> None:
    """Independent grids must georeference the same AOI to the same place."""
    from rasterio.warp import transform_bounds

    google = old_imagery.download(AOI, zoom=17, date="2016-06-01")
    esri = old_imagery.download(AOI, zoom=17, date="2016-06-01", provider="esri")
    esri_4326 = transform_bounds(esri.crs, "EPSG:4326", *esri.bounds)

    for a, b in zip(google.bounds, esri_4326, strict=True):
        assert a == pytest.approx(b, abs=1e-4)
