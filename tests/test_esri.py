"""Offline tests for the Esri Wayback client, against a recorded fixture."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import replace
from pathlib import Path

import pytest
from shapely.geometry import box

from old_imagery import RequestFailed
from old_imagery._esri import (
    _METADATA_MAX_AGE,
    _TILEMAP_MAX_AGE,
    WayBack,
    _parse_capabilities,
)
from old_imagery._region import MercatorTile

SAMPLE = (Path(__file__).parent / "data" / "wayback_capabilities_sample.xml").read_bytes()
TILE = MercatorTile(50651, 20972, 17)


@pytest.fixture
def layers():
    return _parse_capabilities(SAMPLE)


def test_parses_layers_from_capabilities(layers) -> None:
    assert layers
    for layer in layers:
        assert isinstance(layer.date, dt.date)
        assert layer.identifier.startswith("WB")
        assert layer.id > 0
        assert "{TileMatrix}" in layer.resource_url


def test_document_order_is_preserved(layers) -> None:
    """dated_tiles' skip-ahead depends on document order, not date order."""
    from_xml = SAMPLE.decode().count("<Layer>")
    assert len(layers) == from_xml
    assert layers[0].date >= layers[-1].date  # Esri publishes newest first


def test_asset_url_substitutes_tile_coordinates(layers) -> None:
    url = layers[0].asset_url(TILE)
    assert "{" not in url
    assert url.endswith(f"/{TILE.level}/{TILE.row}/{TILE.column}")


def test_tilemap_url_shape(layers) -> None:
    url = layers[0].tilemap_url(TILE)
    assert "/MapServer/tilemap/" in url
    assert url.endswith(f"/{layers[0].id}/{TILE.level}/{TILE.row}/{TILE.column}")


def test_metadata_query_url_shape(layers) -> None:
    layer = layers[0]
    url = layer.metadata_query_url(17)
    assert url.startswith("https://metadata.")
    assert "_Metadata" in url
    assert url.endswith("/MapServer/6/query")  # min(13, 23 - 17) == 6


def test_metadata_query_scale_is_capped(layers) -> None:
    assert layers[0].metadata_query_url(5).endswith("/MapServer/13/query")
    assert layers[0].metadata_query_url(23).endswith("/MapServer/0/query")


def test_capabilities_without_contents_is_rejected() -> None:
    with pytest.raises(ValueError, match="Contents"):
        _parse_capabilities(b'<Capabilities xmlns="https://www.opengis.net/wmts/1.0"/>')


def test_http_namespace_is_also_accepted() -> None:
    """Esri serves https:// OWS namespaces; accept the conventional http:// too."""
    swapped = SAMPLE.replace(b"https://www.opengis.net/ows", b"http://www.opengis.net/ows")
    assert len(_parse_capabilities(swapped)) == len(_parse_capabilities(SAMPLE))


# --------------------------------------------------------------------------
# dated_tiles skip-ahead logic, driven by canned tilemap responses
# --------------------------------------------------------------------------
class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.requested = []
        self.requested_ages = []

    def get(self, url, *, max_age=None):
        self.requested.append(url)
        self.requested_ages.append(max_age)
        if url.endswith("wmtscapabilities.xml"):
            return SAMPLE
        for fragment, payload in self.responses.items():
            if fragment in url:
                return json.dumps(payload).encode()
        return json.dumps({"data": [0]}).encode()


def _wayback(responses):
    client = FakeClient(responses)
    wb = WayBack.__new__(WayBack)
    wb._client = client
    wb.layers = _parse_capabilities(SAMPLE)
    wb._by_id = {layer.id: layer for layer in wb.layers}
    wb._metadata_cache = {}
    import threading

    wb._lock = threading.Lock()
    return wb, client


def test_dated_tiles_empty_when_no_layer_has_data() -> None:
    wb, _ = _wayback({})
    assert wb.dated_tiles(TILE) == []


def test_tilemap_requests_use_a_long_refresh_window() -> None:
    wb, client = _wayback({})
    wb.dated_tiles(TILE)
    assert client.requested_ages
    assert all(age == _TILEMAP_MAX_AGE for age in client.requested_ages)


def test_source_metadata_requests_use_a_daily_refresh_window() -> None:
    wb, client = _wayback({})
    wb._tile_metadata(wb.layers[0], TILE)
    assert client.requested_ages == [_METADATA_MAX_AGE]


def test_dated_tiles_omits_imagery_without_capture_metadata() -> None:
    wb, _ = _wayback({"/tilemap/": {"data": [1]}})
    assert wb.dated_tiles(TILE) == []


def test_missing_capture_metadata_is_cached() -> None:
    wb, client = _wayback({"/tilemap/": {"data": [1]}})
    layer = wb.layers[0]

    assert wb._capture_date(layer, TILE) is None
    first_query_count = sum("/query" in url for url in client.requested)
    assert wb._capture_date(layer, TILE) is None
    assert sum("/query" in url for url in client.requested) == first_query_count


def test_release_identifier_survives_a_catalogue_date_disagreement() -> None:
    """WB_2026_R03 is stable even when Esri's two sources disagree by a day."""
    wb, _ = _wayback({})
    release = replace(
        wb.layers[0],
        identifier="WB_2026_R03",
        title="World Imagery (Wayback 2026-03-25)",
        date=dt.date(2026, 3, 25),
    )
    wb.layers = [release]

    assert wb.release_by_identifier("WB_2026_R03") is release


def test_release_on_or_before_resolves_the_latest_visible_snapshot() -> None:
    wb, _ = _wayback({})
    older = replace(wb.layers[1], date=dt.date(2026, 2, 26))
    release = replace(
        wb.layers[0],
        identifier="WB_2026_R03",
        title="World Imagery (Wayback 2026-03-25)",
        date=dt.date(2026, 3, 25),
    )
    wb.layers = [release, older]

    assert wb.release_on_or_before(dt.date(2026, 3, 26)) is release
    assert wb.release_on_or_before(dt.date(2026, 3, 1)) is older
    with pytest.raises(ValueError, match="earliest catalogue release"):
        wb.release_on_or_before(dt.date(2026, 1, 1))


def test_unknown_release_identifier_is_rejected() -> None:
    wb, _ = _wayback({})
    with pytest.raises(ValueError, match="No Esri Wayback release has identifier"):
        wb.release_by_identifier("WB_2099_R99")


def test_tile_at_release_keeps_the_requested_layer_and_capture_date() -> None:
    captured = dt.datetime(2011, 3, 4, tzinfo=dt.timezone.utc)
    millis = int(captured.timestamp() * 1000)
    wb, _ = _wayback(
        {
            "/query": {
                "features": [
                    {
                        "attributes": {
                            "SRC_DATE2": millis,
                            "SRC_RES": 0.5,
                            "SRC_ACC": 3,
                            "NICE_NAME": "Imagery Co",
                            "NICE_DESC": "Aerial survey",
                            "MinMapLevel": 15,
                            "MaxMapLevel": 19,
                        }
                    }
                ]
            }
        }
    )
    release = wb.layers[1]

    result = wb.tile_at_release(TILE, release)
    assert result.layer is release
    assert result.provider == release.id
    assert result.date == dt.date(2011, 3, 4)
    assert result.source is not None
    assert result.source.provider == "Imagery Co"
    assert result.source.resolution_m == 0.5
    assert result.source.accuracy_m == 3.0
    assert result.source.max_map_level == 19
    assert f"/tile/{release.id}/" in result.asset_url


def test_dated_tiles_uses_metadata_capture_date() -> None:
    captured = dt.datetime(2011, 3, 4, tzinfo=dt.timezone.utc)
    millis = int(captured.timestamp() * 1000)
    wb, _ = _wayback(
        {
            "/tilemap/": {"data": [1]},
            "/query": {"features": [{"attributes": {"SRC_DATE2": millis}}]},
        }
    )
    results = wb.dated_tiles(TILE)
    assert results
    assert results[-1].date == dt.date(2011, 3, 4)


def test_dated_tiles_collapses_identical_dates() -> None:
    """Releases that repeat the same imagery must yield one entry."""
    millis = int(dt.datetime(2011, 3, 4, tzinfo=dt.timezone.utc).timestamp() * 1000)
    wb, _ = _wayback(
        {
            "/tilemap/": {"data": [1]},
            "/query": {"features": [{"attributes": {"SRC_DATE2": millis}}]},
        }
    )
    assert len(wb.dated_tiles(TILE)) == 1


# --------------------------------------------------------------------------
# dated_regions: two-phase region query
# --------------------------------------------------------------------------
from old_imagery._esri import Layer  # noqa: E402

AOI = box(-122.400, 37.792, -122.396, 37.795)


def _layer(layer_id: int, release: str) -> Layer:
    return Layer(
        id=layer_id,
        title=f"World Imagery (Wayback {release})",
        identifier=f"WB_{release[:4]}_R{layer_id:02d}",
        date=dt.date.fromisoformat(release),
        resource_url=(
            "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/"
            f"WMTS/1.0.0/{{TileMatrixSet}}/MapServer/tile/{layer_id}/"
            "{TileMatrix}/{TileRow}/{TileCol}"
        ),
        matrix_set="default028mm",
    )


def _millis(date: dt.date) -> int:
    return int(
        dt.datetime(date.year, date.month, date.day, tzinfo=dt.timezone.utc).timestamp() * 1000
    )


def _esrijson(pairs: list[tuple[int, dt.date]]) -> bytes:
    """A minimal ESRIJSON feature set that GDAL can read.

    Takes several ``(oid, date)`` pairs because geometry is fetched in batches:
    one request carries many OBJECTIDs and the response carries a feature each.
    """
    return json.dumps(
        {
            "geometryType": "esriGeometryPolygon",
            "spatialReference": {"wkid": 4326},
            "fields": [
                {"name": "OBJECTID", "type": "esriFieldTypeOID", "alias": "OBJECTID"},
                {"name": "SRC_DATE2", "type": "esriFieldTypeDate", "alias": "SRC_DATE2"},
                {"name": "SRC_RES", "type": "esriFieldTypeDouble", "alias": "SRC_RES"},
                {"name": "SRC_ACC", "type": "esriFieldTypeDouble", "alias": "SRC_ACC"},
                {
                    "name": "NICE_NAME",
                    "type": "esriFieldTypeString",
                    "alias": "NICE_NAME",
                    "length": 50,
                },
                {
                    "name": "NICE_DESC",
                    "type": "esriFieldTypeString",
                    "alias": "NICE_DESC",
                    "length": 50,
                },
                {
                    "name": "MinMapLevel",
                    "type": "esriFieldTypeSmallInteger",
                    "alias": "MinMapLevel",
                },
                {
                    "name": "MaxMapLevel",
                    "type": "esriFieldTypeSmallInteger",
                    "alias": "MaxMapLevel",
                },
            ],
            "features": [
                {
                    "attributes": {
                        "OBJECTID": oid,
                        "SRC_DATE2": _millis(date),
                        "SRC_RES": 0.3,
                        "SRC_ACC": 5.0,
                        "NICE_NAME": "Test Provider",
                        "NICE_DESC": "Test Source",
                        "MinMapLevel": 16,
                        "MaxMapLevel": 20,
                    },
                    "geometry": {
                        "rings": [
                            [
                                [-122.400, 37.792],
                                [-122.400, 37.795],
                                [-122.396, 37.795],
                                [-122.396, 37.792],
                                [-122.400, 37.792],
                            ]
                        ]
                    },
                }
                for oid, date in pairs
            ],
        }
    ).encode()


class RegionClient:
    """Serves phase-1 (date only) and phase-2 (geometry) responses."""

    def __init__(self, per_layer):
        self.per_layer = per_layer  # {layer_id: [(date, oid), ...]}
        self.date_queries: list[int] = []
        # Every (layer, oid) pair asked for, regardless of how they were grouped.
        self.geometry_queries: list[tuple[int, int]] = []
        # One entry per HTTP request, so batching itself can be asserted on.
        self.geometry_requests: list[tuple[int, tuple[int, ...]]] = []
        self.post_ages: list[float | None] = []

    @staticmethod
    def _layer_id_from(url: str) -> int:
        marker = "_Metadata_"
        token = url[url.index(marker) + len(marker) :].split("/")[0]
        return int(token.split("_r")[-1])

    def post(self, url, data, *, max_age=None):
        self.post_ages.append(max_age)
        layer_id = self._layer_id_from(url)
        if data.get("returnGeometry") == "true":
            oids = [int(token) for token in data["objectIds"].split(",")]
            self.geometry_requests.append((layer_id, tuple(oids)))
            pairs = []
            for oid in oids:
                self.geometry_queries.append((layer_id, oid))
                pairs.append((oid, next(d for d, o in self.per_layer[layer_id] if o == oid)))
            return _esrijson(pairs)
        self.date_queries.append(layer_id)
        rows = self.per_layer.get(layer_id, [])
        return json.dumps(
            {
                "features": [
                    {"attributes": {"OBJECTID": o, "SRC_DATE2": _millis(d)}} for d, o in rows
                ]
            }
        ).encode()

    def get(self, url, *, max_age=None):  # pragma: no cover - unused here
        return SAMPLE


def _wayback_with(layers, per_layer):
    import threading

    client = RegionClient(per_layer)
    wb = WayBack.__new__(WayBack)
    wb._client = client
    wb.layers = layers
    wb._by_id = {layer.id: layer for layer in layers}
    wb._metadata_cache = {}
    wb._lock = threading.Lock()
    return wb, client


CAPTURE = dt.date(2010, 10, 26)


def test_dated_regions_fetches_geometry_once_per_date() -> None:
    """Every release repeats the same imagery; only one footprint is fetched."""
    layers = [_layer(1, "2014-02-20"), _layer(2, "2018-04-11"), _layer(3, "2022-06-08")]
    per_layer = {1: [(CAPTURE, 11)], 2: [(CAPTURE, 22)], 3: [(CAPTURE, 33)]}
    wb, client = _wayback_with(layers, per_layer)

    regions = wb.dated_regions(AOI, 17)

    assert len(client.date_queries) == 3  # phase 1 hits every release
    assert client.geometry_queries == [(1, 11)]  # phase 2 hits the earliest only
    assert len(regions) == 1
    footprint = regions[0]
    assert footprint.date == CAPTURE
    assert "2014-02-20" in footprint.release_title
    assert not footprint.geometry.is_empty


def test_dated_regions_keeps_disjoint_footprints_sharing_a_date() -> None:
    """Two footprints with the same date in one release must both survive."""
    layers = [_layer(1, "2014-02-20"), _layer(2, "2018-04-11")]
    per_layer = {1: [(CAPTURE, 11), (CAPTURE, 12)], 2: [(CAPTURE, 22)]}
    wb, client = _wayback_with(layers, per_layer)

    regions = wb.dated_regions(AOI, 17)

    assert sorted(client.geometry_queries) == [(1, 11), (1, 12)]
    assert len(regions) == 2


def test_dated_regions_never_compares_objectids_across_releases() -> None:
    """Identical OBJECTIDs in different releases are unrelated features."""
    layers = [_layer(1, "2014-02-20"), _layer(2, "2018-04-11")]
    early, late = dt.date(2010, 1, 1), dt.date(2016, 1, 1)
    per_layer = {1: [(early, 7)], 2: [(late, 7)]}  # same OBJECTID, different imagery
    wb, client = _wayback_with(layers, per_layer)

    regions = wb.dated_regions(AOI, 17)

    assert sorted(client.geometry_queries) == [(1, 7), (2, 7)]
    assert {footprint.date for footprint in regions} == {early, late}


def test_dated_regions_filters_by_date_bounds() -> None:
    layers = [_layer(1, "2014-02-20"), _layer(2, "2018-04-11")]
    early, late = dt.date(2010, 1, 1), dt.date(2016, 1, 1)
    per_layer = {1: [(early, 11)], 2: [(late, 22)]}
    wb, _ = _wayback_with(layers, per_layer)

    only_late = wb.dated_regions(AOI, 17, min_date=dt.date(2015, 1, 1))
    assert {footprint.date for footprint in only_late} == {late}

    only_early = wb.dated_regions(AOI, 17, max_date=dt.date(2015, 1, 1))
    assert {footprint.date for footprint in only_early} == {early}


def test_dated_regions_skips_releases_published_before_min_date() -> None:
    """A release published before min_date cannot hold newer imagery."""
    layers = [_layer(1, "2014-02-20"), _layer(2, "2022-06-08")]
    per_layer = {1: [(dt.date(2010, 1, 1), 11)], 2: [(dt.date(2021, 1, 1), 22)]}
    wb, client = _wayback_with(layers, per_layer)

    wb.dated_regions(AOI, 17, min_date=dt.date(2020, 1, 1))
    assert client.date_queries == [2]  # the 2014 release was never queried


def test_dated_regions_empty_when_nothing_matches() -> None:
    layers = [_layer(1, "2014-02-20")]
    wb, _ = _wayback_with(layers, {1: []})
    assert wb.dated_regions(AOI, 17) == []


def test_provider_copyright_maps_layer_id_to_title() -> None:
    wb, _ = _wayback({})
    layer = wb.layers[0]
    assert wb.provider_copyright(layer.id) == layer.title
    assert wb.provider_copyright(-1) is None


# --------------------------------------------------------------------------
# batched geometry fetches
# --------------------------------------------------------------------------
def test_fetch_geometries_batches_ids_into_one_request() -> None:
    """Many OBJECTIDs must cost one request, not one request each."""
    layers = [_layer(1, "2014-02-20")]
    per_layer = {1: [(CAPTURE, oid) for oid in range(11, 21)]}
    wb, client = _wayback_with(layers, per_layer)

    rows = wb._fetch_geometries(layers[0], 17, list(range(11, 21)))

    assert len(rows) == 10
    assert len(client.geometry_requests) == 1
    assert client.geometry_requests[0] == (1, tuple(range(11, 21)))
    assert client.post_ages == [_METADATA_MAX_AGE]
    assert all(row.date == CAPTURE and not row.geometry.is_empty for row in rows)


def test_fetch_geometries_chunks_beyond_the_batch_size() -> None:
    from old_imagery import _esri

    layers = [_layer(1, "2014-02-20")]
    count = _esri._GEOMETRY_BATCH + 5
    oids = list(range(1, count + 1))
    wb, client = _wayback_with(layers, {1: [(CAPTURE, oid) for oid in oids]})

    rows = wb._fetch_geometries(layers[0], 17, oids)

    assert len(rows) == count
    assert len(client.geometry_requests) == 2
    assert len(client.geometry_requests[0][1]) == _esri._GEOMETRY_BATCH
    assert len(client.geometry_requests[1][1]) == 5


def test_fetch_geometries_drops_only_the_failing_batch() -> None:
    """One bad response costs its own footprints, not the whole call."""
    from old_imagery import _esri

    layers = [_layer(1, "2014-02-20")]
    oids = list(range(1, _esri._GEOMETRY_BATCH + 6))
    wb, client = _wayback_with(layers, {1: [(CAPTURE, oid) for oid in oids]})

    real_post = client.post
    calls: list[int] = []

    def flaky(url, data, *, max_age=None):
        if data.get("returnGeometry") == "true":
            calls.append(1)
            if len(calls) == 1:
                raise RequestFailed("first batch is broken")
        return real_post(url, data, max_age=max_age)

    client.post = flaky
    rows = wb._fetch_geometries(layers[0], 17, oids)
    assert len(rows) == 5  # the surviving second chunk


# --------------------------------------------------------------------------
# _query_layer completeness signal
# --------------------------------------------------------------------------
def test_query_layer_reports_complete_on_a_normal_result() -> None:
    layers = [_layer(1, "2014-02-20")]
    wb, client = _wayback_with(layers, {1: [(CAPTURE, 11)]})
    rows, complete = wb._query_layer(layers[0], {}, 17)
    assert rows == [(CAPTURE, 11)]
    assert complete is True
    assert client.post_ages == [_METADATA_MAX_AGE]


def test_query_layer_reports_complete_on_a_genuinely_empty_result() -> None:
    """No features and no error means the release publishes nothing here."""
    layers = [_layer(1, "2014-02-20")]
    wb, _ = _wayback_with(layers, {1: []})
    rows, complete = wb._query_layer(layers[0], {}, 17)
    assert rows == []
    assert complete is True


def test_query_layer_reports_incomplete_when_the_request_fails() -> None:
    layers = [_layer(1, "2014-02-20")]
    wb, client = _wayback_with(layers, {1: [(CAPTURE, 11)]})

    def broken(url, data, *, max_age=None):
        raise RequestFailed("service down")

    client.post = broken
    rows, complete = wb._query_layer(layers[0], {}, 17)
    assert rows == []
    assert complete is False


def test_query_layer_reports_incomplete_on_an_error_payload() -> None:
    layers = [_layer(1, "2014-02-20")]
    wb, client = _wayback_with(layers, {1: [(CAPTURE, 11)]})

    def errored(url, data, *, max_age=None):
        return json.dumps({"error": {"code": 500, "message": "boom"}}).encode()

    client.post = errored
    rows, complete = wb._query_layer(layers[0], {}, 17)
    assert rows == []
    assert complete is False


def test_query_layer_reports_incomplete_when_pagination_runs_away() -> None:
    """The _MAX_FEATURES guard truncates, and must say so."""
    from old_imagery import _esri

    layers = [_layer(1, "2014-02-20")]
    wb, client = _wayback_with(layers, {1: []})

    page = [{"attributes": {"OBJECTID": oid, "SRC_DATE2": _millis(CAPTURE)}} for oid in range(500)]

    def endless(url, data, *, max_age=None):
        return json.dumps({"features": page, "exceededTransferLimit": True}).encode()

    client.post = endless
    rows, complete = wb._query_layer(layers[0], {}, 17)
    assert complete is False
    assert len(rows) > _esri._MAX_FEATURES


# --------------------------------------------------------------------------
# release_footprints: the seam map of one exact release
# --------------------------------------------------------------------------
def test_release_footprints_returns_dated_footprints_for_one_release() -> None:
    layers = [_layer(1, "2014-02-20"), _layer(2, "2018-04-11")]
    per_layer = {1: [(CAPTURE, 11), (CAPTURE, 12)], 2: [(dt.date(2017, 5, 1), 22)]}
    wb, client = _wayback_with(layers, per_layer)

    rows = wb.release_footprints(layers[0], AOI, 17)

    assert len(rows) == 2
    assert {row.date for row in rows} == {CAPTURE}
    assert all(not row.geometry.is_empty for row in rows)
    assert {row.source.provider for row in rows} == {"Test Provider"}
    assert {row.source.description for row in rows} == {"Test Source"}
    assert {row.source.resolution_m for row in rows} == {0.3}
    assert {row.source.accuracy_m for row in rows} == {5.0}
    assert {row.source.min_map_level for row in rows} == {16}
    assert {row.source.max_map_level for row in rows} == {20}
    # Only the release we asked for was touched; this is not a catalogue search.
    assert client.date_queries == [1]
    assert [layer_id for layer_id, _oids in client.geometry_requests] == [1]


def test_release_footprints_asks_the_metadata_layer_for_the_given_zoom() -> None:
    """Wayback composes per scale, so zoom picks a different metadata layer."""
    layers = [_layer(1, "2014-02-20")]
    wb, client = _wayback_with(layers, {1: [(CAPTURE, 11)]})
    seen: list[str] = []
    real_post = client.post

    def recording(url, data, *, max_age=None):
        seen.append(url)
        return real_post(url, data, max_age=max_age)

    client.post = recording
    wb.release_footprints(layers[0], AOI, 19)
    assert all("/MapServer/4/query" in url for url in seen)  # min(13, 23 - 19)


def test_release_footprints_is_empty_when_the_release_publishes_nothing() -> None:
    layers = [_layer(1, "2014-02-20")]
    wb, client = _wayback_with(layers, {1: []})
    assert wb.release_footprints(layers[0], AOI, 17) == []
    assert client.geometry_requests == []


def test_release_footprints_refuses_a_truncated_feature_list() -> None:
    """A partial seam map would read as missing imagery, so it must raise."""
    layers = [_layer(1, "2014-02-20")]
    wb, client = _wayback_with(layers, {1: [(CAPTURE, 11)]})

    def broken(url, data, *, max_age=None):
        raise RequestFailed("metadata service down")

    client.post = broken
    with pytest.raises(RequestFailed, match="did not return a complete feature list"):
        wb.release_footprints(layers[0], AOI, 17)


def test_release_footprints_rejects_more_footprints_than_the_limit() -> None:
    layers = [_layer(1, "2014-02-20")]
    per_layer = {1: [(CAPTURE, oid) for oid in range(1, 12)]}
    wb, client = _wayback_with(layers, per_layer)

    with pytest.raises(ValueError, match="above the limit of 5"):
        wb.release_footprints(layers[0], AOI, 17, max_footprints=5)
    assert client.geometry_requests == []  # refused before fetching geometry


def test_release_footprints_deduplicates_repeated_object_ids() -> None:
    layers = [_layer(1, "2014-02-20")]
    wb, client = _wayback_with(layers, {1: [(CAPTURE, 11), (CAPTURE, 11)]})
    wb.release_footprints(layers[0], AOI, 17)
    assert client.geometry_requests == [(1, (11,))]


# --------------------------------------------------------------------------
# candidate_releases: narrowing the release list with cheap tilemap probes
# --------------------------------------------------------------------------
class ChainClient(RegionClient):
    """Adds tilemap responses so the select skip-ahead can be exercised.

    ``chain`` maps a layer id to the layer id its ``select`` points at; layers
    listed in ``carries`` report imagery (``data == [1]``).
    """

    def __init__(self, per_layer, chain=None, carries=()):
        super().__init__(per_layer)
        self.chain = chain or {}
        self.carries = set(carries)
        self.tilemap_queries: list[int] = []

    def get(self, url, *, max_age=None):
        if "/tilemap/" in url:
            layer_id = int(url.split("/tilemap/")[1].split("/")[0])
            self.tilemap_queries.append(layer_id)
            payload = {"data": [1 if layer_id in self.carries else 0]}
            if layer_id in self.chain:
                payload["select"] = [self.chain[layer_id]]
            return json.dumps(payload).encode()
        return SAMPLE


def _wayback_chain(layers, per_layer, chain=None, carries=()):
    import threading

    client = ChainClient(per_layer, chain, carries)
    wb = WayBack.__new__(WayBack)
    wb._client = client
    wb.layers = layers
    wb._by_id = {layer.id: layer for layer in layers}
    wb._metadata_cache = {}
    wb._lock = threading.Lock()
    return wb, client


def test_candidate_releases_follows_the_select_skip_ahead() -> None:
    """The point of the exercise: don't probe releases the service says are unchanged."""
    layers = [_layer(i, f"20{10 + i:02d}-01-01") for i in range(1, 7)]
    # Layer 1 says "nothing changed until layer 4", so 2 and 3 are never probed.
    wb, client = _wayback_chain(layers, {}, chain={1: 4}, carries={1, 4, 5, 6})

    found = wb.candidate_releases([TILE])

    assert client.tilemap_queries == [1, 4, 5, 6]  # 2 and 3 skipped
    assert {layer.id for layer in found} == {4, 5, 6}  # 1 resolves to its select target


def test_candidate_releases_ignores_releases_without_imagery() -> None:
    layers = [_layer(i, f"20{10 + i:02d}-01-01") for i in range(1, 4)]
    wb, _ = _wayback_chain(layers, {}, carries={2})
    assert {layer.id for layer in wb.candidate_releases([TILE])} == {2}


def test_candidate_releases_unions_over_tiles_and_keeps_document_order() -> None:
    layers = [_layer(i, f"20{10 + i:02d}-01-01") for i in range(1, 5)]
    wb, _ = _wayback_chain(layers, {}, carries={1, 2, 3, 4})
    other = MercatorTile(TILE.row + 1, TILE.column, TILE.level)

    found = wb.candidate_releases([TILE, other])
    assert [layer.id for layer in found] == [1, 2, 3, 4]  # document order, newest first


def test_candidate_releases_is_empty_when_no_release_carries_the_tile() -> None:
    layers = [_layer(i, f"20{10 + i:02d}-01-01") for i in range(1, 4)]
    wb, _ = _wayback_chain(layers, {}, carries=set())
    assert wb.candidate_releases([TILE]) == []


def test_dated_regions_narrows_the_release_list_when_given_tiles() -> None:
    """Tiles in hand, only the releases that changed this area are queried."""
    layers = [_layer(i, f"201{i}-01-01") for i in range(1, 5)]
    per_layer = {i: [(CAPTURE, 10 + i)] for i in range(1, 5)}
    wb, client = _wayback_chain(layers, per_layer, carries={2})

    wb.dated_regions(AOI, 17, tiles=[TILE])

    # Only release 2 reached the slow metadata service, not all four.
    assert client.date_queries == [2]


def test_dated_regions_queries_every_release_without_tiles() -> None:
    layers = [_layer(i, f"201{i}-01-01") for i in range(1, 5)]
    per_layer = {i: [(CAPTURE, 10 + i)] for i in range(1, 5)}
    wb, client = _wayback_chain(layers, per_layer, carries={2})

    wb.dated_regions(AOI, 17)
    assert sorted(client.date_queries) == [1, 2, 3, 4]


def test_dated_regions_skips_narrowing_on_a_large_tile_count() -> None:
    """Above the threshold, probing every tile costs more than it saves."""
    from old_imagery._esri import NARROW_RELEASES_MAX_TILES

    layers = [_layer(i, f"201{i}-01-01") for i in range(1, 5)]
    per_layer = {i: [(CAPTURE, 10 + i)] for i in range(1, 5)}
    wb, client = _wayback_chain(layers, per_layer, carries={2})

    many = [
        MercatorTile(TILE.row + i, TILE.column, TILE.level)
        for i in range(NARROW_RELEASES_MAX_TILES + 1)
    ]
    wb.dated_regions(AOI, 17, tiles=many)

    assert client.tilemap_queries == []  # never probed
    assert sorted(client.date_queries) == [1, 2, 3, 4]
