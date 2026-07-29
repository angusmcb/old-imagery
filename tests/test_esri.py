"""Offline tests for the Esri Wayback client, against a recorded fixture."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from shapely.geometry import box

from old_imagery._esri import WayBack, _parse_capabilities
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

    def get(self, url, *, max_age=None):
        self.requested.append(url)
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
    wb._date_cache = {}
    import threading

    wb._lock = threading.Lock()
    return wb, client


def test_dated_tiles_empty_when_no_layer_has_data() -> None:
    wb, _ = _wayback({})
    assert wb.dated_tiles(TILE) == []


def test_dated_tiles_falls_back_to_release_date_without_metadata() -> None:
    wb, _ = _wayback({"/tilemap/": {"data": [1]}})
    results = wb.dated_tiles(TILE)
    assert results
    # With no SRC_DATE2 available the layer's own release date is used.
    assert all(r.date is not None for r in results)


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
    return int(dt.datetime(date.year, date.month, date.day, tzinfo=dt.timezone.utc).timestamp() * 1000)


def _esrijson(oid: int, date: dt.date) -> bytes:
    """A minimal ESRIJSON feature set that GDAL can read."""
    return json.dumps(
        {
            "geometryType": "esriGeometryPolygon",
            "spatialReference": {"wkid": 4326},
            "fields": [
                {"name": "OBJECTID", "type": "esriFieldTypeOID", "alias": "OBJECTID"},
                {"name": "SRC_DATE2", "type": "esriFieldTypeDate", "alias": "SRC_DATE2"},
            ],
            "features": [
                {
                    "attributes": {"OBJECTID": oid, "SRC_DATE2": _millis(date)},
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
            ],
        }
    ).encode()


class RegionClient:
    """Serves phase-1 (date only) and phase-2 (geometry) responses."""

    def __init__(self, per_layer):
        self.per_layer = per_layer  # {layer_id: [(date, oid), ...]}
        self.date_queries: list[int] = []
        self.geometry_queries: list[tuple[int, int]] = []

    @staticmethod
    def _layer_id_from(url: str) -> int:
        marker = "_Metadata_"
        token = url[url.index(marker) + len(marker) :].split("/")[0]
        return int(token.split("_r")[-1])

    def post(self, url, data, *, max_age=None):
        layer_id = self._layer_id_from(url)
        if data.get("returnGeometry") == "true":
            oid = int(data["objectIds"])
            self.geometry_queries.append((layer_id, oid))
            date = next(d for d, o in self.per_layer[layer_id] if o == oid)
            return _esrijson(oid, date)
        self.date_queries.append(layer_id)
        rows = self.per_layer.get(layer_id, [])
        return json.dumps(
            {"features": [{"attributes": {"OBJECTID": o, "SRC_DATE2": _millis(d)}} for d, o in rows]}
        ).encode()

    def get(self, url, *, max_age=None):  # pragma: no cover - unused here
        return SAMPLE


def _wayback_with(layers, per_layer):
    import threading

    client = RegionClient(per_layer)
    wb = WayBack.__new__(WayBack)
    wb._client = client
    wb.layers = layers
    wb._by_id = {l.id: l for l in layers}
    wb._date_cache = {}
    wb._lock = threading.Lock()
    return wb, client


CAPTURE = dt.date(2010, 10, 26)


def test_dated_regions_fetches_geometry_once_per_date() -> None:
    """Every release repeats the same imagery; only one footprint is fetched."""
    layers = [_layer(1, "2014-02-20"), _layer(2, "2018-04-11"), _layer(3, "2022-06-08")]
    per_layer = {1: [(CAPTURE, 11)], 2: [(CAPTURE, 22)], 3: [(CAPTURE, 33)]}
    wb, client = _wayback_with(layers, per_layer)

    regions = wb.dated_regions(AOI, 17, max_workers=4)

    assert len(client.date_queries) == 3  # phase 1 hits every release
    assert client.geometry_queries == [(1, 11)]  # phase 2 hits the earliest only
    assert len(regions) == 1
    date, geom, title = regions[0]
    assert date == CAPTURE
    assert "2014-02-20" in title  # earliest release carrying this imagery
    assert not geom.is_empty


def test_dated_regions_keeps_disjoint_footprints_sharing_a_date() -> None:
    """Two footprints with the same date in one release must both survive."""
    layers = [_layer(1, "2014-02-20"), _layer(2, "2018-04-11")]
    per_layer = {1: [(CAPTURE, 11), (CAPTURE, 12)], 2: [(CAPTURE, 22)]}
    wb, client = _wayback_with(layers, per_layer)

    regions = wb.dated_regions(AOI, 17, max_workers=4)

    assert sorted(client.geometry_queries) == [(1, 11), (1, 12)]
    assert len(regions) == 2


def test_dated_regions_never_compares_objectids_across_releases() -> None:
    """Identical OBJECTIDs in different releases are unrelated features."""
    layers = [_layer(1, "2014-02-20"), _layer(2, "2018-04-11")]
    early, late = dt.date(2010, 1, 1), dt.date(2016, 1, 1)
    per_layer = {1: [(early, 7)], 2: [(late, 7)]}  # same OBJECTID, different imagery
    wb, client = _wayback_with(layers, per_layer)

    regions = wb.dated_regions(AOI, 17, max_workers=4)

    assert sorted(client.geometry_queries) == [(1, 7), (2, 7)]
    assert {date for date, _g, _t in regions} == {early, late}


def test_dated_regions_filters_by_date_bounds() -> None:
    layers = [_layer(1, "2014-02-20"), _layer(2, "2018-04-11")]
    early, late = dt.date(2010, 1, 1), dt.date(2016, 1, 1)
    per_layer = {1: [(early, 11)], 2: [(late, 22)]}
    wb, _ = _wayback_with(layers, per_layer)

    only_late = wb.dated_regions(AOI, 17, min_date=dt.date(2015, 1, 1), max_workers=4)
    assert {d for d, _g, _t in only_late} == {late}

    only_early = wb.dated_regions(AOI, 17, max_date=dt.date(2015, 1, 1), max_workers=4)
    assert {d for d, _g, _t in only_early} == {early}


def test_dated_regions_skips_releases_published_before_min_date() -> None:
    """A release published before min_date cannot hold newer imagery."""
    layers = [_layer(1, "2014-02-20"), _layer(2, "2022-06-08")]
    per_layer = {1: [(dt.date(2010, 1, 1), 11)], 2: [(dt.date(2021, 1, 1), 22)]}
    wb, client = _wayback_with(layers, per_layer)

    wb.dated_regions(AOI, 17, min_date=dt.date(2020, 1, 1), max_workers=4)
    assert client.date_queries == [2]  # the 2014 release was never queried


def test_dated_regions_empty_when_nothing_matches() -> None:
    layers = [_layer(1, "2014-02-20")]
    wb, _ = _wayback_with(layers, {1: []})
    assert wb.dated_regions(AOI, 17, max_workers=4) == []


def test_provider_copyright_maps_layer_id_to_title() -> None:
    wb, _ = _wayback({})
    layer = wb.layers[0]
    assert wb.provider_copyright(layer.id) == layer.title
    assert wb.provider_copyright(-1) is None
