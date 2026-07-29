"""Esri World Imagery Wayback client.

Ported from ``LibEsri`` in Mbucari/GEHistoricalImagery.
"""

from __future__ import annotations

import datetime as _dt
import io
import json
import re
import threading
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass

from ._http import CachedHttpClient, RequestFailed
from ._region import MercatorGrid, MercatorTile

WMTS_CAPABILITIES = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/world_imagery/"
    "mapserver/wmts/1.0.0/wmtscapabilities.xml"
)
_CAPS_MAX_AGE = 7 * 24 * 3600
# Esri serves these with an https scheme, which is unusual for XML namespaces;
# accept either so the parser does not hinge on that detail.
_OWS_NAMESPACES = ("https://www.opengis.net/ows/1.1", "http://www.opengis.net/ows/1.1")
_KEY_TEXT = "/World_Imagery"
_DATE_IN_TITLE = re.compile(r"\(Wayback (\d{4}-\d{2}-\d{2})\)")
# Guard against a runaway pagination loop on a very large area of interest.
_MAX_FEATURES = 20_000


@dataclass(frozen=True)
class Layer:
    """One Wayback release (a dated snapshot of the World Imagery basemap)."""

    id: int
    title: str
    identifier: str
    date: _dt.date
    resource_url: str
    matrix_set: str

    def asset_url(self, tile: MercatorTile) -> str:
        return (
            self.resource_url.replace("{TileMatrixSet}", self.matrix_set)
            .replace("{TileMatrix}", str(tile.level))
            .replace("{TileRow}", str(tile.row))
            .replace("{TileCol}", str(tile.column))
        )

    def tilemap_url(self, tile: MercatorTile) -> str:
        end = self.resource_url.index(_KEY_TEXT) + len(_KEY_TEXT)
        base = self.resource_url[:end] + "/MapServer/tilemap"
        return f"{base}/{self.id}/{tile.level}/{tile.row}/{tile.column}"

    def metadata_query_url(self, level: int) -> str:
        scale = min(13, 23 - level)
        start = self.resource_url.index("//") + 2
        dot = self.resource_url.index(".", start)
        domain = self.resource_url[:start] + "metadata" + self.resource_url[dot:]
        end = domain.index(_KEY_TEXT) + len(_KEY_TEXT)
        suffix = self.identifier.replace("WB", "").lower()
        return f"{domain[:end]}_Metadata{suffix}/MapServer/{scale}/query"


@dataclass(frozen=True)
class DatedEsriTile:
    """Interface-compatible with :class:`old_imagery._dbroot.DatedTile`."""

    tile: MercatorTile
    date: _dt.date | None
    provider: int
    epoch: int
    layer: Layer

    @property
    def asset_url(self) -> str:
        return self.layer.asset_url(self.tile)


def _find_ows(element, name):
    for namespace in _OWS_NAMESPACES:
        found = element.find(f"{{{namespace}}}{name}")
        if found is not None:
            return found
    return None


def _parse_capabilities(xml: bytes) -> list[Layer]:
    root = ET.fromstring(xml)
    ns = root.tag[1 : root.tag.index("}")] if root.tag.startswith("{") else ""
    q = (lambda name: f"{{{ns}}}{name}") if ns else (lambda name: name)

    contents = root.find(q("Contents"))
    if contents is None:
        raise ValueError("WMTS capabilities document has no Contents element")

    layers: list[Layer] = []
    for element in contents.findall(q("Layer")):
        title_el = _find_ows(element, "Title")
        identifier_el = _find_ows(element, "Identifier")
        resource_el = element.find(q("ResourceURL"))
        if title_el is None or identifier_el is None or resource_el is None:
            continue
        title = (title_el.text or "").strip()
        match = _DATE_IN_TITLE.search(title)
        if match is None:
            continue
        template = resource_el.get("template")
        if not template or _KEY_TEXT not in template:
            continue

        matrix_sets = [
            e.text
            for link in element.findall(q("TileMatrixSetLink"))
            for e in (link.find(q("TileMatrixSet")),)
            if e is not None and e.text
        ]
        try:
            start = template.index("/MapServer/tile/") + len("/MapServer/tile/")
            layer_id = int(template[start : template.index("/", start)])
        except ValueError:
            continue

        layers.append(
            Layer(
                id=layer_id,
                title=title,
                identifier=(identifier_el.text or "").strip(),
                date=_dt.date.fromisoformat(match.group(1)),
                resource_url=template,
                matrix_set=matrix_sets[0] if matrix_sets else "default028mm",
            )
        )

    # Deliberately left in document order (newest release first).  The tilemap
    # endpoint's "select" field names a release that appears *later* in this
    # order, so re-sorting would break the skip-ahead chain in dated_tiles().
    return layers


class WayBack:
    """Esri World Imagery Wayback archive.

    Exposes the same ``dated_tiles`` / ``download_tile_image`` surface as
    :class:`old_imagery._dbroot.DbRoot` so both providers share the mosaicking code.
    """

    grid = MercatorGrid()

    def __init__(self, client: CachedHttpClient):
        self._client = client
        self.layers = _parse_capabilities(client.get(WMTS_CAPABILITIES, max_age=_CAPS_MAX_AGE))
        self._by_id = {layer.id: layer for layer in self.layers}
        self._date_cache: dict[tuple[int, int, int, int], _dt.date | None] = {}
        self._lock = threading.Lock()

    # -- helpers -----------------------------------------------------------
    def _json(self, url: str) -> dict | None:
        try:
            return json.loads(self._client.get(url))
        except (RequestFailed, ValueError, UnicodeDecodeError):
            return None

    def _capture_date(self, layer: Layer, tile: MercatorTile) -> _dt.date | None:
        """The true capture date of ``tile`` in ``layer``, or ``None`` if unavailable."""
        key = (layer.id, tile.level, tile.row, tile.column)
        with self._lock:
            if key in self._date_cache:
                return self._date_cache[key]

        lon, lat = tile.center
        query = {
            "f": "json",
            "outFields": "SRC_DATE2",
            "spatialRel": "esriSpatialRelWithin",
            "geometryType": "esriGeometryPoint",
            "inSR": "4326",
            "geometry": json.dumps({"x": lon, "y": lat, "spatialReference": {"wkid": 4326}}),
            "returnGeometry": "false",
        }
        url = layer.metadata_query_url(tile.level) + "?" + _query_string(query)
        payload = self._json(url)
        # A release date is not an image capture date. If the metadata service
        # gives us nothing usable, omit this version rather than silently
        # changing the meaning of every date exposed by the public API.
        date = None
        if payload is not None:
            try:
                millis = payload["features"][0]["attributes"]["SRC_DATE2"]
                if millis is not None:
                    date = _dt.datetime.fromtimestamp(millis / 1000, _dt.timezone.utc).date()
            except (TypeError, KeyError, IndexError):
                pass

        with self._lock:
            self._date_cache[key] = date
        return date

    def release_by_identifier(self, identifier: str) -> Layer:
        """Return the exact release with stable WMTS ``identifier``."""
        for layer in self.layers:
            if layer.identifier == identifier:
                return layer
        raise ValueError(
            f"No Esri Wayback release has identifier {identifier!r}; "
            "expected a catalogue identifier such as 'WB_2026_R03'"
        )

    def release_on_or_before(self, visible_date: _dt.date) -> Layer:
        """Return the latest WMTS release dated on or before ``visible_date``."""
        matches = [layer for layer in self.layers if layer.date <= visible_date]
        if matches:
            return max(matches, key=lambda layer: layer.date)

        earliest = min((layer.date for layer in self.layers), default=None)
        detail = (
            f"; the earliest catalogue release is {earliest.isoformat()}"
            if earliest is not None
            else ""
        )
        raise ValueError(
            f"No Esri Wayback release was visible on or before {visible_date.isoformat()}{detail}"
        )

    def tile_at_release(self, tile: MercatorTile, layer: Layer) -> DatedEsriTile:
        """The tile served by one exact Wayback release snapshot.

        Unlike :meth:`dated_tiles`, this does not search or fall back across
        releases. The capture date may be unknown, but the requested layer is
        retained so downloading always targets that exact published snapshot.
        """
        return DatedEsriTile(
            tile=tile,
            date=self._capture_date(layer, tile),
            provider=layer.id,
            epoch=layer.id,
            layer=layer,
        )

    # -- provider interface ------------------------------------------------
    def dated_tiles(self, tile: MercatorTile) -> list[DatedEsriTile]:
        """Distinct imagery versions covering ``tile``, newest first.

        Wayback releases the whole basemap on each publication date, so most
        releases repeat the previous imagery for any given tile.  The tilemap
        endpoint reports which release actually changed a tile, and lets us
        skip runs of unchanged releases in one hop.
        """
        results: list[DatedEsriTile] = []
        last_layer: Layer | None = None
        last_date: _dt.date | None = None
        skip_until: int | None = None

        for layer in self.layers:
            if skip_until is not None:
                if skip_until == layer.id:
                    skip_until = None
                else:
                    continue

            payload = self._json(layer.tilemap_url(tile))
            effective = layer
            select = (payload or {}).get("select")
            if select:
                skip_until = int(select[0])
                effective = self._by_id.get(skip_until, layer)

            data = (payload or {}).get("data")
            if not data or data[0] != 1:
                continue

            date = self._capture_date(effective, tile)
            if date is None:
                continue
            if last_date is not None and last_layer is not None and last_date != date:
                # Emit only when the tile's imagery actually changed, so each
                # entry is the earliest release carrying that imagery.
                results.append(self._make(tile, last_date, last_layer))
            last_date, last_layer = date, effective

        if last_date is not None and last_layer is not None:
            results.append(self._make(tile, last_date, last_layer))
        return results

    @staticmethod
    def _make(tile: MercatorTile, date: _dt.date, layer: Layer) -> DatedEsriTile:
        return DatedEsriTile(tile=tile, date=date, provider=layer.id, epoch=layer.id, layer=layer)

    # -- region-wide availability -----------------------------------------
    def dated_regions(
        self,
        aoi,
        zoom: int,
        *,
        min_date: _dt.date | None = None,
        max_date: _dt.date | None = None,
        max_workers: int = 16,
    ) -> list[tuple[_dt.date, object, str]]:
        """Capture footprints intersecting ``aoi``.

        Returns ``(capture_date, geometry, release_title)`` triples, where the
        date is when the imagery was *captured* and the title names the Wayback
        release that first published it -- two different things.

        One query per Wayback release against the metadata feature service,
        rather than probing every release for every tile.  The cost therefore
        scales with the number of distinct capture footprints instead of the
        tile count, and the returned geometry is the true capture footprint
        rather than a tile-quantised approximation.

        ``aoi`` and the returned geometries are in EPSG:4326.
        """
        import concurrent.futures

        # A release published before min_date cannot contain imagery captured
        # after it, so those releases can be dropped outright.
        layers = [layer for layer in self.layers if min_date is None or layer.date >= min_date]
        # Oldest first, so the max_date short-circuit below can cut the tail.
        layers.sort(key=lambda layer: layer.date)
        if not layers:
            return []

        envelope = _envelope_3857(aoi)
        cancel_after: list[_dt.date | None] = [None]
        results: list[tuple[_dt.date, object, str]] = []
        lock = threading.Lock()

        # Phase 1: which capture dates does each release expose here? Cheap,
        # geometry-free, one request per release.
        wanted: list[tuple[Layer, _dt.date, int]] = []

        def query(layer: Layer):
            with lock:
                limit = cancel_after[0]
            if limit is not None and layer.date > limit:
                return  # an older release already proved the tail is empty

            found = self._query_layer(layer, envelope, zoom)
            if not found:
                return

            matched = []
            saw_later = False
            for date, object_id in found:
                if max_date is not None and date > max_date:
                    saw_later = True
                    continue
                if min_date is not None and date < min_date:
                    continue
                matched.append((layer, date, object_id))

            with lock:
                if matched:
                    wanted.extend(matched)
                elif saw_later:  # noqa: SIM102
                    # Nothing here was captured before max_date, so no later
                    # release will have anything either.  Left nested rather
                    # than collapsed: whether this is a saw_later case, and
                    # whether it is the earliest such date, are separate
                    # questions, and merging them buries this comment.
                    if cancel_after[0] is None or layer.date < cancel_after[0]:
                        cancel_after[0] = layer.date

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(query, layers))

        if not wanted:
            return []

        # Phase 2: fetch footprints once per capture date rather than once per
        # release. Nearly every release repeats the same imagery, so this
        # collapses ~195 candidates to a handful of downloads.
        #
        # OBJECTIDs are only unique within a single release's metadata service,
        # so they must never be compared across releases. Instead pick, for each
        # capture date, the earliest release reporting it, then fetch every
        # footprint *that* release listed -- which keeps disjoint footprints
        # sharing a date.
        earliest: dict[_dt.date, Layer] = {}
        for layer, date, _oid in wanted:
            current = earliest.get(date)
            if current is None or layer.date < current.date:
                earliest[date] = layer

        by_layer_date: dict[tuple[_dt.date, int], set[int]] = defaultdict(set)
        for layer, date, oid in wanted:
            by_layer_date[(date, layer.id)].add(oid)

        fetches = [
            (date, layer, oid)
            for date, layer in earliest.items()
            for oid in sorted(by_layer_date[(date, layer.id)])
        ]

        def fetch(item):
            date, layer, oid = item
            geom = self._fetch_geometry(layer, zoom, oid)
            return None if geom is None else (date, geom, layer.title)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for row in pool.map(fetch, fetches):
                if row is not None:
                    results.append(row)
        return results

    def _query_layer(self, layer: Layer, envelope: dict, zoom: int):
        """Return ``[(capture_date, object_id), ...]`` for one release.

        Deliberately requests no geometry.  Capture footprints are large -- one
        sampled polygon had 3,520 vertices -- and almost every release repeats
        the same footprint, so fetching geometry here would download the same
        megabytes ~195 times.  Geometry is fetched once per surviving feature
        in :meth:`_fetch_geometries` instead.
        """
        url = layer.metadata_query_url(zoom)
        offset = 0
        out: list[tuple[_dt.date, int]] = []

        while True:
            form = {
                "f": "json",
                "outFields": "OBJECTID,SRC_DATE2",
                "spatialRel": "esriSpatialRelIntersects",
                "geometryType": "esriGeometryEnvelope",
                "inSR": "3857",
                "geometry": json.dumps(envelope),
                "returnGeometry": "false",
            }
            if offset:
                form["resultOffset"] = str(offset)
            try:
                payload = json.loads(self._client.post(url, form))
            except (RequestFailed, OSError, ValueError):
                return out
            if "error" in payload or not payload.get("features"):
                return out

            for feature in payload["features"]:
                attributes = feature.get("attributes") or {}
                date = _coerce_date(attributes.get("SRC_DATE2"))
                oid = attributes.get("OBJECTID")
                if date is not None and oid is not None:
                    out.append((date, int(oid)))

            if not payload.get("exceededTransferLimit"):
                return out
            offset += len(payload["features"])
            if offset > _MAX_FEATURES:
                return out

    def _fetch_geometry(self, layer: Layer, zoom: int, object_id: int):
        """Fetch one capture footprint by OBJECTID, as an EPSG:4326 geometry."""
        import geopandas as gpd

        form = {
            "f": "json",
            "outFields": "OBJECTID,SRC_DATE2",
            "objectIds": str(object_id),
            "returnGeometry": "true",
            "geometryPrecision": "2",
            "outSR": "3857",
        }
        try:
            raw = self._client.post(layer.metadata_query_url(zoom), form)
            payload = json.loads(raw)
        except (RequestFailed, OSError, ValueError):
            return None
        if "error" in payload or not payload.get("features"):
            return None
        try:
            frame = gpd.read_file(io.BytesIO(raw))
        except Exception:  # noqa: BLE001
            # Deliberately broad: read_file dispatches to GDAL/pyogrio drivers
            # whose failure modes on unexpected bytes are not a stable, listable
            # set.  One unreadable release degrades to None rather than
            # aborting the whole availability call.
            return None
        rows = _rows_to_dated_geometries(frame)
        return rows[0][1] if rows else None

    def download_tile_image(self, dated: DatedEsriTile) -> bytes:
        return self._client.get(dated.asset_url)

    def provider_copyright(self, provider_id: int) -> str | None:
        layer = self._by_id.get(provider_id)
        return layer.title if layer is not None else None


def _query_string(params: dict[str, str]) -> str:
    from urllib.parse import urlencode

    return urlencode(params)


def _envelope_3857(aoi) -> dict:
    """The AOI's bounding box in Web Mercator, as an Esri envelope.

    A bounding box is deliberately used rather than the AOI's own rings: it is
    only a server-side prefilter, and the exact clip happens locally with
    shapely.  That sidesteps Esri's orientation-based ring/hole encoding, which
    is easy to get subtly wrong when translating from a shapely geometry.
    """
    from pyproj import Transformer

    minx, miny, maxx, maxy = aoi.bounds
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x0, y0 = transformer.transform(minx, miny)
    x1, y1 = transformer.transform(maxx, maxy)
    return {
        "xmin": x0,
        "ymin": y0,
        "xmax": x1,
        "ymax": y1,
        "spatialReference": {"wkid": 3857},
    }


def _rows_to_dated_geometries(frame) -> list[tuple[_dt.date, object]]:
    """Convert a queried feature frame to ``(capture_date, geometry_4326)``."""
    from shapely import make_valid

    if frame.crs is not None:
        frame = frame.to_crs("EPSG:4326")

    out: list[tuple[_dt.date, object]] = []
    # strict=True is safe: both operands are columns of the same frame and so
    # are equal length by construction; a mismatch would mean real corruption.
    for date_value, geom in zip(frame.get("SRC_DATE2"), frame.geometry, strict=True):
        if geom is None or date_value is None:
            continue
        date = _coerce_date(date_value)
        if date is None:
            continue
        if not geom.is_valid:
            # Capture footprints are routinely self-intersecting.
            geom = make_valid(geom)  # noqa: PLW2901 — intentional narrowing rebind
            if geom.is_empty:
                continue
        out.append((date, geom))
    return out


def _coerce_date(value) -> _dt.date | None:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, (int, float)):
        return _dt.datetime.fromtimestamp(value / 1000, _dt.timezone.utc).date()
    try:
        return _dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None
