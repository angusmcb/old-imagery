"""Esri World Imagery Wayback client.

Ported from ``LibEsri`` in Mbucari/GEHistoricalImagery.
"""

from __future__ import annotations

import concurrent.futures
import datetime as _dt
import io
import json
import math
import re
import threading
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from ._concurrency import _WARNING_FILTER_LOCK, adaptive_metadata_map, workers_for
from ._http import CachedHttpClient, RequestFailed
from ._region import MercatorGrid, MercatorTile, _polygonal_only

WMTS_CAPABILITIES = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/world_imagery/"
    "mapserver/wmts/1.0.0/wmtscapabilities.xml"
)
_CAPS_MAX_AGE = 7 * 24 * 3600
# A tilemap is tied to one immutable Wayback release id. Keep a long refresh
# window in case Esri repairs or reindexes the historical service.
_TILEMAP_MAX_AGE = 30 * 24 * 3600
# Metadata queries are tied to an immutable Wayback release identifier. Keep
# them indefinitely, just like other content-addressed archive responses.
_METADATA_MAX_AGE = None
# Esri serves these with an https scheme, which is unusual for XML namespaces;
# accept either so the parser does not hinge on that detail.
_OWS_NAMESPACES = ("https://www.opengis.net/ows/1.1", "http://www.opengis.net/ows/1.1")
_KEY_TEXT = "/World_Imagery"
_DATE_IN_TITLE = re.compile(r"\(Wayback (\d{4}-\d{2}-\d{2})\)")
# How many metadata records to request by OBJECTID at once.  Esri caps feature
# responses, but documents returnIdsOnly responses as unlimited.  Discover all
# matching IDs first, then keep each attribute response comfortably bounded.
_ATTRIBUTE_BATCH = 1_000
# Keep multipart polygon query bodies bounded.  A single component may exceed
# this and is still sent whole; splitting it without changing its footprint
# would require clipping against an artificial grid.  MultiPolygon components,
# on the other hand, can be divided into independent queries and their IDs
# safely unioned.
_QUERY_GEOMETRY_MAX_BYTES = 256_000
# How many OBJECTIDs to ask for in one geometry request.  Footprints are large,
# so this trades request count against response size rather than URL length
# (the ids travel in a POST body).
_GEOMETRY_BATCH = 100
# Capture-footprint boundaries describe acquisition provenance, not imagery
# pixels.  Ten-metre generalisation retains meaningful seam detail while
# keeping country-scale responses and topology work tractable.
_FOOTPRINT_TOLERANCE_M = 10.0
_FEATURE_CACHE_VERSION = 1
_SOURCE_FIELDS = "OBJECTID,SRC_DATE2,SRC_RES,SRC_ACC,NICE_NAME,NICE_DESC,MinMapLevel,MaxMapLevel"
_ORGANIZE_POLYGONS_WARNING = (
    r"organizePolygons\(\) received an unexpected geometry\.  Either a polygon with interior "
    r"rings, or a polygon with less than 4 points, or a non-Polygon geometry\.  Return arguments "
    r"as a collection\."
)


def _complete_object_id_response(raw: bytes) -> bool:
    """Whether an Esri ID response is complete enough to cache indefinitely."""
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, UnicodeDecodeError):
        return False
    if not isinstance(payload, dict) or "error" in payload:
        return False
    if "objectIds" not in payload:
        return False
    object_ids = payload["objectIds"]
    return (
        (object_ids is None or isinstance(object_ids, list))
        and not payload.get("exceededTransferLimit", False)
    )


def _feature_cache_key(url: str, object_id: int) -> str:
    return f"esri-feature-v{_FEATURE_CACHE_VERSION}\0{url}\0{object_id}"


def _cached_feature_matches(raw: bytes, object_id: int) -> bool:
    try:
        record = json.loads(raw)
        feature = record["feature"]
        return (
            isinstance(record.get("spatialReference"), dict)
            and int(feature["attributes"]["OBJECTID"]) == object_id
            and "geometry" in feature
        )
    except (KeyError, TypeError, ValueError, UnicodeDecodeError):
        return False

# Above this many tiles, stop narrowing the release list with tilemap probes and
# just ask every release.
#
# PROVISIONAL. Narrowing trades ~38 tilemap requests per tile for ~165 metadata
# requests, so whether it pays turns entirely on the cost ratio between the two
# endpoints: it wins below roughly 4.3x that ratio in tiles. That ratio could not
# be pinned down here -- the metadata endpoint measured 2.21 s/request at one
# point in a session and 0.42 s/request at another, against a tilemap endpoint
# steady near 0.31-0.34 s, on a link known to be degraded. A 7.1x ratio puts the
# threshold near 31; a 1.2x ratio puts it near 5.
#
# 33 is the high end of that range, chosen because narrowing is what makes the
# footprint path affordable on the small areas where it is most wanted, and
# because being wrong here costs throughput rather than correctness -- the answer
# is verified identical either way (see candidate_releases). Re-measure the two
# endpoints on a stable connection before trusting this number.
NARROW_RELEASES_MAX_TILES = 33


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
    source: EsriSource | None = None

    @property
    def asset_url(self) -> str:
        return self.layer.asset_url(self.tile)


@dataclass(frozen=True)
class EsriSource:
    """Source provenance attached to one Esri imagery footprint."""

    provider: str | None = None
    description: str | None = None
    resolution_m: float | None = None
    accuracy_m: float | None = None
    min_map_level: int | None = None
    max_map_level: int | None = None


@dataclass(frozen=True)
class EsriFootprint:
    """One capture footprint and its source provenance, in EPSG:4326."""

    date: _dt.date
    geometry: object
    source: EsriSource
    release_title: str | None = None


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
        self._metadata_cache: dict[
            tuple[int, int, int, int], tuple[_dt.date, EsriSource] | None
        ] = {}
        self._lock = threading.Lock()

    # -- helpers -----------------------------------------------------------
    def _json(self, url: str, *, max_age: float | None = None) -> dict | None:
        try:
            return json.loads(self._client.get(url, max_age=max_age))
        except (RequestFailed, ValueError, UnicodeDecodeError):
            return None

    def _tile_metadata(
        self, layer: Layer, tile: MercatorTile
    ) -> tuple[_dt.date, EsriSource] | None:
        """Capture date and source provenance at the centre of ``tile``."""
        key = (layer.id, tile.level, tile.row, tile.column)
        with self._lock:
            if key in self._metadata_cache:
                return self._metadata_cache[key]

        lon, lat = tile.center
        query = {
            "f": "json",
            "outFields": _SOURCE_FIELDS,
            "spatialRel": "esriSpatialRelWithin",
            "geometryType": "esriGeometryPoint",
            "inSR": "4326",
            "geometry": json.dumps({"x": lon, "y": lat, "spatialReference": {"wkid": 4326}}),
            "returnGeometry": "false",
        }
        url = layer.metadata_query_url(tile.level) + "?" + _query_string(query)
        payload = self._json(url, max_age=_METADATA_MAX_AGE)
        # A release date is not an image capture date. If the metadata service
        # gives us nothing usable, omit this version rather than silently
        # changing the meaning of every date exposed by the public API.
        result = None
        if payload is not None:
            try:
                attributes = payload["features"][0]["attributes"]
                date = _coerce_date(attributes.get("SRC_DATE2"))
                if date is not None:
                    result = (date, _source_from_attributes(attributes))
            except (TypeError, KeyError, IndexError):
                pass

        with self._lock:
            self._metadata_cache[key] = result
        return result

    def _capture_date(self, layer: Layer, tile: MercatorTile) -> _dt.date | None:
        """The true capture date of ``tile`` in ``layer``, or ``None`` if unavailable."""
        metadata = self._tile_metadata(layer, tile)
        return metadata[0] if metadata is not None else None

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

    def tile_at_release(
        self,
        tile: MercatorTile,
        layer: Layer,
        *,
        include_metadata: bool = True,
    ) -> DatedEsriTile:
        """The tile served by one exact Wayback release snapshot.

        Unlike :meth:`dated_tiles`, this does not search or fall back across
        releases. The capture date may be unknown, but the requested layer is
        retained so downloading always targets that exact published snapshot.
        """
        metadata = self._tile_metadata(layer, tile) if include_metadata else None
        return DatedEsriTile(
            tile=tile,
            date=metadata[0] if metadata is not None else None,
            provider=layer.id,
            epoch=layer.id,
            layer=layer,
            source=metadata[1] if metadata is not None else None,
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
        last_source: EsriSource | None = None
        skip_until: int | None = None

        for layer in self.layers:
            if skip_until is not None:
                if skip_until == layer.id:
                    skip_until = None
                else:
                    continue

            payload = self._json(layer.tilemap_url(tile), max_age=_TILEMAP_MAX_AGE)
            effective = layer
            select = (payload or {}).get("select")
            if select:
                skip_until = int(select[0])
                effective = self._by_id.get(skip_until, layer)

            data = (payload or {}).get("data")
            if not data or data[0] != 1:
                continue

            metadata = self._tile_metadata(effective, tile)
            if metadata is None:
                continue
            date, source = metadata
            if last_date is not None and last_layer is not None and last_date != date:
                # Emit only when the tile's imagery actually changed, so each
                # entry is the earliest release carrying that imagery.
                results.append(self._make(tile, last_date, last_layer, last_source))
            last_date, last_layer, last_source = date, effective, source

        if last_date is not None and last_layer is not None:
            results.append(self._make(tile, last_date, last_layer, last_source))
        return results

    @staticmethod
    def _make(
        tile: MercatorTile, date: _dt.date, layer: Layer, source: EsriSource | None
    ) -> DatedEsriTile:
        return DatedEsriTile(
            tile=tile,
            date=date,
            provider=layer.id,
            epoch=layer.id,
            layer=layer,
            source=source,
        )

    # -- region-wide availability -----------------------------------------
    def candidate_releases(self, tiles) -> list[Layer]:
        """Releases that changed the imagery over ``tiles``, newest first.

        Answers "which releases do we even need to ask about?" using only the
        tilemap endpoint, whose ``select`` field names the next release that
        changed a tile -- the same skip-ahead :meth:`dated_tiles` relies on.
        Crucially it never touches the metadata service, trading queries there
        for queries against a cheaper endpoint. How much cheaper could not be
        established: see NARROW_RELEASES_MAX_TILES.

        The catalogue holds ~195 releases but only a handful ever touched any
        given area, so this typically returns 10-20. Verified against querying
        all 195: on a 12-tile z17 area, 20 candidates found the same 11 capture
        dates; on a 9-tile z18 area straddling a capture seam, 10 candidates
        found the same 12.

        Probe at the zoom you intend to answer at. Wayback composes the mosaic
        per scale, so change history is per-scale too: a coarse tile covering
        the same ground is *not* a safe shortcut. Measured on that 9-tile area,
        one z13 tile returned 41 releases yet missed 8 the z18 tiles found, and
        so found only 10 of the 12 dates.

        Every tile is probed rather than a sample. Sampling looks tempting --
        on three test areas (z11/z17/z18, 42/12/9 tiles) every single tile
        returned the identical candidate set, so one would have sufficed -- but
        it buys far less than the request counts suggest. One tile's chain is a
        *serial* dependency: each hop is the previous response's ``select``.
        Tiles, by contrast, probe concurrently. So n tiles cost about
        ceil(n / workers) x chain-length, not n x chain-length: 12 tiles is
        roughly two waves against one, not twelve times the wall clock. Paying
        under 2x to keep the answer exact is the right trade for a path whose
        entire purpose is exactness -- a missed release here would silently drop
        a capture date, which is indistinguishable from the archive not having
        one.
        """
        found = adaptive_metadata_map("esri-tilemap", self._candidate_releases_for_tile, tiles)
        ids = {layer.id for layers in found for layer in layers}
        # Back into document order, which is newest first.
        return [layer for layer in self.layers if layer.id in ids]

    def _candidate_releases_for_tile(self, tile: MercatorTile) -> list[Layer]:
        out: list[Layer] = []
        skip_until: int | None = None
        for layer in self.layers:
            if skip_until is not None:
                if skip_until == layer.id:
                    skip_until = None
                else:
                    continue
            payload = self._json(layer.tilemap_url(tile), max_age=_TILEMAP_MAX_AGE)
            effective = layer
            select = (payload or {}).get("select")
            if select:
                skip_until = int(select[0])
                effective = self._by_id.get(skip_until, layer)
            data = (payload or {}).get("data")
            if not data or data[0] != 1:
                continue
            out.append(effective)
        return out

    def dated_regions(
        self,
        aoi,
        zoom: int,
        *,
        min_date: _dt.date | None = None,
        max_date: _dt.date | None = None,
        tiles: Sequence[MercatorTile] | None = None,
    ) -> list[EsriFootprint]:
        """Capture footprints intersecting ``aoi``.

        Returns source-attributed capture footprints. The capture date and
        Wayback release title remain distinct pieces of provenance.

        One logical query per Wayback release against the metadata feature
        service, rather than probing every release for every tile. Very large
        multipart AOIs may split that query by independent polygon component.
        The returned geometry is the true capture footprint rather than a
        tile-quantised approximation.

        Pass the AOI's ``tiles`` to let :meth:`candidate_releases` cut the
        release list down first, which on a small area replaces most of ~195
        slow metadata queries with a smaller number of cheap tilemap ones. It is
        skipped above ``NARROW_RELEASES_MAX_TILES``, where probing every tile
        would cost more than it saves.

        ``aoi`` and the returned geometries are in EPSG:4326.
        """
        # A release published before min_date cannot contain imagery captured
        # after it, so those releases can be dropped outright.
        layers = [layer for layer in self.layers if min_date is None or layer.date >= min_date]
        if tiles and len(tiles) <= NARROW_RELEASES_MAX_TILES:
            # Releases that never changed this area show the same footprints as
            # the next release that did, so querying them adds nothing.
            wanted_ids = {layer.id for layer in self.candidate_releases(tiles)}
            layers = [layer for layer in layers if layer.id in wanted_ids]
        # Oldest first, so the max_date short-circuit below can cut the tail.
        layers.sort(key=lambda layer: layer.date)
        if not layers:
            return []

        cancel_after: list[_dt.date | None] = [None]
        results: list[EsriFootprint] = []
        lock = threading.Lock()

        # Phase 1: which capture dates does each release expose here? Cheap and
        # geometry-free; normally one request per release, with large multipart
        # query bodies split by independent AOI component.
        wanted: list[tuple[Layer, _dt.date, int]] = []

        def query(layer: Layer) -> bool:
            with lock:
                limit = cancel_after[0]
            if limit is not None and layer.date > limit:
                return True  # an older release already proved the tail is empty

            # Partial results are accepted here on purpose: availability over a
            # flaky archive degrades to "less found" rather than raising.
            found, _complete = self._query_layer(layer, aoi, zoom)
            if not found:
                return _complete

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
            return _complete

        adaptive_metadata_map("esri-feature", query, layers, is_acceptable=bool)

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

        # One fetch group per release rather than per (release, date): a single
        # request carries many OBJECTIDs and reports each footprint's own
        # SRC_DATE2, so the date no longer has to be paired up here.
        groups: dict[int, tuple[Layer, set[int], set[_dt.date]]] = {}
        for date, layer in earliest.items():
            _, oids, dates = groups.setdefault(layer.id, (layer, set(), set()))
            oids.update(by_layer_date[(date, layer.id)])
            dates.add(date)

        def fetch(group):
            layer, oids, dates = group
            return [
                EsriFootprint(
                    date=footprint.date,
                    geometry=footprint.geometry,
                    source=footprint.source,
                    release_title=layer.title,
                )
                # Only dates this release was chosen for: another release may be
                # the earliest for a date it also lists, and the min_date /
                # max_date bounds were applied to `wanted`, not to the service.
                for footprint in self._fetch_geometries(layer, zoom, sorted(oids))
                if footprint.date in dates
            ]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers_for("esri", len(groups))
        ) as pool:
            for rows in pool.map(fetch, groups.values()):
                results.extend(rows)
        return results

    # -- one exact release ------------------------------------------------
    def release_footprints(
        self, layer: Layer, aoi, zoom: int, *, max_footprints: int = 1_000
    ) -> list[EsriFootprint]:
        """Capture footprints displayed by one exact release at one zoom.

        Returns source-attributed footprints in EPSG:4326 -- the seam map of a
        single published snapshot, generalised to a 10-metre tolerance rather
        than quantised to whole tiles.

        ``zoom`` matters: :meth:`Layer.metadata_query_url` selects a metadata
        layer per scale, so the same ground in the same release can carry a
        different capture date at one zoom than at another.

        Unlike :meth:`dated_regions`, this refuses partial answers.  A caller
        asking what a release displays is building a map, and a quietly
        truncated feature list would produce holes indistinguishable from ground
        the release genuinely does not cover.
        """
        object_ids, complete = self._query_object_ids(layer, aoi, zoom)
        if not complete:
            raise RequestFailed(
                f"The Esri metadata service did not return a complete feature "
                f"list for release {layer.identifier} at zoom {zoom}. Retrying "
                f"may succeed; a partial list is refused here because it would "
                f"read as missing imagery rather than a failed request."
            )
        if len(object_ids) > max_footprints:
            raise ValueError(
                f"Release {layer.identifier} publishes {len(object_ids):,} capture "
                f"footprints over this area at zoom {zoom}, above the limit of "
                f"{max_footprints:,}. Use a smaller area or a lower zoom, or "
                f"raise max_footprints."
            )
        if not object_ids:
            return []
        return self._fetch_geometries(layer, zoom, object_ids)

    def _query_layer(
        self, layer: Layer, aoi, zoom: int
    ) -> tuple[list[tuple[_dt.date, int]], bool]:
        """Return ``([(capture_date, object_id), ...], complete)`` for one release.

        Deliberately requests no geometry.  Capture footprints are large -- one
        sampled polygon had 3,520 vertices -- and almost every release repeats
        the same footprint, so fetching geometry here would download the same
        megabytes ~195 times.  Geometry is fetched in batches by
        :meth:`_fetch_geometries` instead.

        Esri limits feature responses but not ``returnIdsOnly`` responses.  The
        matching IDs are therefore discovered with size-bounded exact-polygon
        requests and their capture dates fetched in bounded batches, rather
        than walking an open-ended sequence of offsets.

        ``complete`` is False when either ID discovery or an attribute batch is
        failed, malformed, or truncated.  Callers that build a map of what is
        displayed must not treat a partial list as the whole truth;
        :meth:`dated_regions` accepts partial results on purpose, while
        :meth:`release_footprints` refuses them.
        """
        object_ids, complete = self._query_object_ids(layer, aoi, zoom)
        if not object_ids:
            return [], complete

        out: list[tuple[_dt.date, int]] = []
        for start in range(0, len(object_ids), _ATTRIBUTE_BATCH):
            batch = object_ids[start : start + _ATTRIBUTE_BATCH]
            rows = self._fetch_attribute_batch(layer, zoom, batch)
            if rows is None:
                return out, False
            out.extend(rows)
        return out, complete

    def _query_object_ids(
        self, layer: Layer, aoi, zoom: int
    ) -> tuple[list[int], bool]:
        """Return IDs intersecting the exact AOI and whether every query completed."""
        url = layer.metadata_query_url(zoom)
        object_ids: set[int] = set()
        for geometry in _polygon_queries_3857(aoi):
            form = {
                "f": "json",
                "spatialRel": "esriSpatialRelIntersects",
                "geometryType": "esriGeometryPolygon",
                "inSR": "3857",
                "geometry": geometry,
                "returnGeometry": "false",
                "returnIdsOnly": "true",
            }
            try:
                payload = json.loads(
                    self._client.post(
                        url,
                        form,
                        max_age=_METADATA_MAX_AGE,
                        accept_response=_complete_object_id_response,
                    )
                )
            except (RequestFailed, OSError, ValueError):
                return sorted(object_ids), False
            if "error" in payload or "objectIds" not in payload:
                return sorted(object_ids), False

            raw_ids = payload.get("objectIds")
            if raw_ids is None:
                raw_ids = []
            if not isinstance(raw_ids, list):
                return sorted(object_ids), False
            try:
                object_ids.update(int(oid) for oid in raw_ids)
            except (TypeError, ValueError):
                return sorted(object_ids), False
            if payload.get("exceededTransferLimit", False):
                return sorted(object_ids), False
        return sorted(object_ids), True

    def _fetch_attribute_batch(
        self, layer: Layer, zoom: int, object_ids: Sequence[int]
    ) -> list[tuple[_dt.date, int]] | None:
        """Fetch capture dates, splitting a batch if the service truncates it."""
        form = {
            "f": "json",
            "outFields": "OBJECTID,SRC_DATE2",
            "objectIds": ",".join(str(oid) for oid in object_ids),
            "returnGeometry": "false",
        }
        try:
            payload = json.loads(
                self._client.post(
                    layer.metadata_query_url(zoom), form, max_age=_METADATA_MAX_AGE
                )
            )
        except (RequestFailed, OSError, ValueError):
            return None
        if "error" in payload or not isinstance(payload.get("features"), list):
            return None

        by_id: dict[int, _dt.date | None] = {}
        for feature in payload["features"]:
            attributes = feature.get("attributes") or {}
            oid = attributes.get("OBJECTID")
            if oid is None:
                return None
            try:
                oid = int(oid)
            except (TypeError, ValueError):
                return None
            if oid in by_id:
                return None
            by_id[oid] = _coerce_date(attributes.get("SRC_DATE2"))

        expected = set(object_ids)
        if set(by_id) == expected and not payload.get("exceededTransferLimit", False):
            return [
                (date, oid)
                for oid in object_ids
                if (date := by_id[oid]) is not None
            ]

        # Some services enforce a smaller feature limit than advertised.  Split
        # and retry instead of baking another service-specific ceiling into the
        # client.  A one-record batch cannot make further progress.
        if len(object_ids) <= 1 or not set(by_id).issubset(expected):
            return None
        middle = len(object_ids) // 2
        left = self._fetch_attribute_batch(layer, zoom, object_ids[:middle])
        if left is None:
            return None
        right = self._fetch_attribute_batch(layer, zoom, object_ids[middle:])
        if right is None:
            return None
        return left + right

    def _fetch_geometries(
        self, layer: Layer, zoom: int, object_ids: Sequence[int]
    ) -> list[EsriFootprint]:
        """Fetch capture footprints by OBJECTID, as EPSG:4326 geometries.

        Network requests carry up to ``_GEOMETRY_BATCH`` ids, but each returned
        feature is cached independently. Overlapping AOIs therefore reuse a
        footprint even when their request batches differ. The capture date of
        each footprint is read from its own ``SRC_DATE2`` attribute.

        A batch whose request or decode fails is dropped, so a single bad
        response costs its own footprints rather than the whole call.
        """
        batches = [
            object_ids[start : start + _GEOMETRY_BATCH]
            for start in range(0, len(object_ids), _GEOMETRY_BATCH)
        ]
        if len(batches) == 1:
            return self._fetch_geometry_batch(layer, zoom, batches[0])

        def fetch_batch(batch: Sequence[int]) -> list[EsriFootprint]:
            return self._fetch_geometry_batch(layer, zoom, batch)

        out: list[EsriFootprint] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers_for("esri", len(batches))
        ) as pool:
            for rows in pool.map(fetch_batch, batches):
                out.extend(rows)
        return out

    def _fetch_geometry_batch(
        self, layer: Layer, zoom: int, object_ids: Sequence[int]
    ) -> list[EsriFootprint]:
        import geopandas as gpd

        url = layer.metadata_query_url(zoom)
        cached: dict[int, bytes] = {}
        missing: list[int] = []
        for object_id in object_ids:
            raw_feature = self._client._read_cache(
                _feature_cache_key(url, object_id), _METADATA_MAX_AGE
            )
            if raw_feature is None or not _cached_feature_matches(raw_feature, object_id):
                missing.append(object_id)
            else:
                cached[object_id] = raw_feature

        if missing:
            fetched = self._fetch_uncached_geometry_features(layer, zoom, missing)
            for object_id, raw_feature in fetched.items():
                self._client._write_cache(_feature_cache_key(url, object_id), raw_feature)
            cached.update(fetched)

        records = [json.loads(cached[object_id]) for object_id in object_ids if object_id in cached]
        if not records:
            return []
        features = [record["feature"] for record in records]
        raw = json.dumps(
            {
                "geometryType": "esriGeometryPolygon",
                "spatialReference": records[0]["spatialReference"],
                "features": features,
            },
            separators=(",", ":"),
        ).encode()
        try:
            with _WARNING_FILTER_LOCK, warnings.catch_warnings():
                # GDAL emits this while converting malformed Esri polygon rings
                # to a GeometryCollection. _rows_to_dated_geometries repairs and
                # extracts their polygonal parts immediately afterward.
                warnings.filterwarnings(
                    "ignore", message=_ORGANIZE_POLYGONS_WARNING, category=RuntimeWarning
                )
                frame = gpd.read_file(io.BytesIO(raw))
        except Exception:  # noqa: BLE001
            # Deliberately broad: read_file dispatches to GDAL/pyogrio drivers
            # whose failure modes on unexpected bytes are not a stable, listable
            # set. One unreadable batch degrades to [] rather than aborting the
            # whole call.
            return []
        return _rows_to_dated_geometries(frame)

    def _fetch_uncached_geometry_features(
        self, layer: Layer, zoom: int, object_ids: Sequence[int]
    ) -> dict[int, bytes]:
        # Esri cannot clip returned features to the query AOI, so country-scale
        # queries otherwise download and repair millions of vertices lying far
        # outside the final seams. The fixed tolerance reflects the provenance
        # precision of acquisition boundaries rather than the requested imagery
        # pixel size. Coordinates are rounded to whole metres, one tenth of the
        # permitted deviation.
        form = {
            "f": "json",
            "outFields": _SOURCE_FIELDS,
            "objectIds": ",".join(str(oid) for oid in object_ids),
            "returnGeometry": "true",
            "maxAllowableOffset": str(_FOOTPRINT_TOLERANCE_M),
            "geometryPrecision": "0",
            "outSR": "3857",
        }
        try:
            raw = self._client.post_uncached(layer.metadata_query_url(zoom), form)
            payload = json.loads(raw)
        except (RequestFailed, OSError, ValueError):
            return {}
        features = payload.get("features")
        if "error" in payload or not isinstance(features, list):
            return {}

        expected = set(object_ids)
        by_id: dict[int, bytes] = {}
        for feature in features:
            try:
                object_id = int(feature["attributes"]["OBJECTID"])
            except (KeyError, TypeError, ValueError):
                return {}
            if object_id in by_id or object_id not in expected or "geometry" not in feature:
                return {}
            by_id[object_id] = json.dumps(
                {
                    "spatialReference": payload.get("spatialReference") or {"wkid": 3857},
                    "feature": feature,
                },
                separators=(",", ":"),
            ).encode()

        if set(by_id) == expected and not payload.get("exceededTransferLimit", False):
            return by_id
        if len(object_ids) <= 1 or not set(by_id).issubset(expected):
            return {}
        middle = len(object_ids) // 2
        left = self._fetch_uncached_geometry_features(layer, zoom, object_ids[:middle])
        right = self._fetch_uncached_geometry_features(layer, zoom, object_ids[middle:])
        return left | right

    def download_tile_image(self, dated: DatedEsriTile) -> bytes:
        return self._client.get(dated.asset_url)

    def provider_copyright(self, provider_id: int) -> str | None:
        layer = self._by_id.get(provider_id)
        return layer.title if layer is not None else None


def _query_string(params: dict[str, str]) -> str:
    from urllib.parse import urlencode

    return urlencode(params)


def _polygon_queries_3857(aoi) -> list[str]:
    """Encode an AOI as size-bounded Esri JSON multipart polygon queries.

    Esri expects clockwise exterior rings and counter-clockwise holes.  Each
    polygon and its holes stay in the same request; independent components may
    be split across requests, whose returned OBJECTIDs are unioned by the
    caller.  This keeps a sparse AOI sparse instead of expanding it to one vast
    bounding envelope.
    """
    from pyproj import Transformer
    from shapely import make_valid
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
    from shapely.geometry.polygon import orient
    from shapely.ops import transform

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    projected = make_valid(transform(transformer.transform, aoi))

    def polygons(geometry):
        if isinstance(geometry, Polygon):
            yield geometry
        elif isinstance(geometry, (MultiPolygon, GeometryCollection)):
            for part in geometry.geoms:
                yield from polygons(part)

    # Store each polygon's exterior and holes as one atomic group so chunking
    # cannot separate a hole from the exterior ring that gives it meaning.
    encoded_parts: list[list[str]] = []
    for polygon in polygons(projected):
        oriented = orient(polygon, sign=-1.0)
        rings = [oriented.exterior, *oriented.interiors]
        encoded_parts.append(
            [
                json.dumps(list(ring.coords), separators=(",", ":"))
                for ring in rings
            ]
        )

    prefix = '{"rings":['
    suffix = '],"spatialReference":{"wkid":3857}}'

    def encoded_size(rings: Sequence[str]) -> int:
        return len(prefix) + sum(map(len, rings)) + max(0, len(rings) - 1) + len(suffix)

    queries: list[str] = []
    current: list[str] = []
    for part in encoded_parts:
        candidate = current + part
        if current and encoded_size(candidate) > _QUERY_GEOMETRY_MAX_BYTES:
            queries.append(prefix + ",".join(current) + suffix)
            current = list(part)
        else:
            current = candidate
    if current:
        queries.append(prefix + ",".join(current) + suffix)
    if not queries:
        raise ValueError("The area of interest has no polygonal components")
    return queries


def _rows_to_dated_geometries(frame) -> list[EsriFootprint]:
    """Convert queried features to source-attributed EPSG:4326 footprints."""
    if frame.crs is not None:
        frame = frame.to_crs("EPSG:4326")

    out: list[EsriFootprint] = []
    for _, row in frame.iterrows():
        date_value = row.get("SRC_DATE2")
        geom = row.geometry
        if geom is None or date_value is None:
            continue
        date = _coerce_date(date_value)
        if date is None:
            continue
        # Capture footprints are routinely self-intersecting. Repair them, but
        # discard collapsed line/point remnants: a footprint must have area.
        geom = _polygonal_only(geom)
        if geom is None:
            continue
        out.append(
            EsriFootprint(
                date=date,
                geometry=geom,
                source=_source_from_attributes(row),
            )
        )
    return out


def _source_from_attributes(attributes) -> EsriSource:
    """Normalise optional Esri source fields without inventing missing values."""
    return EsriSource(
        provider=_optional_text(attributes.get("NICE_NAME")),
        description=_optional_text(attributes.get("NICE_DESC")),
        resolution_m=_optional_float(attributes.get("SRC_RES")),
        accuracy_m=_optional_float(attributes.get("SRC_ACC")),
        min_map_level=_optional_int(attributes.get("MinMapLevel")),
        max_map_level=_optional_int(attributes.get("MaxMapLevel")),
    )


def _optional_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


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
