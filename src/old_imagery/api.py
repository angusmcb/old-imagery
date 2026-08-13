"""Public functions for finding, selecting, and downloading historical imagery."""

from __future__ import annotations

import datetime as _dt
import json
import os
import threading
import warnings
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely.geometry.base
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from shapely import MultiPoint, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from ._concurrency import workers_for
from ._dbroot import Database, DbRoot
from ._http import DEFAULT_CACHE_DIR, CachedHttpClient, RequestFailed
from ._region import TILE_PX, dissolve, normalize_aoi, sort_by_nearest_date

WGS84 = "EPSG:4326"
MERCATOR = "EPSG:3857"

DateLike = _dt.date | str
TileGeometry = Point | MultiPoint | Polygon | MultiPolygon


@dataclass(frozen=True)
class SourceMetadata:
    """Provider provenance sampled at the centre of one native image tile."""

    provider: str | None = None
    description: str | None = None
    resolution_m: float | None = None
    accuracy_m: float | None = None
    min_map_level: int | None = None
    max_map_level: int | None = None


@dataclass(frozen=True)
class DownloadedTile:
    """One unchanged provider image payload and its selection provenance."""

    content: bytes
    image_format: str
    media_type: str
    provider: Literal["google", "esri"]
    tile_scheme: str
    zoom: int
    column: int
    row: int
    bounds_wgs84: tuple[float, float, float, float]
    capture_date_at_center: _dt.date | None
    source_metadata_at_center: SourceMetadata | None
    release_id: str | None = None
    release_date: _dt.date | None = None
    release_title: str | None = None


# The two string-valued options are spelled out as Literals in each signature
# rather than hidden behind named aliases, so `help()`, an IDE tooltip and the
# rendered docs all show the accepted values without a lookup.
#
# They constrain callers under a type checker only. Both are still validated at
# runtime -- `_backend` for provider, `sort_by_nearest_date` for date_match --
# because most callers run unchecked, and a typo should raise rather than behave
# arbitrarily. The internal helpers keep plain `str` parameters for the same
# reason: narrowing them would make their own validation branches statically
# unreachable, and mypy runs here with warn_unreachable.

AVAILABILITY_COLUMNS = [
    "date",
    "coverage",
    "complete",
    "providers",
    "source_providers",
    "source_descriptions",
    "source_resolutions_m",
    "source_accuracies_m",
    "min_map_levels",
    "max_map_levels",
    "geometry",
]

# `coverage` is a float ratio, so `complete` cannot test it against 1.0
# exactly. A clip that lands this close to the AOI's own area is a
# reprojection artefact, not a real gap.
_COMPLETE_TOLERANCE = 1e-9
ESRI_MOSAIC_COLUMNS = [
    "zoom",
    "date",
    "area_fraction",
    "release_id",
    "source_provider",
    "source_description",
    "source_resolution_m",
    "source_accuracy_m",
    "min_map_level",
    "max_map_level",
    "geometry",
]

# Esri availability always resolves through capture footprints
# (WayBack.dated_regions), never by probing tiles. There used to be a `method`
# option choosing between them; it is gone, because the footprint path is more
# accurate and the trade-off it was there to expose could not be substantiated.
#
# Request counts are the reliable part -- they do not depend on the link.
# Measured on a 12-tile z17 area:
#
#     per-tile   456 tilemap + 240 metadata   320 KiB
#     footprint  456 tilemap +  30 metadata  2843 KiB
#
# Per-tile issues one metadata point query per (release, tile); the footprint
# path issues one envelope query per candidate release, so its metadata load is
# flat in the tile count rather than linear. Both find the same 11 capture
# dates, and the footprint answer carries true seam geometry.
#
# Wall-clock, cold cache, end to end -- and read this with suspicion:
#
#     tiles         6      12      36
#     per-tile   254.9    42.2    75.3
#     footprint   14.9    25.8   127.4
#
# The footprint path loses at 36 tiles. Not because of query counts (195
# metadata queries is ~7s at 10-wide) but because of payload: it downloads real
# polygons -- one sampled had 3,520 vertices -- and that grows with AOI area
# while per-tile's tiny responses do not. See _esri._GEOMETRY_BATCH and
# `geometryPrecision` if that becomes worth attacking.
#
# Do not calibrate anything against those timings. Per-request latency on the
# link they were taken over swung ~7x *within one session*: the metadata
# endpoint measured 2.21 s/request at one point and 0.42 s/request at another,
# against a tilemap endpoint that stayed near 0.31-0.34 s. Any threshold derived
# from their ratio is a measurement of the connection, not of Esri.
#
# Google has no footprint equivalent: dbRoot reports dates per tile and nothing
# finer, so its geometry is a union of tile extents. That difference between
# providers is real and is documented, rather than papered over by quantising
# Esri down to match.

# Deepest zoom at which each service actually publishes imagery.
#
# These are *not* the tile schemes' addressing limits. Keyhole addresses to
# level 30 and Web Mercator to 23 (see _keyhole.MAX_LEVEL and
# MercatorGrid.max_level, both ported from upstream's KeyholeTile.MaxLevel and
# EsriTile.MaxLevel), but upstream's docs report that imagery "practically caps
# out at 21" for Google Earth and 20 for Wayback. Requests above those levels
# are well-formed and return nothing useful, while costing 4x the tiles per
# level, so they are rejected rather than silently served.
#
# Upstream's CLI applies a single [1,23] bound to both providers; that is a
# consequence of one shared --zoom flag, not a per-provider fact, so it is not
# what is enforced here.
MAX_IMAGERY_ZOOM = {"google": 21, "esri": 20}


def _validate_zoom(zoom: int, provider: str) -> None:
    """Reject zooms deeper than the provider publishes imagery for.

    Unknown providers pass through: _backend raises for those, and its message
    is the more useful one.
    """
    cap = MAX_IMAGERY_ZOOM.get(provider)
    if cap is not None and zoom > cap:
        raise ValueError(
            f"zoom {zoom} is deeper than {provider} publishes imagery for "
            f"(max {cap}). Tiles exist at that level but carry no imagery."
        )


def _as_date(value: DateLike | None) -> _dt.date | None:
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    return _dt.date.fromisoformat(str(value))


def _resolve_release(backend, release_id: str | None):
    """Resolve an exact Esri release by its stable catalogue identifier.

    Deliberately the only release selector on :func:`download`.  Resolving a
    *service date* to a release is :func:`esri_mosaic_as_of`'s job, and it
    reports the identifier it chose -- so there is one place where Esri's
    catalogue-date inconsistencies are handled, not two.
    """
    if release_id is None:
        return None
    if not release_id:
        raise ValueError("esri_wayback_release_id cannot be empty")
    return backend.release_by_identifier(release_id)


def _backend(provider: str, cache_dir: str | os.PathLike | None):
    """Build a provider backend and the HTTP client that owns its connections."""
    client = CachedHttpClient(cache_dir)
    try:
        if provider == "google":
            return DbRoot(Database.TIME_MACHINE, client), client
        if provider == "esri":
            from ._esri import WayBack

            return WayBack(client), client
    except Exception:
        client.close()
        raise
    client.close()
    raise ValueError(f"Unknown provider {provider!r}; expected 'google' or 'esri'")


# --------------------------------------------------------------------------
# availability
# --------------------------------------------------------------------------
def availability(
    aoi: shapely.Polygon | shapely.MultiPolygon | shapely.GeometryCollection,
    zoom: int,
    *,
    min_date: DateLike | None = None,
    max_date: DateLike | None = None,
    provider: Literal["google", "esri"] = "google",
    cache_dir: str | os.PathLike[str] | None = DEFAULT_CACHE_DIR,
    max_tiles: int = 1_000,
) -> gpd.GeoDataFrame:
    """Find which imagery capture dates are available over an area.

    Parameters
    ----------
    aoi : shapely.Polygon | shapely.MultiPolygon | shapely.GeometryCollection
        Area of interest in EPSG:4326 (lon/lat degrees). Must enclose some
        area: coverage is reported as a fraction of it, so a Point or
        LineString is rejected -- buffer it first.
    zoom : int
        Tile zoom level.  Availability is reported at tile granularity, so a
        higher zoom gives finer date boundaries and costs more requests.
    min_date, max_date : datetime.date | str | None
        Optional inclusive bounds on capture date.
    provider : str
        ``"google"`` for Google Earth historical imagery, ``"esri"`` for the
        Esri World Imagery Wayback archive.
    cache_dir : str | os.PathLike[str] | None
        On-disk response cache. Pass ``None`` to disable caching.
    max_tiles : int
        Maximum number of tile-grid cells spanned by the AOI's bounding box.
        Default 1,000.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per capture date, sorted newest first, in EPSG:4326:

        ``date``
            The capture date.
        ``coverage``
            How much of the AOI's **area** this date covers, as a planar ratio
            computed in EPSG:3857. Independent of ``zoom`` and of the provider,
            so a date covering half the AOI reports ``0.5`` whichever way it was
            resolved.
        ``complete``
            True when that date covers the whole AOI (``coverage == 1.0``).
        ``providers``
            Google imagery provider names or Esri Wayback release titles,
            where known.
        ``source_providers``, ``source_descriptions``
            Distinct Esri imagery source names and descriptions for this date.
            Empty tuples for Google or when Esri omits the metadata.
        ``source_resolutions_m``, ``source_accuracies_m``
            Distinct Esri native resolutions and positional accuracies in
            metres. These describe the source imagery, not output pixel size.
        ``min_map_levels``, ``max_map_levels``
            Distinct Esri scale ranges, passed through as provenance only. They
            do not select, cap or otherwise change ``zoom``.
        ``geometry``
            The covered area, clipped to the AOI.

        All dates are **image capture dates**, never a provider's publication
        or release date.

        Google Earth tiles whose imagery carries no capture date (a provider's
        undated default imagery) are excluded, matching upstream behaviour.

        ``geometry`` is resolved as finely as each provider allows, with no
        option to choose. Esri returns true capture footprints, so its date
        boundaries follow real imagery seams. Google's dbRoot reports dates per
        tile and nothing finer, so its geometry is a union of tile extents --
        the AOI's own outline survives the clip, but internal date boundaries
        are tile-shaped.

        ``gdf.attrs`` records ``zoom``, ``provider``, ``n_aoi_tiles`` and
        ``method`` -- ``"per-tile"`` for Google, ``"region-query"`` for Esri, or
        ``"none"`` when the AOI selects no tiles. It is reported so a result can
        say how it was obtained; it is not selectable.

    See Also
    --------
    esri_mosaic_as_of : What one published Esri Wayback snapshot *displays*, and
        where each piece of it came from. That is a different question from this
        one: this function searches the archive for capture dates, that one reads
        the seam map of a single release.
    """
    aoi = normalize_aoi(aoi)
    _validate_zoom(zoom, provider)
    min_d, max_d = _as_date(min_date), _as_date(max_date)
    backend, client = _backend(provider, cache_dir)

    try:
        tiles = backend.grid.tiles(aoi, zoom, max_tiles)
        if not tiles:
            return _empty_availability(zoom, provider)

        # Footprints where the provider publishes them, tiles where it does
        # not. Not a caller's choice: the footprint path is both more accurate
        # and cheaper, so there is nothing to trade off.
        if hasattr(backend, "dated_regions"):
            return _availability_by_region(backend, aoi, tiles, zoom, provider, min_d, max_d)

        by_date: dict[_dt.date, set] = defaultdict(set)
        providers: dict[_dt.date, set[int]] = defaultdict(set)
        lock = threading.Lock()

        def work(tile) -> None:
            dated = backend.dated_tiles(tile)
            with lock:
                for d in dated:
                    if d.date is None:
                        continue
                    if min_d is not None and d.date < min_d:
                        continue
                    if max_d is not None and d.date > max_d:
                        continue
                    by_date[d.date].add(tile)
                    providers[d.date].add(d.provider)

        with ThreadPoolExecutor(max_workers=workers_for(provider, len(tiles))) as pool:
            list(pool.map(work, tiles))

        if not by_date:
            return _empty_availability(zoom, provider, len(tiles), "per-tile")

        dates = sorted(by_date, reverse=True)
        geoms = [dissolve(by_date[date]).intersection(aoi) for date in dates]
        names = [
            ", ".join(
                sorted({n for n in (backend.provider_copyright(p) for p in providers[date]) if n})
            )
            for date in dates
        ]
        rows = _availability_rows(aoi, dates, geoms, names)
        return _finish_availability(rows, zoom, provider, len(tiles), "per-tile")
    finally:
        client.close()


def _availability_by_region(backend, aoi, tiles, zoom, provider, min_d, max_d) -> gpd.GeoDataFrame:
    """Availability from region-wide capture-footprint queries.

    Produces the same columns as the per-tile path.  ``geometry`` is the
    provider-reported capture footprint clipped to the AOI rather than a union
    of whole tiles, and ``coverage`` is measured as area either way, so the two
    paths are directly comparable.

    Tiles are used only to bound the request cost and to narrow the release
    list; they do not shape the answer.
    """
    # Passing the tiles lets the backend skip releases that never touched this
    # area, trading ~175 slow metadata queries for cheap tilemap probes.
    regions = backend.dated_regions(aoi, zoom, min_date=min_d, max_date=max_d, tiles=tiles)
    if not regions:
        return _empty_availability(zoom, provider, len(tiles), "region-query")

    by_date: dict[_dt.date, list] = defaultdict(list)
    titles: dict[_dt.date, set[str]] = defaultdict(set)
    sources: dict[_dt.date, set] = defaultdict(set)
    for footprint in regions:
        by_date[footprint.date].append(footprint.geometry)
        sources[footprint.date].add(footprint.source)
        if footprint.release_title:
            titles[footprint.date].add(footprint.release_title)

    geometry_by_date: dict[_dt.date, shapely.geometry.base.BaseGeometry] = {}
    for date, geoms in by_date.items():
        clipped = unary_union(geoms).intersection(aoi)
        if not clipped.is_empty:
            geometry_by_date[date] = clipped

    if not geometry_by_date:
        return _empty_availability(zoom, provider, len(tiles), "region-query")

    dates = sorted(geometry_by_date, reverse=True)
    rows = _availability_rows(
        aoi,
        dates,
        [geometry_by_date[date] for date in dates],
        [", ".join(sorted(titles.get(date, ()))) for date in dates],
        [sources[date] for date in dates],
    )
    return _finish_availability(rows, zoom, provider, len(tiles), "region-query")


def _availability_rows(aoi, dates, geoms, names, sources=None) -> list[dict]:
    """Assemble availability rows, with coverage measured as area.

    Deliberately not a tile count. The tile grid is a transport detail the
    caller never chose and cannot see, and counting tiles means different things
    on the two paths: a footprint that clips a sliver off every AOI tile would
    report full coverage over almost no ground. An area fraction says the one
    thing a caller actually wants to know -- how much of my area does this date
    cover -- and says it identically for both providers.
    """
    fractions = _area_fractions(aoi, list(geoms))
    if sources is None:
        sources = [set() for _ in dates]
    rows = []
    for date, geom, name, source_set, fraction in zip(
        dates, geoms, names, sources, fractions, strict=True
    ):
        rows.append(
            {
                "date": date,
                "coverage": fraction,
                "complete": fraction >= 1.0 - _COMPLETE_TOLERANCE,
                "providers": name,
                "source_providers": _source_values(source_set, "provider"),
                "source_descriptions": _source_values(source_set, "description"),
                "source_resolutions_m": _source_values(source_set, "resolution_m"),
                "source_accuracies_m": _source_values(source_set, "accuracy_m"),
                "min_map_levels": _source_values(source_set, "min_map_level"),
                "max_map_levels": _source_values(source_set, "max_map_level"),
                "geometry": geom,
            }
        )
    return rows


def _source_values(sources, attribute: str) -> tuple:
    """Sorted distinct non-null values for one source-provenance attribute."""
    return tuple(
        sorted({value for source in sources if (value := getattr(source, attribute)) is not None})
    )


def _finish_availability(rows, zoom, provider, n_tiles, method) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(rows, columns=AVAILABILITY_COLUMNS, geometry="geometry", crs=WGS84)
    gdf.attrs.update(zoom=zoom, provider=provider, n_aoi_tiles=n_tiles, method=method)
    return gdf


def _empty_availability(
    zoom: int,
    provider: str,
    n_tiles: int = 0,
    method: str = "none",
) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        {name: [] for name in AVAILABILITY_COLUMNS}, geometry="geometry", crs=WGS84
    )
    gdf.attrs.update(zoom=zoom, provider=provider, n_aoi_tiles=n_tiles, method=method)
    return gdf


# --------------------------------------------------------------------------
# esri_wayback_releases
# --------------------------------------------------------------------------
def esri_wayback_releases(
    *,
    cache_dir: str | os.PathLike | None = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Return the published Esri Wayback release catalogue.

    Returns one row per release, newest first. ``release_id`` is Esri's stable
    WMTS identifier, ``release_date`` is the publication date, and
    ``release_title`` is the title published in Esri's catalogue.
    """
    backend, client = _backend("esri", cache_dir)
    try:
        layers = sorted(backend.layers, key=lambda layer: layer.date, reverse=True)
        return pd.DataFrame(
            {
                "release_id": pd.Series([layer.identifier for layer in layers], dtype="string"),
                "release_date": pd.Series([layer.date for layer in layers], dtype="datetime64[ns]"),
                "release_title": pd.Series([layer.title for layer in layers], dtype="string"),
            }
        )
    finally:
        client.close()


# --------------------------------------------------------------------------
# esri_mosaic_as_of
# --------------------------------------------------------------------------
def esri_mosaic_as_of(
    aoi: shapely.Polygon | shapely.MultiPolygon | shapely.GeometryCollection,
    zoom: int | Sequence[int],
    as_of: DateLike | str,
    *,
    cache_dir: str | os.PathLike[str] | None = DEFAULT_CACHE_DIR,
    max_footprints: int = 500,
) -> gpd.GeoDataFrame:
    """Map what one Esri Wayback snapshot displays, and where each piece came from.

    A Wayback *release* is a published snapshot of the World Imagery basemap,
    mosaicked from imagery flown across many years.  This answers "if I looked at
    the basemap on this date, what would I be seeing, and how old is each part of
    it?" -- which is a different question from :func:`availability`, and it is
    answered from Esri's own capture footprints rather than by probing tiles.

    Parameters
    ----------
    aoi : shapely.Polygon | shapely.MultiPolygon | shapely.GeometryCollection
        Area of interest in EPSG:4326 (lon/lat degrees). Must enclose some
        area: coverage is reported as a fraction of it, so a Point or
        LineString is rejected -- buffer it first.
    zoom : int | Sequence[int]
        One zoom level, or several.  This is not merely a resolution knob: Esri
        composes the mosaic per scale and publishes metadata per scale, so the
        same ground in the same release can carry a different capture date at
        different zooms.  Pass several to see that.  Zooms of 10 and below all
        resolve to the same metadata layer and so return identical geometry.
    as_of : datetime.date | str
        The date to look at the archive on, or an exact stable release ID such
        as ``"WB_2026_R07"``. For a date, the latest release published on or
        before it is used. It is a **publication** date, not a capture date,
        and no capture-date matching happens.
    cache_dir : str | os.PathLike[str] | None
        On-disk response cache. Pass ``None`` to disable caching.
    max_footprints : int
        Reject areas where the release publishes more capture footprints than
        this at some zoom. Default 500. Footprint count, not tile count, is what
        this call costs.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per distinct ``(zoom, capture date, source metadata)`` value,
        ordered by zoom then newest date first, in EPSG:4326:

        ``zoom``
            The zoom the row was resolved at.
        ``date``
            The **capture date** of the imagery displayed in this area.
        ``area_fraction``
            This row's share of the AOI, as a planar area ratio computed in
            EPSG:3857.  Mercator's area distortion largely cancels between the
            row and the AOI over a modest latitude span, but this is not a true
            area ratio for a tall AOI -- reproject ``geometry`` yourself if you
            need one.
        ``release_id``
            Stable catalogue identifier of the resolved release, for example
            ``"WB_2026_R03"``.  Constant for the whole frame, and a column
            rather than only an attr so concatenating two snapshots cannot lose
            track of which release each row came from.
        ``source_provider``, ``source_description``
            Esri's source name and description, where supplied.
        ``source_resolution_m``, ``source_accuracy_m``
            Native source resolution and positional accuracy in metres, where
            supplied. These describe the imagery, not the output pixel size.
        ``min_map_level``, ``max_map_level``
            Esri's scale range for this source, reported as provenance only.
            These values never change or cap the requested zoom.
        ``geometry``
            The area displaying that capture date, clipped to the AOI.  Real
            capture-footprint boundaries, not tile edges.

        ``gdf.attrs`` records ``release_id``, ``release_date`` (the publication
        date), ``release_title``, ``as_of`` and ``zooms``.

        Footprints sharing a zoom, capture date and identical source metadata
        are dissolved into one row. Ground the release publishes no footprint
        metadata for is absent entirely, so ``area_fraction`` need not sum to 1
        across a zoom.

    Raises
    ------
    ValueError
        If ``as_of`` is unknown or precedes the archive, a zoom is out of range,
        or the release publishes more than ``max_footprints`` footprints here.
    RequestFailed
        If Esri's metadata service returned an incomplete feature list.  A
        partial seam map is refused rather than returned, because its holes
        would be indistinguishable from ground the release does not cover.

    See Also
    --------
    availability : Which capture dates exist over an area, across all releases.
    download : Fetch the pixels; pass ``esri_wayback_release_id=`` with the
        ``release_id`` from this frame to get exactly the snapshot mapped here.
    """
    aoi = normalize_aoi(aoi)
    zooms = _normalise_zooms(zoom)
    backend, client = _backend("esri", cache_dir)
    try:
        if isinstance(as_of, str):
            try:
                requested_as_of: _dt.date | str = _dt.date.fromisoformat(as_of)
            except ValueError:
                requested_as_of = as_of
                layer = backend.release_by_identifier(as_of)
            else:
                layer = backend.release_on_or_before(requested_as_of)
        else:
            requested_date = _as_date(as_of)
            if requested_date is None:
                raise ValueError("as_of is required")
            requested_as_of = requested_date
            layer = backend.release_on_or_before(requested_date)

        def work(z: int) -> list[tuple]:
            footprints = backend.release_footprints(layer, aoi, z, max_footprints=max_footprints)
            by_source: dict[tuple[_dt.date, object], list] = defaultdict(list)
            for footprint in footprints:
                by_source[(footprint.date, footprint.source)].append(footprint.geometry)

            out = []
            for (date, source), geoms in by_source.items():
                clipped = unary_union(geoms).intersection(aoi)
                if not clipped.is_empty:
                    out.append((z, date, source, clipped))
            return out

        # One worker per zoom, capped: release_footprints issues its requests
        # serially, so this bounds requests in flight against Wayback to the
        # measured cap rather than multiplying it by the number of zooms.
        with ThreadPoolExecutor(max_workers=workers_for("esri", len(zooms))) as pool:
            found = [row for rows in pool.map(work, zooms) for row in rows]
    finally:
        client.close()

    found.sort(key=lambda row: (row[0], -row[1].toordinal(), repr(row[2])))
    fractions = _area_fractions(aoi, [geom for _z, _d, _s, geom in found])
    rows = [
        {
            "zoom": z,
            "date": date,
            "area_fraction": fraction,
            "release_id": layer.identifier,
            "source_provider": source.provider,
            "source_description": source.description,
            "source_resolution_m": source.resolution_m,
            "source_accuracy_m": source.accuracy_m,
            "min_map_level": source.min_map_level,
            "max_map_level": source.max_map_level,
            "geometry": geom,
        }
        for (z, date, source, geom), fraction in zip(found, fractions, strict=True)
    ]

    gdf = gpd.GeoDataFrame(
        rows if rows else {name: [] for name in ESRI_MOSAIC_COLUMNS},
        columns=ESRI_MOSAIC_COLUMNS,
        geometry="geometry",
        crs=WGS84,
    )
    gdf.attrs.update(
        release_id=layer.identifier,
        release_date=layer.date,
        release_title=layer.title,
        as_of=requested_as_of,
        zooms=zooms,
    )
    return gdf


def _normalise_zooms(zoom: int | Sequence[int]) -> list[int]:
    """Validate one zoom or several, returning them sorted and deduplicated."""
    if isinstance(zoom, int) and not isinstance(zoom, bool):
        candidates = [zoom]
    elif isinstance(zoom, Sequence) and not isinstance(zoom, (str, bytes)):
        candidates = list(zoom)
        if not candidates:
            raise ValueError("zoom cannot be an empty sequence")
    else:
        raise TypeError(f"zoom must be an int or a sequence of ints, not {type(zoom).__name__}")

    for value in candidates:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"zoom values must be ints, not {type(value).__name__}")
        # The grid would normally enforce the lower bound in MercatorGrid.tiles;
        # this path never enumerates tiles, so it is checked here.
        if value < 0:
            raise ValueError(f"zoom must be at least 0, not {value}")
        _validate_zoom(value, "esri")
    return sorted(set(candidates))


def _area_fractions(
    aoi: shapely.Polygon | shapely.MultiPolygon | shapely.GeometryCollection,
    geoms: list[shapely.geometry.base.BaseGeometry],
) -> list[float]:
    """Each geometry's share of the AOI, as planar areas in EPSG:3857."""
    if not geoms:
        return []
    areas = gpd.GeoSeries([aoi, *geoms], crs=WGS84).to_crs(MERCATOR).area.to_numpy()
    aoi_area = float(areas[0])
    if aoi_area <= 0.0:
        # Unreachable through the public API -- normalize_aoi rejects a
        # zero-area AOI -- but kept so this helper stays total for any caller.
        # A line or point AOI: every clip is zero-area too, so the ratio is
        # genuinely undefined rather than zero.
        return [float("nan")] * len(geoms)
    return [float(area) / aoi_area for area in areas[1:]]


# --------------------------------------------------------------------------
# download_tiles / download
# --------------------------------------------------------------------------
def _validate_download_selection(
    provider: str,
    date: DateLike | None,
    date_match: str,
    esri_wayback_release_id: str | None,
) -> _dt.date | None:
    """Validate the selection shared by raw tiles and mosaics."""
    target = _as_date(date)
    if esri_wayback_release_id is not None:
        if provider != "esri":
            raise ValueError(
                "esri_wayback_release_id requires provider='esri'; "
                "it selects a publication snapshot, not a capture date"
            )
        if target is not None:
            raise ValueError(
                "Choose either date (capture-date mode) or esri_wayback_release_id, not both"
            )
        if date_match != "closest":
            raise ValueError(
                "date_match cannot be set with esri_wayback_release_id; "
                "the exact release layer is used without date fallback"
            )
    elif target is None:
        raise ValueError("A target date is required")
    else:
        # Validate even when the geometry later selects no tiles.
        sort_by_nearest_date([], target, date_match)
    return target


def _validate_tile_geometry(
    geometry: shapely.geometry.base.BaseGeometry,
) -> TileGeometry:
    """Validate one WGS84 point collection or polygonal tile selector."""
    allowed = (Point, MultiPoint, Polygon, MultiPolygon)
    if not isinstance(geometry, allowed):
        choices = "Point, MultiPoint, Polygon or MultiPolygon"
        raise TypeError(f"geometry must be a {choices}, not {type(geometry).__name__}")
    if geometry.is_empty:
        raise ValueError("The tile-selection geometry is empty")
    minx, miny, maxx, maxy = geometry.bounds
    if minx < -180.0 or maxx > 180.0:
        raise ValueError("Longitudes must lie within [-180, 180]")
    if miny < -90.0 or maxy > 90.0:
        raise ValueError("Latitudes must lie within [-90, 90]")
    return geometry


def _tiles_for_geometry(grid, geometry: TileGeometry, zoom: int, max_tiles: int) -> list:
    """Resolve point or polygon selectors onto one provider's native grid."""
    geometry = _validate_tile_geometry(geometry)
    if isinstance(geometry, (Point, MultiPoint)):
        points = [geometry] if isinstance(geometry, Point) else list(geometry.geoms)
        tiles = {grid.tile_at_point(float(point.x), float(point.y), zoom) for point in points}
        if len(tiles) > max_tiles:
            raise ValueError(
                f"The geometry selects {len(tiles):,} tiles at zoom {zoom}, above the "
                f"limit of {max_tiles:,}. Use fewer points or raise max_tiles."
            )
        return sorted(tiles, key=lambda tile: (tile.row, tile.column))
    return grid.tiles(normalize_aoi(geometry), zoom, max_tiles)


def _source_metadata(backend, candidate, provider: str) -> SourceMetadata | None:
    source = getattr(candidate, "source", None)
    if source is not None:
        return SourceMetadata(
            provider=source.provider,
            description=source.description,
            resolution_m=source.resolution_m,
            accuracy_m=source.accuracy_m,
            min_map_level=source.min_map_level,
            max_map_level=source.max_map_level,
        )
    if provider == "google":
        name = backend.provider_copyright(candidate.provider)
        return SourceMetadata(provider=name) if name else None
    return None


def _inspect_tile_payload(raw: bytes) -> tuple[str, str]:
    """Validate a native tile entirely in memory without changing its bytes."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with MemoryFile(raw) as memory, memory.open() as source:
                driver = str(source.driver).upper()
                size = (source.width, source.height)
    except Exception as error:
        raise ValueError("provider returned an invalid image payload") from error
    if size != (TILE_PX, TILE_PX):
        raise ValueError(
            f"provider returned a {size[0]}x{size[1]} tile; expected {TILE_PX}x{TILE_PX}"
        )
    formats = {
        "JPEG": ("jpeg", "image/jpeg"),
        "PNG": ("png", "image/png"),
        "WEBP": ("webp", "image/webp"),
    }
    try:
        return formats[driver]
    except KeyError as error:
        raise ValueError(f"provider returned unsupported image format {driver!r}") from error


def download_tiles(
    geometry: TileGeometry,
    zoom: int,
    date: DateLike | None = None,
    *,
    date_match: Literal["closest", "exact", "before", "after"] = "closest",
    provider: Literal["google", "esri"] = "google",
    esri_wayback_release_id: str | None = None,
    cache_dir: str | os.PathLike[str] | None = DEFAULT_CACHE_DIR,
    max_tiles: int = 1_000,
    include_metadata: bool = True,
) -> list[DownloadedTile]:
    """Download complete native image tiles selected by a WGS84 geometry.

    ``Point`` and ``MultiPoint`` select the one provider-native tile containing
    each point. ``Polygon`` and ``MultiPolygon`` select every tile they
    intersect. Returned payloads are full, unclipped provider images in
    deterministic row-major order. Any selected tile that is missing, fails to
    download, or is not a supported 256x256 image aborts the whole call.

    Payloads remain in memory and are returned unchanged. This function never
    creates output or temporary files. The existing HTTP response cache is the
    only possible disk write; pass ``cache_dir=None`` for a fully diskless call.

    Date and release selection have exactly the same meaning as in
    :func:`download`. Esri capture and source metadata are sampled at each
    tile's centre and therefore need not describe every pixel where a source
    footprint seam crosses the tile.
    """
    _validate_zoom(zoom, provider)
    target = _validate_download_selection(provider, date, date_match, esri_wayback_release_id)
    backend, client = _backend(provider, cache_dir)
    try:
        release_layer = _resolve_release(backend, esri_wayback_release_id)
        tiles = _tiles_for_geometry(backend.grid, geometry, zoom, max_tiles)
        if not tiles:
            raise ValueError("The geometry selects no tiles")

        def work(tile) -> DownloadedTile:
            if release_layer is not None:
                candidate = backend.tile_at_release(
                    tile, release_layer, include_metadata=include_metadata
                )
            else:
                assert target is not None
                candidates = backend.dated_tiles(tile)
                ordered = sort_by_nearest_date(candidates, target, date_match)
                candidate = next(ordered, None)
                if candidate is None:
                    raise ValueError(
                        f"No imagery found for tile z{zoom}/{tile.column}/{tile.row} "
                        f"with date_match={date_match!r}"
                    )
            raw = backend.download_tile_image(candidate)
            image_format, media_type = _inspect_tile_payload(raw)
            layer = getattr(candidate, "layer", None) or release_layer
            return DownloadedTile(
                content=raw,
                image_format=image_format,
                media_type=media_type,
                provider=provider,
                tile_scheme=backend.grid.tile_scheme,
                zoom=zoom,
                column=tile.column,
                row=tile.row,
                bounds_wgs84=tile.bounds_wgs84,
                capture_date_at_center=candidate.date if include_metadata else None,
                source_metadata_at_center=(
                    _source_metadata(backend, candidate, provider) if include_metadata else None
                ),
                release_id=getattr(layer, "identifier", None),
                release_date=getattr(layer, "date", None),
                release_title=getattr(layer, "title", None),
            )

        with ThreadPoolExecutor(max_workers=workers_for(provider, len(tiles))) as pool:
            return list(pool.map(work, tiles))
    finally:
        client.close()


def download(
    aoi: shapely.Polygon | shapely.MultiPolygon | shapely.GeometryCollection,
    zoom: int,
    date: DateLike | None = None,
    *,
    date_match: Literal["closest", "exact", "before", "after"] = "closest",
    provider: Literal["google", "esri"] = "google",
    esri_wayback_release_id: str | None = None,
    cache_dir: str | os.PathLike[str] | None = DEFAULT_CACHE_DIR,
    max_tiles: int = 1_000,
) -> rasterio.DatasetReader:
    """Download and mosaic historical imagery over an area.

    Parameters
    ----------
    aoi : shapely.Polygon | shapely.MultiPolygon | shapely.GeometryCollection
        Area of interest in EPSG:4326 (lon/lat degrees). Must enclose some
        area: coverage is reported as a fraction of it, so a Point or
        LineString is rejected -- buffer it first.
        The output covers the AOI's bounding box, snapped to tile pixels.
    zoom : int
        Tile zoom level.  Ground resolution is roughly ``156543 / 2**zoom``
        metres per pixel at the equator.
    date : datetime.date | str | None
        Target capture date (``datetime.date`` or ISO string). Required for
        normal capture-date selection and forbidden when selecting an exact
        Esri Wayback release.
    date_match : str
        How to resolve tiles lacking imagery on the exact target date:
        ``"closest"`` (default), ``"exact"``, ``"before"`` or ``"after"``.
        Matching happens per tile, so a mosaic may mix dates; the dates
        actually used are recorded in the dataset tags.
    provider : str
        ``"google"`` for Google Earth historical imagery, ``"esri"`` for the
        Esri World Imagery Wayback archive.
    esri_wayback_release_id : str | None
        Force imagery from the exact Esri Wayback release with this stable
        identifier, for example ``"WB_2026_R03"``. Requires ``provider="esri"``
        and ``date=None``, and does not use ``date_match``.

        A release is a mosaic of imagery captured on many dates, so no value of
        ``date`` reproduces one: ask for a single capture date over an area that
        straddles a seam and the rest comes back masked. This selector also
        keeps tiles whose capture metadata is missing, which ``date`` cannot
        reach at all for Esri.

        To go from a *service* date to a release, use
        :func:`esri_mosaic_as_of` and pass the ``release_id`` it reports.
    cache_dir : str | os.PathLike[str] | None
        On-disk response cache. Pass ``None`` to disable caching.
    max_tiles : int
        Maximum number of tile-grid cells spanned by the AOI's bounding box.
        Default 1,000.

    Returns
    -------
    rasterio.DatasetReader
        An open, in-memory 3-band uint8 RGB GeoTIFF.  The CRS is EPSG:4326 for
        ``provider="google"`` and EPSG:3857 for ``provider="esri"``, matching
        each provider's native tile grid so no resampling occurs — reproject
        with ``rasterio.warp`` if you need something else.

        Tiles with no imagery are left black and excluded by the dataset mask,
        so use ``ds.dataset_mask()`` or ``ds.read(masked=True)`` to ignore gaps.
        Dataset tags always carry ``selection_mode``, ``zoom``, ``provider``,
        ``tiles_total``, ``tiles_missing`` and
        ``tiles_capture_date_unknown``. ``dates`` contains comma-separated
        capture dates when at least one is known. Capture-date mode also
        records ``target_date`` and ``date_match``. Release mode instead
        records the resolved Esri release ID, catalogue date and title. Esri
        downloads also record the distinct source records used as compact JSON
        in ``esri_source_metadata`` when that metadata is available; map-level
        values in that tag never alter the requested zoom.

    See Also
    --------
    esri_mosaic_as_of : Resolve a *service* date to a release, and see which
        capture dates that release displays where. Its ``release_id`` is what
        ``esri_wayback_release_id`` takes.

    Raises
    ------
    ValueError
        If the AOI selects no tiles, or no imagery matches the request.
    """
    _validate_zoom(zoom, provider)
    aoi = normalize_aoi(aoi)
    target = _validate_download_selection(provider, date, date_match, esri_wayback_release_id)
    backend, client = _backend(provider, cache_dir)

    try:
        release_layer = _resolve_release(backend, esri_wayback_release_id)
        grid = backend.grid
        tiles = grid.tiles(aoi, zoom, max_tiles)
        if not tiles:
            raise ValueError("The area of interest selects no tiles")

        left, top, right, bottom = grid.pixel_bounds(aoi, zoom)
        width, height = right - left, bottom - top

        rgb = np.zeros((3, height, width), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        used_dates: set[_dt.date] = set()
        used_esri_sources: set = set()
        missing = 0
        unknown_capture_dates = 0
        lock = threading.Lock()

        def work(tile) -> None:
            nonlocal missing, unknown_capture_dates
            if release_layer is not None:
                result = _fetch_release_tile(backend, tile, release_layer)
            else:
                assert target is not None
                result = _fetch_tile(backend, tile, target, date_match)
            if result is None:
                with lock:
                    missing += 1
                return
            arr, candidate = result
            tile_date = candidate.date
            ox, oy = grid.tile_origin(tile)
            x0, x1 = max(ox, left), min(ox + TILE_PX, right)
            y0, y1 = max(oy, top), min(oy + TILE_PX, bottom)
            if x0 >= x1 or y0 >= y1:
                return
            with lock:
                rgb[:, y0 - top : y1 - top, x0 - left : x1 - left] = arr[
                    :, y0 - oy : y1 - oy, x0 - ox : x1 - ox
                ]
                mask[y0 - top : y1 - top, x0 - left : x1 - left] = 255
                if tile_date is not None:
                    used_dates.add(tile_date)
                else:
                    unknown_capture_dates += 1
                source = getattr(candidate, "source", None)
                if source is not None:
                    used_esri_sources.add(source)

        with ThreadPoolExecutor(max_workers=workers_for(provider, len(tiles))) as pool:
            list(pool.map(work, tiles))

        if not mask.any():
            if release_layer is not None:
                raise ValueError(
                    f"No imagery found in Esri Wayback release "
                    f"{release_layer.date.isoformat()} over this area at zoom {zoom}"
                )
            raise ValueError(
                f"No imagery found for {target} over this area at zoom {zoom} "
                f"with date_match={date_match!r}"
            )

        memfile = MemoryFile()
        with memfile.open(
            driver="GTiff",
            width=width,
            height=height,
            count=3,
            dtype="uint8",
            crs=grid.crs,
            transform=grid.transform(left, top, zoom),
            photometric="RGB",
        ) as dst:
            dst.write(rgb)
            dst.write_mask(mask)
            tags = dict(
                dates=",".join(d.isoformat() for d in sorted(used_dates)),
                zoom=str(zoom),
                provider=provider,
                tiles_total=str(len(tiles)),
                tiles_missing=str(missing),
                tiles_capture_date_unknown=str(unknown_capture_dates),
            )
            if release_layer is not None:
                tags.update(
                    selection_mode="esri-wayback-release",
                    esri_wayback_release_id=release_layer.identifier,
                    esri_wayback_catalogue_date=release_layer.date.isoformat(),
                    esri_wayback_release_title=release_layer.title,
                )
            else:
                assert target is not None
                tags.update(
                    selection_mode="capture-date",
                    target_date=target.isoformat(),
                    date_match=date_match,
                )
            if provider == "esri" and used_esri_sources:
                tags["esri_source_metadata"] = json.dumps(
                    [_source_as_dict(source) for source in sorted(used_esri_sources, key=repr)],
                    separators=(",", ":"),
                )
            dst.update_tags(**tags)

        dataset = memfile.open()
        # Keep the backing MemoryFile alive for as long as the reader is used.
        dataset._old_imagery_memfile = memfile
        return dataset
    finally:
        client.close()


def _fetch_tile(backend, tile, target: _dt.date, date_match: str):
    """Return ``(array, selected candidate)`` for the best match, or ``None``."""
    candidates = backend.dated_tiles(tile)
    if not candidates:
        return None

    for candidate in sort_by_nearest_date(candidates, target, date_match):
        try:
            raw = backend.download_tile_image(candidate)
        except (RequestFailed, OSError, ValueError):
            continue  # try the next-nearest date
        arr = _decode_image(raw)
        if arr is not None:
            return arr, candidate
    return None


def _fetch_release_tile(backend, tile, release_layer):
    """Return ``(array, candidate)`` from one exact Esri release, or ``None``."""
    candidate = backend.tile_at_release(tile, release_layer)
    try:
        raw = backend.download_tile_image(candidate)
    except (RequestFailed, OSError, ValueError):
        return None
    arr = _decode_image(raw)
    return None if arr is None else (arr, candidate)


def _source_as_dict(source) -> dict:
    """JSON-safe representation used in raster provenance tags."""
    return {
        "provider": source.provider,
        "description": source.description,
        "resolution_m": source.resolution_m,
        "accuracy_m": source.accuracy_m,
        "min_map_level": source.min_map_level,
        "max_map_level": source.max_map_level,
    }


def _decode_image(raw: bytes) -> np.ndarray | None:
    """Decode tile bytes into a ``(3, TILE_PX, TILE_PX)`` uint8 array."""
    try:
        with warnings.catch_warnings():
            # Bare tiles carry no geotransform; we supply one when mosaicking.
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with MemoryFile(raw, ext=".jpg") as mem, mem.open() as src:
                arr = src.read()
    except Exception:  # noqa: BLE001
        # Deliberately broad: a tile's bytes come from a remote service and may
        # be truncated, HTML, or a format GDAL rejects in driver-specific ways.
        # An undecodable tile is left black and masked out, per the README, so
        # one bad tile must not abort the mosaic.
        return None

    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    elif arr.shape[0] > 3:
        arr = arr[:3]
    if arr.shape[0] != 3 or arr.shape[1:] != (TILE_PX, TILE_PX):
        return None
    return arr
