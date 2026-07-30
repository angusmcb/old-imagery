"""Public API: :func:`availability` and :func:`download`."""

from __future__ import annotations

import datetime as _dt
import os
import threading
import warnings
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep

from ._concurrency import workers_for
from ._dbroot import Database, DbRoot
from ._http import DEFAULT_CACHE_DIR, CachedHttpClient, RequestFailed
from ._region import MAX_TILES, TILE_PX, dissolve, normalize_aoi, sort_by_nearest_date

WGS84 = "EPSG:4326"
MERCATOR = "EPSG:3857"

DateLike = _dt.date | str

# Closed sets for the three string-valued options on the public functions.
#
# These constrain callers under a type checker only. Every one of them is still
# validated at runtime -- `_backend` for Provider, `_use_region_query` for
# Method, `sort_by_nearest_date` for DateMatch -- because most callers run
# unchecked, and a typo should raise rather than behave arbitrarily. For that
# reason the internal helpers keep plain `str` parameters: narrowing them to
# these aliases would make their own validation branches statically
# unreachable, and mypy runs here with warn_unreachable.
Provider = Literal["google", "esri"]
Method = Literal["auto", "per-tile", "region"]
DateMatch = Literal["closest", "exact", "before", "after"]

AVAILABILITY_COLUMNS = ["date", "n_tiles", "coverage", "complete", "providers", "geometry"]
ESRI_MOSAIC_COLUMNS = ["zoom", "date", "area_fraction", "release_id", "geometry"]

# Esri exposes two ways to resolve availability, and which wins is not obvious
# from request counts: per-tile probing issues far more but small requests,
# while each region query hits a slow metadata service. Measured against the
# live service (cold cache, 16 requests in flight, seconds):
#
#     tiles      4     12     30     72
#     per-tile  30.9   32.8   62.0   67.9
#     region    63.3   96.7  143.4   40.7
#
# So per-tile wins by 2-3x on small areas and region wins on large ones, with
# the crossover somewhere in the tens of tiles.
#
# Treat this as an order-of-magnitude result, not a calibration. Run-to-run
# variance on an identical AOI reached 2.3x, which is the same size as the
# effect being measured. A later attempt to re-measure at the 10-wide setting
# _concurrency now uses produced 1.8x *slower* for per-tile and 1.9x *faster*
# for region on the same AOI -- two opposite conclusions of equal magnitude,
# i.e. noise, on a link that was known to be degraded at the time. The threshold
# is therefore a rough midpoint that has never been sharply located, and is
# worth revisiting only with many repeats on a stable connection. Override it
# for a specific call by passing `method` explicitly.
ESRI_REGION_QUERY_MIN_TILES = 50

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
    aoi: BaseGeometry,
    zoom: int,
    *,
    min_date: DateLike | None = None,
    max_date: DateLike | None = None,
    provider: Provider = "google",
    method: Method = "auto",
    cache_dir: str | os.PathLike[str] | None = DEFAULT_CACHE_DIR,
    max_tiles: int = MAX_TILES,
) -> gpd.GeoDataFrame:
    """Find which imagery capture dates are available over an area.

    Parameters
    ----------
    aoi : shapely.geometry.base.BaseGeometry
        Area of interest as a shapely geometry in EPSG:4326 (lon/lat degrees).
    zoom : int
        Tile zoom level.  Availability is reported at tile granularity, so a
        higher zoom gives finer date boundaries and costs more requests.
    min_date, max_date : datetime.date | str | None
        Optional inclusive bounds on capture date.
    provider : str
        ``"google"`` for Google Earth historical imagery, ``"esri"`` for the
        Esri World Imagery Wayback archive.
    method : str
        How Esri availability is resolved. ``"auto"`` (default) picks
        ``"region"`` for areas of at least ``ESRI_REGION_QUERY_MIN_TILES``
        tiles and ``"per-tile"`` below that. ``"region"`` returns
        provider-reported capture footprints and is faster on large areas but
        2-3x slower on small ones; ``"per-tile"`` reports coverage quantised to
        whole tiles. Ignored for Google, which only has a per-tile path.
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
        ``n_tiles``
            Number of AOI tiles carrying imagery from that date.
        ``coverage``
            ``n_tiles`` as a fraction of the AOI's tiles.
        ``complete``
            True when that date covers every AOI tile (equivalent to
            ``coverage == 1.0``). This is a tile-resolution measure, not a
            guarantee that every point in the AOI has imagery.
        ``providers``
            Google imagery provider names or Esri Wayback release titles,
            where known.
        ``geometry``
            The covered area, clipped to the AOI.

        All dates are **image capture dates**, never a provider's publication
        or release date.

        Google Earth tiles whose imagery carries no capture date (a provider's
        undated default imagery) are excluded, matching upstream behaviour.

        ``gdf.attrs`` records ``zoom``, ``provider``, ``n_aoi_tiles`` and
        ``method`` -- ``"per-tile"``, ``"region-query"``, or ``"none"`` when the
        AOI selects no tiles.

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

        if _use_region_query(backend, len(tiles), method):
            return _availability_by_region(
                backend, aoi, tiles, zoom, provider, min_d, max_d
            )

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

        rows = []
        for date in sorted(by_date, reverse=True):
            date_tiles = by_date[date]
            names = sorted(
                {n for n in (backend.provider_copyright(p) for p in providers[date]) if n}
            )
            rows.append(
                {
                    "date": date,
                    "n_tiles": len(date_tiles),
                    "coverage": len(date_tiles) / len(tiles),
                    "complete": len(date_tiles) == len(tiles),
                    "providers": ", ".join(names),
                    "geometry": dissolve(date_tiles).intersection(aoi),
                }
            )

        gdf = _finish_availability(rows, zoom, provider, len(tiles), "per-tile")
        return gdf
    finally:
        client.close()


def _use_region_query(backend, n_tiles: int, method: str = "auto") -> bool:
    if method not in ("auto", "region", "per-tile"):
        raise ValueError(f"Unknown method {method!r}; expected 'auto', 'region' or 'per-tile'")
    if method == "per-tile":
        return False
    supported = hasattr(backend, "dated_regions")
    if method == "region":
        if not supported:
            raise ValueError("method='region' is only available for provider='esri'")
        return True
    return supported and n_tiles >= ESRI_REGION_QUERY_MIN_TILES


def _availability_by_region(
    backend, aoi, tiles, zoom, provider, min_d, max_d
) -> gpd.GeoDataFrame:
    """Availability from region-wide capture-footprint queries.

    Produces the same columns as the per-tile path.  ``geometry`` is the
    provider-reported capture footprint clipped to the AOI rather than a union
    of whole tiles.

    ``n_tiles`` / ``coverage`` / ``complete`` stay tile-based, but note that
    "tile-based" does not mean the same thing here as in the per-tile path: a
    tile counts when a footprint *intersects* its extent, whereas the per-tile
    path counts a tile only when the provider reports imagery for the whole
    tile.  A footprint clipping a sliver off every AOI tile therefore reports
    full coverage over near-zero area.  Compare ``geometry`` areas, not
    ``coverage``, when comparing the two paths.
    """
    regions = backend.dated_regions(aoi, zoom, min_date=min_d, max_date=max_d)
    if not regions:
        return _empty_availability(zoom, provider, len(tiles), "region-query")

    by_date: dict[_dt.date, list] = defaultdict(list)
    titles: dict[_dt.date, set[str]] = defaultdict(set)
    for date, geom, title in regions:
        by_date[date].append(geom)
        if title:
            titles[date].add(title)

    prepared_tiles = [(tile, box(*tile.bounds_wgs84)) for tile in tiles]

    geometry_by_date: dict[_dt.date, BaseGeometry] = {}
    tiles_by_date: dict[_dt.date, set] = {}
    for date, geoms in by_date.items():
        clipped = unary_union(geoms).intersection(aoi)
        if clipped.is_empty:
            continue
        prepared = prep(clipped)
        covered = {tile for tile, extent in prepared_tiles if prepared.intersects(extent)}
        if not covered:
            continue
        geometry_by_date[date] = clipped
        tiles_by_date[date] = covered

    if not geometry_by_date:
        return _empty_availability(zoom, provider, len(tiles), "region-query")

    rows = [
        {
            "date": date,
            "n_tiles": len(tiles_by_date[date]),
            "coverage": len(tiles_by_date[date]) / len(tiles),
            "complete": len(tiles_by_date[date]) == len(tiles),
            "providers": ", ".join(sorted(titles.get(date, ()))),
            "geometry": geometry_by_date[date],
        }
        for date in sorted(geometry_by_date, reverse=True)
    ]
    return _finish_availability(rows, zoom, provider, len(tiles), "region-query")


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
# esri_mosaic_as_of
# --------------------------------------------------------------------------
def esri_mosaic_as_of(
    aoi: BaseGeometry,
    zoom: int | Sequence[int],
    as_of_date: DateLike,
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
    aoi : shapely.geometry.base.BaseGeometry
        Area of interest as a shapely geometry in EPSG:4326 (lon/lat degrees).
    zoom : int | Sequence[int]
        One zoom level, or several.  This is not merely a resolution knob: Esri
        composes the mosaic per scale and publishes metadata per scale, so the
        same ground in the same release can carry a different capture date at
        different zooms.  Pass several to see that.  Zooms of 10 and below all
        resolve to the same metadata layer and so return identical geometry.
    as_of_date : datetime.date | str
        The date to look at the archive on.  The latest release published on or
        before this date is used; it is a **publication** date, not a capture
        date, and no capture-date matching happens.
    cache_dir : str | os.PathLike[str] | None
        On-disk response cache. Pass ``None`` to disable caching.
    max_footprints : int
        Reject areas where the release publishes more capture footprints than
        this at some zoom. Default 500. Footprint count, not tile count, is what
        this call costs.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per ``(zoom, capture date)``, ordered by zoom then newest date
        first, in EPSG:4326:

        ``zoom``
            The zoom the row was resolved at.
        ``date``
            The **capture date** of the imagery displayed in this area.
        ``area_fraction``
            This row's share of the AOI, as a planar area ratio computed in
            EPSG:3857.  Mercator's area distortion largely cancels between the
            row and the AOI over a modest latitude span, but this is not a true
            area ratio for a tall AOI -- reproject ``geometry`` yourself if you
            need one.  ``nan`` for a zero-area (line or point) AOI.
        ``release_id``
            Stable catalogue identifier of the resolved release, for example
            ``"WB_2026_R03"``.  Constant for the whole frame, and a column
            rather than only an attr so concatenating two snapshots cannot lose
            track of which release each row came from.
        ``geometry``
            The area displaying that capture date, clipped to the AOI.  Real
            capture-footprint boundaries, not tile edges.

        ``gdf.attrs`` records ``release_id``, ``release_date`` (the publication
        date), ``release_title``, ``as_of_date`` and ``zooms``.

        Footprints sharing a zoom and capture date are dissolved into one row.
        Ground the release publishes no footprint metadata for is absent
        entirely, so ``area_fraction`` need not sum to 1 across a zoom.

    Raises
    ------
    ValueError
        If ``as_of_date`` precedes the archive, a zoom is out of range, or the
        release publishes more than ``max_footprints`` footprints here.
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
    as_of = _as_date(as_of_date)
    if as_of is None:
        raise ValueError("as_of_date is required")

    backend, client = _backend("esri", cache_dir)
    try:
        layer = backend.release_on_or_before(as_of)

        def work(z: int) -> list[tuple[int, _dt.date, BaseGeometry]]:
            footprints = backend.release_footprints(
                layer, aoi, z, max_footprints=max_footprints
            )
            by_date: dict[_dt.date, list] = defaultdict(list)
            for date, geom in footprints:
                by_date[date].append(geom)

            out = []
            for date, geoms in by_date.items():
                clipped = unary_union(geoms).intersection(aoi)
                if not clipped.is_empty:
                    out.append((z, date, clipped))
            return out

        # One worker per zoom, capped: release_footprints issues its requests
        # serially, so this bounds requests in flight against Wayback to the
        # measured cap rather than multiplying it by the number of zooms.
        with ThreadPoolExecutor(max_workers=workers_for("esri", len(zooms))) as pool:
            found = [row for rows in pool.map(work, zooms) for row in rows]
    finally:
        client.close()

    found.sort(key=lambda row: (row[0], -row[1].toordinal()))
    fractions = _area_fractions(aoi, [geom for _z, _d, geom in found])
    rows = [
        {
            "zoom": z,
            "date": date,
            "area_fraction": fraction,
            "release_id": layer.identifier,
            "geometry": geom,
        }
        for (z, date, geom), fraction in zip(found, fractions, strict=True)
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
        as_of_date=as_of,
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


def _area_fractions(aoi: BaseGeometry, geoms: list[BaseGeometry]) -> list[float]:
    """Each geometry's share of the AOI, as planar areas in EPSG:3857."""
    if not geoms:
        return []
    areas = gpd.GeoSeries([aoi, *geoms], crs=WGS84).to_crs(MERCATOR).area.to_numpy()
    aoi_area = float(areas[0])
    if aoi_area <= 0.0:
        # A line or point AOI: every clip is zero-area too, so the ratio is
        # genuinely undefined rather than zero.
        return [float("nan")] * len(geoms)
    return [float(area) / aoi_area for area in areas[1:]]


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------
def download(
    aoi: BaseGeometry,
    zoom: int,
    date: DateLike | None = None,
    *,
    date_match: DateMatch = "closest",
    provider: Provider = "google",
    esri_wayback_release_id: str | None = None,
    cache_dir: str | os.PathLike[str] | None = DEFAULT_CACHE_DIR,
    max_tiles: int = MAX_TILES,
) -> rasterio.DatasetReader:
    """Download and mosaic historical imagery over an area.

    Parameters
    ----------
    aoi : shapely.geometry.base.BaseGeometry
        Area of interest as a shapely geometry in EPSG:4326 (lon/lat degrees).
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
        records the resolved Esri release ID, catalogue date and title.

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
    target = _as_date(date)
    if esri_wayback_release_id is not None:
        if provider != "esri":
            raise ValueError(
                "esri_wayback_release_id requires provider='esri'; "
                "it selects a publication snapshot, not a capture date"
            )
        if target is not None:
            raise ValueError(
                "Choose either date (capture-date mode) or "
                "esri_wayback_release_id, not both"
            )
        if date_match != "closest":
            raise ValueError(
                "date_match cannot be set with esri_wayback_release_id; "
                "the exact release layer is used without date fallback"
            )
    elif target is None:
        raise ValueError("A target date is required")
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
            arr, tile_date = result
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
            dst.update_tags(**tags)

        dataset = memfile.open()
        # Keep the backing MemoryFile alive for as long as the reader is used.
        dataset._old_imagery_memfile = memfile
        return dataset
    finally:
        client.close()


def _fetch_tile(backend, tile, target: _dt.date, date_match: str):
    """Return ``(array, date)`` for the best-matching image, or ``None``."""
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
            return arr, candidate.date
    return None


def _fetch_release_tile(backend, tile, release_layer):
    """Return ``(array, capture_date)`` from one exact Esri release, or ``None``."""
    candidate = backend.tile_at_release(tile, release_layer)
    try:
        raw = backend.download_tile_image(candidate)
    except (RequestFailed, OSError, ValueError):
        return None
    arr = _decode_image(raw)
    return None if arr is None else (arr, candidate.date)


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
