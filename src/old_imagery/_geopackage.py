"""GeoPackage output for provider-native image tiles and local overviews."""

from __future__ import annotations

import datetime as dt
import io
import json
import math
import os
import sqlite3
import tempfile
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from ._region import MERCATOR_EQUATOR, TILE_PX

_RESERVED_TABLES = {
    "gpkg_contents",
    "gpkg_geometry_columns",
    "gpkg_ogr_contents",
    "gpkg_spatial_ref_sys",
    "gpkg_tile_matrix",
    "gpkg_tile_matrix_set",
    "old_imagery_metadata",
    "old_imagery_tile_metadata",
    "sqlite_sequence",
}

_SQLITE_PAGE_SIZE = 16 * 1024
_SQLITE_CACHE_SIZE_KIB = 256 * 1024
_OVERVIEW_JPEG_QUALITY = 60


def _quoted_identifier(value: str) -> str:
    """Quote a SQLite identifier after rejecting unsafe/reserved names."""
    if not value or "\x00" in value:
        raise ValueError("table_name must be a non-empty SQLite identifier")
    if value.lower().startswith(("gpkg_", "sqlite_")) or value.lower() in _RESERVED_TABLES:
        raise ValueError(f"table_name {value!r} is reserved")
    return '"' + value.replace('"', '""') + '"'


def _tile_address(tile) -> tuple[int, int, int]:
    """Return the GeoPackage zoom, column and north-origin row."""
    if tile.tile_scheme == "WebMercatorQuad":
        return tile.zoom, tile.column, tile.row
    if tile.tile_scheme == "GoogleEarthKeyhole":
        if tile.zoom < 2:
            raise ValueError(
                "Google zooms 0 and 1 cannot be represented as complete CRS84 tiles "
                "without cutting or combining native images; use zoom 2 or greater"
            )
        n = 1 << tile.zoom
        row = (n - 1 - tile.row) - n // 4
        if not (0 <= row < n // 2):
            raise ValueError(
                f"Google tile z{tile.zoom}/{tile.column}/{tile.row} lies outside the "
                "CRS84 latitude domain"
            )
        return tile.zoom - 1, tile.column, row
    raise ValueError(f"Unsupported native tile scheme {tile.tile_scheme!r}")


def _grid(tile) -> tuple[str, str, int, int, float, float, float]:
    """Return scheme, CRS, dimensions, origin, and pixel size for one level."""
    if tile.tile_scheme == "WebMercatorQuad":
        n = 1 << tile.zoom
        return (
            "GoogleMapsCompatible",
            "EPSG:3857",
            n * TILE_PX,
            n * TILE_PX,
            -MERCATOR_EQUATOR / 2.0,
            MERCATOR_EQUATOR / 2.0,
            MERCATOR_EQUATOR / (n * TILE_PX),
        )
    if tile.tile_scheme == "GoogleEarthKeyhole":
        if tile.zoom < 2:
            # Keep the public error independent of whether the tile list is
            # inspected here or in _tile_address first.
            _tile_address(tile)
        n = 1 << tile.zoom
        return (
            "InspireCRS84Quad",
            "EPSG:4326",
            n * TILE_PX,
            n // 2 * TILE_PX,
            -180.0,
            90.0,
            360.0 / (n * TILE_PX),
        )
    raise ValueError(f"Unsupported native tile scheme {tile.tile_scheme!r}")


def _content_bounds(addresses, pixel_size: float, origin_x: float, origin_y: float):
    zoom, min_col, min_row = addresses[0]
    max_col = min_col
    max_row = min_row
    for item_zoom, column, row in addresses[1:]:
        if item_zoom != zoom:
            raise ValueError("A GeoPackage write currently accepts exactly one zoom level")
        min_col = min(min_col, column)
        max_col = max(max_col, column)
        min_row = min(min_row, row)
        max_row = max(max_row, row)
    span = TILE_PX * pixel_size
    return (
        origin_x + min_col * span,
        origin_y - (max_row + 1) * span,
        origin_x + (max_col + 1) * span,
        origin_y - min_row * span,
    )


def _content_shape(addresses) -> tuple[int, int]:
    """Return the pixel shape of the tile-bounding rectangle."""
    _, min_col, min_row = addresses[0]
    max_col = min_col
    max_row = min_row
    for item_zoom, column, row in addresses[1:]:
        if item_zoom != addresses[0][0]:
            raise ValueError("A GeoPackage write currently accepts exactly one zoom level")
        min_col = min(min_col, column)
        max_col = max(max_col, column)
        min_row = min(min_row, row)
        max_row = max(max_row, row)
    return (max_col - min_col + 1) * TILE_PX, (max_row - min_row + 1) * TILE_PX


def _overview_factors(width: int, height: int, geopackage_zoom: int) -> tuple[int, ...]:
    """Return useful power-of-two overview factors for a tile pyramid."""
    factors: list[int] = []
    previous_size: tuple[int, int] | None = None
    for shift in range(1, geopackage_zoom + 1):
        factor = 1 << shift
        size = (max(1, math.ceil(width / factor)), max(1, math.ceil(height / factor)))
        if size == previous_size:
            break
        factors.append(factor)
        previous_size = size
        if size == (1, 1):
            break
    return tuple(factors)


def _positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _cgroup_cpu_limit() -> int | None:
    """Return a whole-CPU cgroup quota when one is visible."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass

    try:
        quota_value = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period_value = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota_value > 0:
            return max(1, quota_value // period_value)
    except (OSError, ValueError):
        pass
    return None


def _available_cpu_count() -> int:
    """Return CPUs available to this task, not CPUs installed on its host."""
    limits: list[int] = []

    # Slurm may leave the process affinity mask broad on some clusters. Its
    # per-task allocation is therefore an independent upper bound.
    slurm_limit = _positive_int(os.environ.get("SLURM_CPUS_PER_TASK"))
    if slurm_limit is not None:
        limits.append(slurm_limit)

    if hasattr(os, "sched_getaffinity"):
        with suppress(OSError):
            limits.append(len(os.sched_getaffinity(0)))

    cgroup_limit = _cgroup_cpu_limit()
    if cgroup_limit is not None:
        limits.append(cgroup_limit)

    process_cpu_count = getattr(os, "process_cpu_count", None)
    detected = process_cpu_count() if process_cpu_count is not None else os.cpu_count()
    if detected is not None and detected > 0:
        limits.append(detected)
    return max(1, min(limits)) if limits else 1


def _rgba_tile(payload: bytes) -> np.ndarray:
    """Decode one native or overview payload to an RGBA uint8 tile."""
    with Image.open(io.BytesIO(payload)) as image:
        if image.size != (TILE_PX, TILE_PX):
            raise ValueError(
                f"overview source payload is {image.width}x{image.height}; "
                f"expected {TILE_PX}x{TILE_PX}"
            )
        data = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    if data.shape != (TILE_PX, TILE_PX, 4):
        raise ValueError(f"decoded overview source has unexpected shape {data.shape!r}")
    return data.transpose(2, 0, 1)


def _average_2x2(rgba: np.ndarray) -> np.ndarray:
    """Average a tile by two, excluding transparent pixels from RGB values."""
    values = rgba.astype(np.float32)
    top_left = values[:, 0::2, 0::2]
    top_right = values[:, 0::2, 1::2]
    bottom_left = values[:, 1::2, 0::2]
    bottom_right = values[:, 1::2, 1::2]
    weights = (top_left[3] + top_right[3] + bottom_left[3] + bottom_right[3]) / 255.0
    weighted_rgb = (
        top_left[:3] * top_left[3]
        + top_right[:3] * top_right[3]
        + bottom_left[:3] * bottom_left[3]
        + bottom_right[:3] * bottom_right[3]
    ) / 255.0

    result = np.zeros((4, TILE_PX // 2, TILE_PX // 2), dtype=np.uint8)
    rgb = np.zeros_like(weighted_rgb)
    np.divide(weighted_rgb, weights[None, ...], out=rgb, where=weights[None, ...] > 0)
    result[:3] = np.rint(rgb).astype(np.uint8)
    result[3] = np.rint(weights * (255.0 / 4.0)).astype(np.uint8)
    return result


def _encode_rgba_tile(rgba: np.ndarray) -> bytes:
    """Encode one derived overview tile as JPEG or transparent PNG."""
    opaque = bool(np.all(rgba[3] == 255))
    data = rgba[:3] if opaque else rgba
    image = Image.fromarray(data.transpose(1, 2, 0))
    output = io.BytesIO()
    if opaque:
        image.save(output, format="JPEG", quality=_OVERVIEW_JPEG_QUALITY)
    else:
        image.save(output, format="PNG")
    return output.getvalue()


def _build_overview_parent(item) -> tuple[int, int, bytes]:
    """Build one parent tile in a worker thread."""
    (parent_column, parent_row), children = item
    parent = np.zeros((4, TILE_PX, TILE_PX), dtype=np.uint8)
    for column, row, payload in children:
        reduced = _average_2x2(_rgba_tile(payload))
        left = (column & 1) * (TILE_PX // 2)
        top = (row & 1) * (TILE_PX // 2)
        parent[:, top : top + TILE_PX // 2, left : left + TILE_PX // 2] = reduced
    return parent_column, parent_row, _encode_rgba_tile(parent)


def _bounded_parent_map(
    pool: ThreadPoolExecutor,
    items: Iterable[tuple[tuple[int, int], list[tuple[int, int, bytes]]]],
    workers: int,
) -> Iterator[tuple[int, int, bytes]]:
    """Map in order without retaining a future and result for every tile."""
    iterator = iter(items)
    pending: deque[Future[tuple[int, int, bytes]]] = deque()
    for _ in range(workers * 2):
        try:
            item = next(iterator)
        except StopIteration:
            break
        pending.append(pool.submit(_build_overview_parent, item))

    while pending:
        yield pending.popleft().result()
        try:
            item = next(iterator)
        except StopIteration:
            continue
        pending.append(pool.submit(_build_overview_parent, item))


def _build_sparse_overviews(
    connection: sqlite3.Connection,
    quoted_table: str,
    addresses: list[tuple[int, int, int]],
    payloads: list[bytes],
    overview_factors: tuple[int, ...],
    workers: int,
) -> None:
    """Build overview rows only for parents of populated child tiles.

    GDAL's dataset-level overview builder has to inspect the whole raster
    canvas, which is particularly expensive when the native tile set is
    sparse. The tile matrix is a quadtree, so a two-to-one pass can derive the
    same pyramid while touching only occupied child tiles. Transparent pixels
    represent gaps and are excluded from the average.
    """
    if not overview_factors:
        return

    zoom = addresses[0][0]
    current = {
        (column, row): payload
        for (_, column, row), payload in zip(addresses, payloads, strict=True)
    }
    insert_sql = (
        f"INSERT INTO {quoted_table} "
        "(zoom_level, tile_column, tile_row, tile_data) VALUES (?, ?, ?, ?)"
    )
    pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for level, factor in enumerate(overview_factors, start=1):
            expected_factor = 1 << level
            if factor != expected_factor:
                raise ValueError("Overview factors must be consecutive powers of two")
            parents: dict[tuple[int, int], list[tuple[int, int, bytes]]] = {}
            for (column, row), payload in current.items():
                parents.setdefault((column // 2, row // 2), []).append((column, row, payload))

            next_level: dict[tuple[int, int], bytes] = {}
            try:
                items = sorted(parents.items())
                results = (
                    _bounded_parent_map(pool, items, workers)
                    if pool is not None
                    else map(_build_overview_parent, items)
                )
                for parent_column, parent_row, encoded in results:
                    connection.execute(
                        insert_sql,
                        (
                            zoom - level,
                            parent_column,
                            parent_row,
                            sqlite3.Binary(encoded),
                        ),
                    )
                    next_level[(parent_column, parent_row)] = encoded
            except Exception as error:
                raise ValueError("Could not build an overview tile") from error
            current = next_level
    finally:
        if pool is not None:
            pool.shutdown()


def _overall_metadata(
    tiles,
    selection: Mapping[str, object],
    geopackage_zoom: int,
    overview_factors: tuple[int, ...],
    build_overviews: bool,
) -> str:
    first = tiles[0]
    values = {
        "schema_version": 1,
        "provider": first.provider,
        "native_tile_scheme": first.tile_scheme,
        "native_zoom": first.zoom,
        "geopackage_zoom": geopackage_zoom,
        "tile_count": len(tiles),
        "selection": selection,
        "overviews": {
            "enabled": build_overviews,
            "resampling": "average",
            "factors": overview_factors,
            "jpeg_quality": _OVERVIEW_JPEG_QUALITY,
        },
    }
    return json.dumps(values, separators=(",", ":"), default=str)


def _tile_metadata(tile, address: tuple[int, int, int]) -> str:
    values = {
        "geopackage_zoom": address[0],
        "geopackage_column": address[1],
        "geopackage_row": address[2],
        "native_zoom": tile.zoom,
        "native_column": tile.column,
        "native_row": tile.row,
        "capture_date": tile.capture_date_at_center,
        "source_metadata": (
            asdict(tile.source_metadata_at_center) if tile.source_metadata_at_center else None
        ),
        "release_id": tile.release_id,
        "release_date": tile.release_date,
        "release_title": tile.release_title,
    }
    return json.dumps(values, separators=(",", ":"), default=str)


def write_geopackage(
    tiles,
    output: str | os.PathLike[str],
    *,
    table_name: str,
    selection: Mapping[str, object],
    overwrite: bool,
    build_overviews: bool,
) -> Path:
    """Write native tiles and optional lower-resolution overviews atomically."""
    if not tiles:
        raise ValueError("Cannot create a GeoPackage without tiles")
    quoted_table = _quoted_identifier(table_name)
    first = tiles[0]
    if any(
        (tile.provider, tile.tile_scheme, tile.zoom)
        != (first.provider, first.tile_scheme, first.zoom)
        for tile in tiles
    ):
        raise ValueError("All GeoPackage tiles must share one provider, scheme and zoom")
    formats = {tile.image_format for tile in tiles}
    unsupported = formats - {"jpeg", "png"}
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"GeoPackage output does not support native tile format(s): {names}")

    addresses = [_tile_address(tile) for tile in tiles]
    scheme, crs, width, height, origin_x, origin_y, pixel_size = _grid(first)
    bounds = _content_bounds(addresses, pixel_size, origin_x, origin_y)
    geopackage_zoom = addresses[0][0]
    content_width, content_height = _content_shape(addresses)
    overview_factors = (
        _overview_factors(content_width, content_height, geopackage_zoom) if build_overviews else ()
    )

    destination = Path(output)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    if not destination.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {destination.parent}")
    workers = min(_available_cpu_count(), len(tiles)) if overview_factors else 1

    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.", suffix=".gpkg", dir=destination.parent, delete=False
    ) as temporary_file:
        temporary = Path(temporary_file.name)
    # The GDAL GeoPackage driver creates rather than truncates its target.
    temporary.unlink()
    try:
        # Let GDAL create the normative GeoPackage core tables, constraints,
        # triggers, application ID and version. No raster blocks are written;
        # the unchanged provider payloads are inserted below.
        with rasterio.open(
            temporary,
            "w",
            driver="GPKG",
            RASTER_TABLE=table_name,
            TILING_SCHEME=scheme,
            TILE_FORMAT="JPEG" if formats == {"jpeg"} else "PNG_JPEG",
            # Baseline GeoPackage stores CRS definitions as legacy WKT1.
            # Also write the official CRS-WKT extension: current QGIS builds
            # prefer its unambiguous WKT2 definition for raster layers.
            CRS_WKT_EXTENSION="YES",
            # Provenance is attached through the standard GeoPackage metadata
            # extension. Ad-hoc user tables make QGIS 4.2 discard the CRS of a
            # raster in the same container.
            METADATA_TABLES="YES",
            width=width,
            height=height,
            count=3,
            dtype="uint8",
            crs=crs,
            transform=from_origin(origin_x, origin_y, pixel_size, pixel_size),
        ):
            pass

        connection = sqlite3.connect(temporary)
        try:
            # GDAL has created only the small GeoPackage schema at this point,
            # so changing from SQLite's 4 KiB default costs almost nothing.
            # Larger pages turn large tile blobs into far fewer filesystem
            # writes without the space inflation measured at 32/64 KiB.
            connection.execute(f"PRAGMA page_size = {_SQLITE_PAGE_SIZE}")
            connection.execute("VACUUM")
            # This is an unpublished, reproducible build artifact. Keep its
            # small rollback journal in memory and omit durable syncs; the
            # completed database is checked before it is published.
            connection.execute("PRAGMA journal_mode = MEMORY")
            connection.execute("PRAGMA synchronous = OFF")
            connection.execute("PRAGMA locking_mode = EXCLUSIVE")
            connection.execute(f"PRAGMA cache_size = -{_SQLITE_CACHE_SIZE_KIB}")
            connection.execute("PRAGMA temp_store = MEMORY")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                "UPDATE gpkg_contents SET identifier = ?, description = ?, "
                "last_change = ?, min_x = ?, min_y = ?, max_x = ?, max_y = ? "
                "WHERE table_name = ?",
                (
                    table_name,
                    "Historical imagery downloaded by old-imagery",
                    dt.datetime.now(dt.timezone.utc)
                    .isoformat(timespec="milliseconds")
                    .replace("+00:00", "Z"),
                    *bounds,
                    table_name,
                ),
            )
            connection.executemany(
                f"INSERT INTO {quoted_table} "
                "(zoom_level, tile_column, tile_row, tile_data) VALUES (?, ?, ?, ?)",
                # sqlite3.Binary makes the byte-preservation intent explicit.
                [
                    (*address, sqlite3.Binary(tile.content))
                    for tile, address in zip(tiles, addresses, strict=True)
                ],
            )
            standard_uri = "https://github.com/angusmcb/old-imagery#geopackage-provenance-v1"
            cursor = connection.execute(
                "INSERT INTO gpkg_metadata "
                "(md_scope, md_standard_uri, mime_type, metadata) "
                "VALUES ('dataset', ?, 'application/json', ?)",
                (
                    standard_uri,
                    _overall_metadata(
                        tiles,
                        selection,
                        geopackage_zoom,
                        overview_factors,
                        build_overviews,
                    ),
                ),
            )
            overall_metadata_id = cursor.lastrowid
            connection.execute(
                "INSERT INTO gpkg_metadata_reference "
                "(reference_scope, table_name, md_file_id) VALUES ('table', ?, ?)",
                (table_name, overall_metadata_id),
            )

            tile_ids = {
                (zoom, column, row): tile_id
                for tile_id, zoom, column, row in connection.execute(
                    f"SELECT id, zoom_level, tile_column, tile_row FROM {quoted_table}"
                )
            }
            for tile, address in zip(tiles, addresses, strict=True):
                cursor = connection.execute(
                    "INSERT INTO gpkg_metadata "
                    "(md_scope, md_standard_uri, mime_type, metadata) "
                    "VALUES ('tile', ?, 'application/json', ?)",
                    (standard_uri, _tile_metadata(tile, address)),
                )
                connection.execute(
                    "INSERT INTO gpkg_metadata_reference "
                    "(reference_scope, table_name, row_id_value, md_file_id, md_parent_id) "
                    "VALUES ('row', ?, ?, ?, ?)",
                    (
                        table_name,
                        tile_ids[address],
                        cursor.lastrowid,
                        overall_metadata_id,
                    ),
                )
            # Build lower-resolution raster tile matrices from the unchanged
            # native tiles. CPU-heavy decoding and encoding run concurrently;
            # this connection remains the sole SQLite writer.
            if overview_factors:
                _build_sparse_overviews(
                    connection,
                    quoted_table,
                    addresses,
                    [tile.content for tile in tiles],
                    overview_factors,
                    workers,
                )

            connection.commit()
            foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_key_errors:
                raise ValueError("GeoPackage validation found a broken foreign key")
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity != ("ok",):
                raise ValueError("GeoPackage SQLite integrity check failed")
        finally:
            connection.close()

        # synchronous=OFF avoids repeated network-filesystem barriers while
        # building. One explicit barrier makes the completed database durable
        # before GDAL validates it and the path becomes public.
        with temporary.open("rb") as completed:
            os.fsync(completed.fileno())

        # Reopen through GDAL before publishing the file. This catches schema,
        # georeferencing and driver-compatibility errors that SQLite alone does
        # not know how to identify.
        with rasterio.open(temporary) as dataset:
            if dataset.crs is None or dataset.width <= 0 or dataset.height <= 0:
                raise ValueError("GDAL could not validate the completed GeoPackage")

        if overwrite:
            os.replace(temporary, destination)
        else:
            # A hard link publishes without the check-then-replace race that
            # os.rename has on POSIX. Both paths are in the same directory.
            os.link(temporary, destination)
            temporary.unlink()
        return destination
    finally:
        temporary.unlink(missing_ok=True)
