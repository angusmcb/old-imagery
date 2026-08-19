"""GeoPackage output for provider-native image tiles and local overviews."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import sqlite3
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import rasterio
from rasterio.enums import Resampling
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


def _overall_metadata(
    tiles,
    selection: Mapping[str, object],
    geopackage_zoom: int,
    overview_factors: tuple[int, ...],
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
        "overviews": {"resampling": "lanczos", "factors": overview_factors},
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
) -> Path:
    """Write native tiles and local lower-resolution overviews atomically."""
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
    overview_factors = _overview_factors(content_width, content_height, geopackage_zoom)

    destination = Path(output)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    if not destination.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {destination.parent}")

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
                    _overall_metadata(tiles, selection, geopackage_zoom, overview_factors),
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
            connection.commit()
            foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_key_errors:
                raise ValueError("GeoPackage validation found a broken foreign key")
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity != ("ok",):
                raise ValueError("GeoPackage SQLite integrity check failed")
        finally:
            connection.close()

        # Build lower-resolution raster tile matrices from the unchanged native
        # tiles. This is local resampling only: no provider payloads are
        # replaced, and no additional network requests are made.
        if overview_factors:
            with rasterio.open(temporary, "r+") as dataset:
                dataset.build_overviews(overview_factors, Resampling.lanczos)
                dataset.update_tags(ns="rio_overview", resampling="lanczos")

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
