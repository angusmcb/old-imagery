# old-imagery

Historical aerial imagery from **Google Earth** and the **Esri World Imagery Wayback** archive, as two Python functions.

This is a Python port of the protocol layer of [Mbucari/GEHistoricalImagery](https://github.com/Mbucari/GEHistoricalImagery) (a .NET CLI). It is a library, not a command line tool: areas of interest go in as shapely geometries, availability comes back as a GeoDataFrame, and imagery comes back as an open rasterio dataset.

> ### Before you use this
>
> **This library retrieves imagery. It does not license it.**
>
> Imagery you fetch with `old-imagery` stays the copyright of Google, Esri, or their imagery providers, and their terms of service govern what you may do with it. Google's terms restrict automated access to their services, and both providers restrict bulk extraction and redistribution of imagery. **Whether a particular use is permitted is your call to make, and your responsibility** — check the terms, and get permission where you need it. Research, archival and journalistic uses are not automatically exempt.
>
> The library ships conservative defaults (a `max_tiles` guard, on-disk caching to avoid refetching, 16 concurrent requests) but it cannot tell whether your use is allowed. Consider the licensed alternatives first: [Google Earth Engine](https://earthengine.google.com/) and Esri's [ArcGIS World Imagery Wayback](https://livingatlas.arcgis.com/wayback/) both offer sanctioned access to historical imagery.
>
> **Not affiliated with Google or Esri.** Google and Google Earth are trademarks of Google LLC; Esri, ArcGIS and World Imagery Wayback are trademarks of Environmental Systems Research Institute, Inc. They are named here only to identify the services this software talks to. This project is not affiliated with, endorsed by, or sponsored by either company.

```python
import old_imagery
from shapely.geometry import box

aoi = box(-122.4020, 37.7900, -122.3880, 37.7990)   # SF Embarcadero, EPSG:4326

# What imagery exists here, and when?
dates = old_imagery.availability(aoi, zoom=17)
print(dates[["date", "coverage", "complete", "providers"]].head())

# Fetch a mosaic for one of those dates.
with old_imagery.download(aoi, zoom=17, date="1938-08-01") as src:
    rgb = src.read()            # (3, H, W) uint8
    transform = src.transform   # georeferenced, EPSG:4326
```

## Install

```bash
pip install old-imagery
```

Requires Python 3.10+. Everything comes from wheels — `rasterio` bundles its own GDAL, so there is no system GDAL to install and no `protoc` build step.

## API

### `availability(aoi, zoom, *, min_date=None, max_date=None, provider="google", method="auto", ...)`

Returns a `GeoDataFrame` in EPSG:4326, one row per capture date, newest first:

| column | meaning |
| --- | --- |
| `date` | **image capture date** — see the note below |
| `n_tiles` | AOI tiles carrying imagery from that date |
| `coverage` | `n_tiles` as a fraction of the AOI's tiles |
| `complete` | date alone yields a gap-free mosaic |
| `providers` | imagery provider / release names, where known |
| `geometry` | covered area, clipped to the AOI |

`gdf.attrs` records `zoom`, `provider`, `n_aoi_tiles` and `method`.

### Dates always mean capture, never release

Every `date` in this library — in `availability`, in `download`, in the raster tags — is the date the imagery was **captured**.

This matters for Esri. A Wayback *release* (`"World Imagery (Wayback 2014-02-20)"`) is when Esri published a snapshot of the basemap; the imagery inside it may have been flown years earlier. In one San Francisco tile, imagery captured `2010-10-26` first appeared in the `2014-02-20` release. Upstream's CLI has a `--layer-date` flag that reinterprets `--date` as a release date; that flag is deliberately **not** ported, so there is no way to accidentally mix the two here. Release titles appear in the `providers` column as context only.

### How availability is resolved

`zoom` sets how finely date boundaries are traced. Zoom 15–17 is usually the right trade-off; the date *list* barely changes above that, only the precision of the polygons.

Esri has two ways to resolve availability, selected by `method`:

- `"per-tile"` — probe each tile against every Wayback release. Coverage is quantised to whole tiles.
- `"region"` — one query per release against the metadata feature service. Returns **true capture footprints**, so `geometry` is exact and zoom-independent.
- `"auto"` (default) — `"region"` at or above `ESRI_REGION_QUERY_MIN_TILES` (50) tiles, `"per-tile"` below.

`gdf.attrs["method"]` reports which ran. Google only has a per-tile path and ignores `method`.

Which is faster is **not** obvious from request counts — per-tile issues far more requests, but they are small and run 16-wide, while each region query hits a slow metadata service. Measured against the live service, cold cache, seconds:

| tiles | 4 | 12 | 30 | 72 |
| --- | --- | --- | --- | --- |
| `per-tile` | 30.9 | 32.8 | 62.0 | 67.9 |
| `region` | 63.3 | 96.7 | 143.4 | **40.7** |

Per-tile wins by 2–3× on small areas; region wins on large ones. Run-to-run variance on an identical AOI reached 2.3×, so the threshold is a rough midpoint, not a sharp optimum — set `method` explicitly if it matters. Reach for `method="region"` on small areas anyway when you want exact footprint geometry rather than tile-quantised coverage.

`n_tiles`, `coverage` and `complete` stay tile-based in both paths, so the numbers remain comparable across methods and providers.

### `download(aoi, zoom, date, *, date_match="closest", provider="google", ...)`

Returns an open, in-memory **3-band uint8 RGB** `rasterio.DatasetReader` covering the AOI's bounding box, snapped to tile pixels.

The CRS is each provider's **native tile grid** — `EPSG:4326` for Google, `EPSG:3857` for Esri — so no resampling happens on the way out. Reproject with `rasterio.warp` if you need something else.

`date_match` controls per-tile fallback when a tile has nothing on the target date: `"closest"` (default), `"exact"`, `"before"`, `"after"`. Because matching is per tile, a mosaic can mix dates; check the tags:

```python
ds.tags()
# {'dates': '1993-07-10', 'target_date': '1993-07-10', 'date_match': 'closest',
#  'zoom': '18', 'provider': 'google', 'tiles_total': '15', 'tiles_missing': '0'}
```

Tiles with no imagery are left black and excluded by the dataset mask, so use `ds.dataset_mask()` or `ds.read(masked=True)` to ignore gaps.

Both functions also take `max_workers` (default 16), `cache_dir`, and `max_tiles` (a guard against accidentally requesting a continent). Transient HTTP failures are retried; a request that still fails raises `old_imagery.RequestFailed`, and callers degrade one tile or one release rather than aborting the whole call.

## Caching

Responses are cached on disk under `~/.cache/old-imagery` (override with `$OLD_IMAGERY_CACHE_DIR` or the `cache_dir` argument; pass `cache_dir=None` to disable). Keyhole assets are addressed by epoch and therefore immutable, so cache entries never go stale; only dbRoot and the Esri capabilities document are re-fetched, daily and weekly respectively.

## Notes and limitations

- **Antimeridian.** AOIs must lie within longitude −180…180. Split geometries that cross it and query each half. Upstream handles the wrap; this port raises a clear error instead.
- **Esri is slow either way.** Wayback exposes no bulk per-tile date query, so the per-tile path probes ~195 releases per tile (~58 requests per tile) and the region path issues ~195 metadata queries plus one footprint fetch per capture date. Both take tens of seconds on a cold cache — see the table above. Google is far quicker than either.
- **Zoom limits.** `availability` and `download` reject zooms above **21 for Google** and **20 for Esri Wayback** — the deepest levels at which each service actually publishes imagery, per [upstream's docs](https://github.com/Mbucari/GEHistoricalImagery/blob/master/docs/availability.md). The tile schemes themselves address deeper (Keyhole to level 30, Web Mercator to 23, matching upstream's `KeyholeTile.MaxLevel` and `EsriTile.MaxLevel`), but those levels return well-formed tiles carrying no imagery while costing 4× the requests per level, so they raise rather than fail quietly. The caps are readable as `old_imagery.MAX_IMAGERY_ZOOM`. Upstream's CLI instead applies one `[1,23]` bound to both providers; that follows from its single shared `--zoom` flag rather than from either service's limits, so it is not what is enforced here.
- **Undated imagery.** Google tiles sometimes carry a provider's undated default imagery. It is excluded from `availability` (matching upstream) but is used by `download` as a last-resort fallback, in which case it contributes nothing to the `dates` tag.
- Only `availability` and `download` are ported. Upstream's `info`, `dump`, DXF output, terrain meshes, and the non-time-machine databases (Mars, Moon, Sky) are not.

## Development

```bash
pip install -e ".[dev]" && pre-commit install
```

```bash
pytest                  # 164 offline tests, no network
pytest -m network       # 9 live tests against Google and Esri
```

Run the network tests sparingly and against small AOIs — they hit the live services.

### Linting and types

`ruff` (lint + format) and `mypy` run on every commit via [pre-commit](.pre-commit-config.yaml), and again in CI:

```bash
pre-commit run --all-files
```

The mypy hook runs from your dev install rather than an isolated environment, so `pip install -e ".[dev]"` is a prerequisite — that way it resolves the same dependency tree CI does, instead of a second list that can drift.

mypy covers `src/` and `tools/`, not `tests/`: the suite deliberately passes wrong types to assert the errors they raise, and substitutes fake HTTP clients. Full `strict` is not on yet — it reports 61 findings, 38 of them missing annotations.

The offline suite includes upstream's own pre-computed quadtree subindex fixtures (`test/LibGoogleEarthTest/*IndexDictionary.json`), so a passing run means this port agrees with the C# implementation node-for-node.

### How the protobuf schemas got here

The package loads serialized `FileDescriptorProto` blobs from `src/old_imagery/_descriptors/*.desc` into a private descriptor pool at import, so installing needs no `protoc`. Those blobs are build outputs; the sources are in [`proto/`](proto/) and are regenerated with:

```bash
python tools/regen_descriptors.py
```

The two schemas have different provenance:

- **`dbroot_v2.proto`** is Google's own, copied unmodified from [google/earthenterprise](https://github.com/google/earthenterprise) (`earth_enterprise/src/keyhole/proto/dbroot/dbroot_v2.proto`) with its Apache-2.0 header intact.
- **`quadtreeset.proto`** has no published upstream. Earth Enterprise ships `dbroot_v2.proto` but not quadtreeset, whose internal name — `quadtreeset.protodevel` — survives in descriptors embedded in Google Earth clients. This file is a transcription of that schema into readable source: the field numbers, types and enum values were read out of the `FileDescriptorProto` embedded in upstream's `protoc`-generated C#.

`regen_descriptors.py` diffs every regenerated descriptor against the committed one and refuses to write if any message, field number, type, label or enum value would be lost; purely additive upstream changes are reported and allowed. `--check` verifies the committed blobs match `proto/` without writing, which is worth running in CI.

## Licence

**GPL-3.0-only.** See [LICENSE](LICENSE) and [NOTICE](NOTICE).

This is a port of [GEHistoricalImagery](https://github.com/Mbucari/GEHistoricalImagery) by Mbucari, which is GPL-3.0. A port is a derivative work, so old-imagery is distributed under the same licence — it cannot be MIT or BSD. **If you link `old-imagery` into your own program, the GPL's terms apply to that program too.** If that is a problem for you, the imagery is also reachable through [Google Earth Engine](https://earthengine.google.com/) and [Esri's Wayback](https://livingatlas.arcgis.com/wayback/) under their own terms.

`proto/dbroot_v2.proto` is Apache-2.0, copyright 2017 Google Inc. The test fixtures in `tests/data/` are copied from upstream — see [tests/data/README.md](tests/data/README.md). Full attribution and a list of changes made in this port are in [NOTICE](NOTICE).

Imagery retrieved with this library is not covered by this licence; see the notice at the top of this file.
