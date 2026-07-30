# old-imagery

Historical aerial imagery from **Google Earth** and the **Esri World Imagery Wayback** archive, as a handful of Python functions.

This is a Python port of the protocol layer of [Mbucari/GEHistoricalImagery](https://github.com/Mbucari/GEHistoricalImagery) (a .NET CLI). It is a library, not a command line tool: areas of interest go in as shapely geometries, availability comes back as a GeoDataFrame, and imagery comes back as an open rasterio dataset.

> ### Before you use this
>
> **This library retrieves imagery. It does not license it.**
>
> Imagery you fetch with `old-imagery` stays the copyright of Google, Esri, or their imagery providers, and their terms of service govern what you may do with it. [Google Earth's terms](https://maps.google.com/intl/en_all/help/terms_maps-earth/) prohibit mass downloads and bulk feeds; Esri content may carry provider-specific rights and restrictions. **Whether a particular use is permitted is your call to make, and your responsibility** — check the current terms and item details, and get permission where you need it. Research, archival and journalistic uses are not automatically exempt.
>
> The library defaults to at most 1,000 tiles, caches responses to avoid refetching, and caps concurrency at what each service has been measured to tolerate (16 for Google, 10 for Esri) rather than letting callers raise it, but it cannot tell whether your use is allowed. Consider official alternatives first: the [Google Earth Engine data catalogue](https://developers.google.com/earth-engine/datasets/) provides other historical Earth-observation datasets under their listed terms (not the Google Earth basemap archive), and Esri's [ArcGIS World Imagery Wayback](https://livingatlas.arcgis.com/wayback/) is the official interface to the Wayback archive.
>
> **Not affiliated with Google or Esri.** Google and Google Earth are trademarks of Google LLC; Esri, ArcGIS and World Imagery Wayback are trademarks of Environmental Systems Research Institute, Inc. They are named here only to identify the services this software talks to. This project is not affiliated with, endorsed by, or sponsored by either company.

```python
import old_imagery
from shapely.geometry import box

aoi = box(-122.4020, 37.7900, -122.3880, 37.7990)   # SF Embarcadero, EPSG:4326

# What imagery exists here, and when?
dates = old_imagery.availability(aoi, zoom=17)
print(dates[["date", "coverage", "complete", "providers"]].head())

# Fetch exactly one of the dates availability returned.
target = dates.iloc[-1]["date"]  # oldest capture in this example
with old_imagery.download(aoi, zoom=17, date=target, date_match="exact") as src:
    rgb = src.read()            # (3, H, W) uint8
    transform = src.transform   # georeferenced, EPSG:4326
```

Pass `provider="esri"` to both calls to use Esri Wayback instead. Every AOI is
interpreted as longitude/latitude in **EPSG:4326**; shapely geometries do not
carry a CRS, so reproject before calling the library.

## Install

```bash
pip install old-imagery
```

Requires Python 3.10+. On supported Linux, macOS and Windows platforms, pip
normally installs a `rasterio` wheel containing GDAL, so a separate system GDAL
is not needed. Source installs of rasterio may require GDAL development files.
The `old-imagery` package itself needs no `protoc` build step.

## API

The public surface is deliberately small: the three functions below, the option
types their signatures name, the constants those defaults refer to, and the two
exceptions. `old_imagery.__all__` is the complete list.

`availability` and `download` are the common pair — which capture dates exist,
then give me the pixels for one. `esri_mosaic_as_of` answers a separate
question, about what a published Esri snapshot displays rather than what the
archive holds.

The three string-valued options are closed sets, exported as type aliases so a
type checker rejects a typo before it reaches the network:

```python
Provider  = Literal["google", "esri"]
Method    = Literal["auto", "per-tile", "region"]
DateMatch = Literal["closest", "exact", "before", "after"]
```

Each is still validated at runtime, so callers without a type checker get a
`ValueError` naming the accepted values rather than undefined behaviour.

The protocol layer beneath this — the Keyhole quadtree walker, dbRoot client
and cached HTTP client — lives in underscore modules. It is reachable, but it
is not a supported interface: it tracks Google's and Esri's wire formats and
changes when they do.

### `availability`

```python
availability(
    aoi: BaseGeometry,
    zoom: int,
    *,
    min_date: date | str | None = None,
    max_date: date | str | None = None,
    provider: Provider = "google",
    method: Method = "auto",
    cache_dir: str | os.PathLike[str] | None = DEFAULT_CACHE_DIR,
    max_tiles: int = 1_000,
) -> geopandas.GeoDataFrame
```

`aoi` is a shapely `BaseGeometry` interpreted in EPSG:4326. Date bounds are
inclusive. The function returns a `GeoDataFrame` in EPSG:4326, one row per
capture date, newest first:

| column | meaning |
| --- | --- |
| `date` | **image capture date** — see the note below |
| `n_tiles` | AOI tiles carrying imagery from that date |
| `coverage` | `n_tiles` as a fraction of the AOI's tiles |
| `complete` | date covers every AOI tile (`coverage == 1.0`); measured at tile resolution, not a guarantee that every point is covered |
| `providers` | imagery provider / release names, where known |
| `geometry` | covered area, clipped to the AOI |

`gdf.attrs` records `zoom`, `provider`, `n_aoi_tiles` and `method` —
`"per-tile"`, `"region-query"`, or `"none"` when the AOI selects no tiles.

`availability` reports **capture dates only**. To ask what one published Esri
snapshot displays, use [`esri_mosaic_as_of`](#esri_mosaic_as_of).

### Dates always mean capture, never release

The `date` argument, every `date` returned by `availability` and
`esri_mosaic_as_of`, and every value in the raster `dates` tag is the date the
imagery was **captured**. Esri imagery versions whose capture metadata is
unavailable are omitted; a Wayback release date is never substituted for a
missing capture date.

Exactly one argument in this package means publication instead:
`esri_mosaic_as_of`'s `as_of_date`, together with `download`'s
`esri_wayback_*` release selectors. They are named so you can see it at the
call site.

This matters for Esri. A Wayback *release* (`"World Imagery (Wayback 2014-02-20)"`) is when Esri published a snapshot of the basemap; the imagery inside it may have been flown years earlier. In one San Francisco tile, imagery captured `2010-10-26` first appeared in the `2014-02-20` release. The `date` argument is never reinterpreted as a release date.

### `esri_mosaic_as_of`

```python
esri_mosaic_as_of(
    aoi: BaseGeometry,
    zoom: int | Sequence[int],
    as_of_date: date | str,
    *,
    cache_dir: str | os.PathLike[str] | None = DEFAULT_CACHE_DIR,
    max_footprints: int = 500,
) -> geopandas.GeoDataFrame
```

`availability` asks *which capture dates exist here*. This asks a different
question: **if I had looked at the Esri basemap on this date, what would I have
been seeing, and how old is each part of it?**

A Wayback release is a mosaic stitched from imagery flown across many years, so
"what it displays" is its internal seam map. That map comes from Esri's own
capture footprints, not from probing tiles:

```python
seams = old_imagery.esri_mosaic_as_of(aoi, zoom=18, as_of_date="2020-06-01")
print(seams[["zoom", "date", "area_fraction", "release_id"]])
#    zoom        date  area_fraction   release_id
# 0    18  2016-04-02          0.617  WB_2020_R05
# 1    18  2010-10-26          0.383  WB_2020_R05
```

| column | meaning |
| --- | --- |
| `zoom` | the zoom this row was resolved at |
| `date` | **capture date** of the imagery displayed here |
| `area_fraction` | this row's share of the AOI, as a planar area ratio in EPSG:3857 |
| `release_id` | stable identifier of the resolved release, e.g. `WB_2026_R03` |
| `geometry` | the area showing that date, clipped to the AOI — real footprint boundaries, not tile edges |

`gdf.attrs` records `release_id`, `release_date` (the **publication** date),
`release_title`, `as_of_date` and `zooms`.

`as_of_date` is the one date in this package that means publication rather than
capture, and it is named to say so. The rule is deterministic: take the
catalogue release with the greatest date less than or equal to the one asked
for, never a future release. That handles Esri's known one-day inconsistency for
`WB_2026_R03`, which other Esri configuration calls `2026-03-26` while the WMTS
title calls `2026-03-25`.

#### Zoom is an axis here, not just a resolution knob

Esri composes the mosaic per scale and publishes metadata per scale, so the same
ground in the same release can show a *different capture date* at different
zooms. Pass several zooms to see that:

```python
seams = old_imagery.esri_mosaic_as_of(aoi, [13, 16, 19], "2020-06-01")
```

Zooms of 10 and below all resolve to the same metadata layer and so return
identical geometry. Cost scales with the number of capture footprints, not the
number of tiles — which is why the guard is `max_footprints` and there is no
`max_tiles`.

Unlike `availability`, this refuses partial answers: if Esri's metadata service
returns an incomplete feature list, it raises `RequestFailed` rather than
returning a map whose holes would be indistinguishable from ground the release
genuinely does not cover.

#### Getting the matching pixels

`download` takes the `release_id` this function reports, so a seam map
round-trips to imagery:

```python
with old_imagery.download(
    aoi, zoom=18, provider="esri", esri_wayback_release_id=seams.attrs["release_id"]
) as src:
    print(src.tags()["selection_mode"])  # "esri-wayback-release"
```

This is the only release selector on `download`, and the only way to resolve a
*service* date to a release is through `esri_mosaic_as_of` — so Esri's
catalogue-date inconsistencies get handled in exactly one place.

`esri_wayback_release_id` requires `provider="esri"` and `date=None` (the
default), and rejects a non-default `date_match`; those combinations are errors
rather than ambiguous requests. Release downloads record
`esri_wayback_release_id`, `esri_wayback_catalogue_date` and
`esri_wayback_release_title` in the raster tags instead of `target_date` and
`date_match`.

#### Why `date=` cannot replace it

A release is a mosaic of imagery captured on many dates, so knowing the capture
date is not enough to reproduce one. On a 9-tile AOI straddling a real seam
between `2017-08-31` and `2017-08-20` in release `WB_2020_R07`:

| | `esri_wayback_release_id=…` | `date="2017-08-31", date_match="exact"` |
| --- | --- | --- |
| tiles missing | **0 / 9** | **5 / 9** |
| `dates` tag | `2017-08-20, 2017-08-31` | `2017-08-31` |
| elapsed | 2.5 s | 19.1 s |

Asking for one capture date masks out everything captured on any other, and
re-derives each tile's whole release history to do it (~1 request per tile
against ~58). An exact release download also retains a tile whose capture
metadata is missing — counted in `tiles_capture_date_unknown` — which `date=`
cannot reach at all, because Esri's per-tile path drops undated versions.

### How availability is resolved

`zoom` sets how finely date boundaries are traced. Zoom 15–17 is usually the right trade-off; the date *list* barely changes above that, only the precision of the polygons.

Esri has two ways to resolve availability, selected by `method`:

- `"per-tile"` — probe each tile against every Wayback release. Coverage is quantised to whole tiles.
- `"region"` — one query per release against the metadata feature service. Returns provider-reported capture footprints, so `geometry` is not quantised to tiles. It is *not* zoom-independent, though: Esri publishes metadata per scale (`min(13, 23 - zoom)`), so a different zoom queries a different metadata layer and can return different footprints.
- `"auto"` (default) — `"region"` at or above `ESRI_REGION_QUERY_MIN_TILES` (50) tiles, `"per-tile"` below.

`gdf.attrs["method"]` reports `"per-tile"` or `"region-query"` according to
which path ran.

Google has only a per-tile path. It accepts `method="auto"` and
`method="per-tile"` — both resolve per-tile — but `method="region"` raises
`ValueError` rather than silently falling back, since a caller asking for
footprint geometry would otherwise receive tile-quantised geometry without
being told.

Which is faster is **not** obvious from request counts — per-tile issues far more requests, but they are small and run concurrently, while each region query hits a slow metadata service. Measured against the live service, cold cache, seconds:

| tiles | 4 | 12 | 30 | 72 |
| --- | --- | --- | --- | --- |
| `per-tile` | 30.9 | 32.8 | 62.0 | 67.9 |
| `region` | 63.3 | 96.7 | 143.4 | **40.7** |

Per-tile wins by 2–3× on small areas; region wins on large ones. Run-to-run variance on an identical AOI reached 2.3×, so the threshold is a rough midpoint, not a sharp optimum — set `method` explicitly if it matters. Reach for `method="region"` on small areas anyway when you want exact footprint geometry rather than tile-quantised coverage.

`n_tiles`, `coverage` and `complete` are tile-based in both paths, but "tile-based" does not mean quite the same thing in each. The per-tile path counts a tile when the provider reports imagery for the **whole tile**; the region path counts it when a footprint merely **intersects** its extent. A footprint clipping a sliver off every AOI tile therefore reports `coverage == 1.0` and `complete == True` over almost no area. Compare `geometry` areas rather than `coverage` when comparing the two paths, and note that crossing the `method="auto"` threshold changes which of the two meanings you get.

### `download`

```python
download(
    aoi: BaseGeometry,
    zoom: int,
    date: date | str | None = None,
    *,
    date_match: DateMatch = "closest",
    provider: Provider = "google",
    esri_wayback_release_id: str | None = None,
    cache_dir: str | os.PathLike[str] | None = DEFAULT_CACHE_DIR,
    max_tiles: int = 1_000,
) -> rasterio.DatasetReader
```

`aoi` is a shapely `BaseGeometry` interpreted in EPSG:4326. The function returns
an open, in-memory **3-band uint8 RGB** `rasterio.DatasetReader` covering the
AOI's bounding box, snapped to tile pixels.

The CRS is each provider's **native tile grid** — `EPSG:4326` for Google, `EPSG:3857` for Esri — so no resampling happens on the way out. Reproject with `rasterio.warp` if you need something else.

`date_match` controls per-tile fallback when a tile has nothing on the target date: `"closest"` (default), `"exact"`, `"before"`, `"after"`. Because matching is per tile, a mosaic can mix dates; check the tags:

```python
ds.tags()
# {'selection_mode': 'capture-date', 'dates': '1993-07-10',
#  'target_date': '1993-07-10', 'date_match': 'closest', 'zoom': '18',
#  'provider': 'google', 'tiles_total': '15', 'tiles_missing': '0',
#  'tiles_capture_date_unknown': '0'}
```

Tiles with no imagery are left black and excluded by the dataset mask, so use `ds.dataset_mask()` or `ds.read(masked=True)` to ignore gaps.

To save the in-memory result without changing its pixels or georeferencing:

```python
from rasterio.shutil import copy as rio_copy

with old_imagery.download(aoi, 17, target, date_match="exact") as src:
    rio_copy(src, "mosaic.tif", driver="GTiff")
```

Both functions also take `cache_dir` and `max_tiles` (default 1,000), which
limits the tile grid spanned by the AOI's bounding box. `download` holds the RGB
mosaic and its mask in memory: 1,000 full tiles require about 250 MiB for those
raw buffers, plus the in-memory GeoTIFF and decoding overhead. Raise the guard
only after estimating the resulting memory and request cost.

### Concurrency is not a knob

There is no `max_workers`. Every parallel section here is network-bound, so the
pool is sized by what the *service* tolerates and by how much work there is —
never more threads than tasks, and never more than the per-provider cap (16 for
Google, 10 for Esri).

Upstream's `--concurrency` defaults to ALL_CPUS. CPU count is the wrong basis by
about two orders of magnitude. The only real CPU work is JPEG decode, and
decoding 256×256 RGB tiles on a 10-CPU machine measures:

| threads | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | --- | --- | --- | --- | --- | --- |
| tiles/s | 2347 | 4291 | 7373 | 7054 | 7052 | 6800 |

Decode does release the GIL, but it saturates at four threads and costs ~0.4 ms
per tile against tens of milliseconds to fetch one — roughly 1% of the per-tile
cost. Any width chosen for the network already exceeds what decode can use, so a
single network-sized pool is right and a separate CPU-sized pool would be
machinery for nothing.

The caps are deliberately fixed rather than adaptive. Finding a service's limit
means exceeding it, and Google's terms prohibit bulk feeds — the polite client
never goes looking for the ceiling. See `src/old_imagery/_concurrency.py` for
the measurements and their (stated) uncertainty.

Transient HTTP failures are retried. Failures for an individual tile, packet,
or Wayback release are treated as missing data so one bad response does not
usually abort the whole call. A failure while loading the initial Google
dbRoot or Esri capabilities document raises `old_imagery.RequestFailed`.

## Caching

Responses are cached on disk under `~/.cache/old-imagery` (override with
`$OLD_IMAGERY_CACHE_DIR` before importing the package, or with the `cache_dir`
argument; pass `cache_dir=None` to disable). Keyhole assets are addressed by
epoch and therefore immutable, so cache entries never go stale; only dbRoot and
the Esri capabilities document are re-fetched, daily and weekly respectively.
There is no automatic size limit or eviction policy, so long-running workflows
should monitor or periodically remove this directory.

## Notes and limitations

- **Antimeridian.** AOIs must lie within longitude −180…180. Split geometries that cross it and query each half. Upstream handles the wrap; this port raises a clear error instead.
- **Esri is slow either way.** Wayback exposes no bulk per-tile date query, so the per-tile path probes ~195 releases per tile (~58 requests per tile) and the region path issues ~195 metadata queries plus one footprint fetch per capture date. Both take tens of seconds on a cold cache — see the table above. Google is far quicker than either.
- **Zoom limits.** `availability`, `download` and `esri_mosaic_as_of` reject zooms above **21 for Google** and **20 for Esri Wayback** — the deepest levels at which each service actually publishes imagery, per [upstream's docs](https://github.com/Mbucari/GEHistoricalImagery/blob/master/docs/availability.md). The tile schemes themselves address deeper (Keyhole to level 30, Web Mercator to 23, matching upstream's `KeyholeTile.MaxLevel` and `EsriTile.MaxLevel`), but those levels return well-formed tiles carrying no imagery while costing 4× the requests per level, so they raise rather than fail quietly. The caps are readable as `old_imagery.MAX_IMAGERY_ZOOM`. Upstream's CLI instead applies one `[1,23]` bound to both providers; that follows from its single shared `--zoom` flag rather than from either service's limits, so it is not what is enforced here.
- **Undated imagery.** Google tiles sometimes carry a provider's undated default imagery. It is excluded from `availability` (matching upstream) but is used by `download` as a last-resort fallback, in which case it contributes nothing to the `dates` tag.
- **Missing Esri capture metadata.** Normal capture-date searches omit an Esri
  imagery version when its metadata service does not provide a usable capture
  date. Exact release downloads retain its pixels and count the tile in
  `tiles_capture_date_unknown`. Its Wayback release date is never used as a
  substitute capture date.
- Only `availability` and `download` are ported. Upstream's `info`, `dump`, DXF output, terrain meshes, and the non-time-machine databases (Mars, Moon, Sky) are not.

## Development

```bash
pip install -e ".[dev]" && pre-commit install
```

```bash
pytest                  # 184 offline tests, no network
pytest -m network       # 10 live tests against Google and Esri
```

Run the network tests sparingly and against small AOIs — they hit the live services.

### Linting and types

`ruff` (lint + format) and `mypy` run on every commit via [pre-commit](https://github.com/angusmcb/old-imagery/blob/main/.pre-commit-config.yaml), and again in CI:

```bash
pre-commit run --all-files
```

The mypy hook runs from your dev install rather than an isolated environment, so `pip install -e ".[dev]"` is a prerequisite — that way it resolves the same dependency tree CI does, instead of a second list that can drift.

mypy covers `src/` and `tools/`, not `tests/`: the suite deliberately passes
wrong types to assert the errors they raise, and substitutes fake HTTP clients.
Full `strict` is not enabled yet; the regular check still validates annotated
code and checks function bodies.

The offline suite includes upstream's own pre-computed quadtree subindex fixtures (`test/LibGoogleEarthTest/*IndexDictionary.json`), so a passing run means this port agrees with the C# implementation node-for-node.

### How the protobuf schemas got here

The package loads serialized `FileDescriptorProto` blobs from `src/old_imagery/_descriptors/*.desc` into a private descriptor pool at import, so installing needs no `protoc`. Those blobs are build outputs; the sources are in [`proto/`](https://github.com/angusmcb/old-imagery/tree/main/proto) and are regenerated with:

```bash
python tools/regen_descriptors.py
```

The two schemas have different provenance:

- **`dbroot_v2.proto`** is Google's own, copied unmodified from [google/earthenterprise](https://github.com/google/earthenterprise) (`earth_enterprise/src/keyhole/proto/dbroot/dbroot_v2.proto`) with its Apache-2.0 header intact.
- **`quadtreeset.proto`** has no published upstream. Earth Enterprise ships `dbroot_v2.proto` but not quadtreeset, whose internal name — `quadtreeset.protodevel` — survives in descriptors embedded in Google Earth clients. This file is a transcription of that schema into readable source: the field numbers, types and enum values were read out of the `FileDescriptorProto` embedded in upstream's `protoc`-generated C#.

`regen_descriptors.py` diffs every regenerated descriptor against the committed
one and refuses to write if any message, field number, type, label or enum value
would be lost; purely additive upstream changes are reported and allowed.
`--check` verifies the committed blobs match `proto/` without writing, and CI
runs that check.

## Licence

**GPL-3.0-only.** See [LICENSE](https://github.com/angusmcb/old-imagery/blob/main/LICENSE) and [NOTICE](https://github.com/angusmcb/old-imagery/blob/main/NOTICE).

This is a port of [GEHistoricalImagery](https://github.com/Mbucari/GEHistoricalImagery)
by Mbucari, which is GPL-3.0-or-later. This project exercises the GPLv3 option
and distributes the combined work as GPL-3.0-only. If you distribute a program
combined with `old-imagery`, the GPL's terms apply to the combined work; private
use does not by itself require publication. If that does not fit your intended
distribution, use independently licensed imagery from an official data
provider instead.

`proto/dbroot_v2.proto` is Apache-2.0, copyright 2017 Google Inc. The test fixtures in `tests/data/` are copied from upstream — see [tests/data/README.md](https://github.com/angusmcb/old-imagery/blob/main/tests/data/README.md). Full attribution and a list of changes made in this port are in [NOTICE](https://github.com/angusmcb/old-imagery/blob/main/NOTICE).

Imagery retrieved with this library is not covered by this licence; see the notice at the top of this file.
