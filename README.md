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

Pass `provider="esri"` to both calls to use Esri Wayback instead. Every `aoi`
argument is a shapely `Polygon`, `MultiPolygon` or `GeometryCollection` (`box()`
returns a `Polygon`) enclosing some area, interpreted as longitude/latitude in
**EPSG:4326** — shapely geometries do not carry a CRS, so reproject before
calling the library.

[`examples/getting-started.ipynb`](https://github.com/angusmcb/old-imagery/blob/main/examples/getting-started.ipynb)
tours the whole API with plots: availability, downloads and their tags, both
providers, seam maps, saving, and the guardrails. Run it with
`pip install "old-imagery[examples]"`.

## Install

```bash
pip install old-imagery
```

Requires Python 3.10+. On supported Linux, macOS and Windows platforms, pip
normally installs a `rasterio` wheel containing GDAL, so a separate system GDAL
is not needed. Source installs of rasterio may require GDAL development files.
The `old-imagery` package itself needs no `protoc` build step.

## API

The public surface is the four functions below, the constants their defaults
refer to, and the two exceptions; `old_imagery.__all__` is the complete list.

`availability` and `download` are the common pair — which capture dates exist,
then give me the pixels for one. `esri_mosaic_as_of` answers a separate
question, about what a published Esri snapshot displays rather than what the
archive holds.

The protocol layer beneath this — the Keyhole quadtree walker, dbRoot client
and cached HTTP client — lives in underscore modules. It is reachable, but it
is not a supported interface: it tracks Google's and Esri's wire formats and
changes when they do.

### `availability`

```python
availability(
    aoi: shapely.Polygon | shapely.MultiPolygon | shapely.GeometryCollection,
    zoom: int,
    *,
    min_date: date | str | None = None,
    max_date: date | str | None = None,
    provider: Literal["google", "esri"] = "google",
    cache_dir: str | os.PathLike[str] | None = DEFAULT_CACHE_DIR,
    max_tiles: int = 1_000,
) -> geopandas.GeoDataFrame
```

`coverage` is a fraction of the AOI's area, so a `Point` or `LineString` is
rejected with a message telling you to buffer it. Date bounds are inclusive.
The function returns a `GeoDataFrame` in EPSG:4326, one row per capture date,
newest first:

| column | meaning |
| --- | --- |
| `date` | **image capture date** — see the note below |
| `coverage` | fraction of the AOI's **area** covered by imagery from that date |
| `complete` | that date covers the whole AOI (`coverage == 1.0`) |
| `providers` | imagery provider / release names, where known |
| `source_providers` | distinct Esri imagery source names; an empty tuple for Google or missing metadata |
| `source_descriptions` | distinct Esri source descriptions |
| `source_resolutions_m` | distinct Esri native source resolutions in metres |
| `source_accuracies_m` | distinct Esri positional accuracies in metres |
| `min_map_levels`, `max_map_levels` | Esri source scale ranges, reported as provenance; never affect the requested zoom |
| `geometry` | covered area, clipped to the AOI |

`coverage` is an area fraction, computed as a planar ratio in EPSG:3857. That
makes it mean the same thing for both providers and independent of `zoom` — a
date covering half your AOI reports `0.5` whether you asked at zoom 13 or 19.

`geometry` is resolved as finely as each provider allows, with nothing to
choose: Esri returns true capture footprints, Google a union of tile extents.
See [How availability is resolved](#how-availability-is-resolved).

`gdf.attrs` records `zoom`, `provider`, `n_aoi_tiles` and `method` —
`"region-query"` for Esri, `"per-tile"` for Google, or `"none"` when the AOI
selects no tiles.

`availability` reports **capture dates only**. To ask what one published Esri
snapshot displays, use [`esri_mosaic_as_of`](#esri_mosaic_as_of).

### `esri_wayback_releases`

```python
esri_wayback_releases(
    *,
    cache_dir: str | os.PathLike | None = DEFAULT_CACHE_DIR,
) -> pandas.DataFrame
```

Returns Esri's published Wayback release catalogue, newest first. Use it to
choose an exact stable release ID for `esri_mosaic_as_of` or `download`:

```python
releases = old_imagery.esri_wayback_releases()
print(releases.head())
```

| column | type | meaning |
| --- | --- | --- |
| `release_id` | string | stable ID such as `WB_2026_R07` |
| `release_date` | `datetime64[ns]` | publication date |
| `release_title` | string | Esri catalogue title |

### Dates always mean capture, never release

The `date` argument, every `date` returned by `availability` and
`esri_mosaic_as_of`, and every value in the raster `dates` tag is the date the
imagery was **captured**. Esri imagery versions whose capture metadata is
unavailable are omitted; a Wayback release date is never substituted for a
missing capture date.

Publication is represented by `esri_wayback_releases`'s `release_date`,
`esri_mosaic_as_of`'s `as_of`, and `download`'s `esri_wayback_*` release
selectors rather than by capture-date fields.

This matters for Esri. A Wayback *release* (`"World Imagery (Wayback 2014-02-20)"`) is when Esri published a snapshot of the basemap; the imagery inside it may have been flown years earlier. In one San Francisco tile, imagery captured `2010-10-26` first appeared in the `2014-02-20` release.

### `esri_mosaic_as_of`

```python
esri_mosaic_as_of(
    aoi: shapely.Polygon | shapely.MultiPolygon | shapely.GeometryCollection,
    zoom: int | Sequence[int],
    as_of: date | str,
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
seams = old_imagery.esri_mosaic_as_of(aoi, zoom=18, as_of="2020-06-01")
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
| `source_provider`, `source_description` | Esri's imagery source name and description |
| `source_resolution_m`, `source_accuracy_m` | source resolution and positional accuracy in metres |
| `min_map_level`, `max_map_level` | Esri's source scale range, reported as provenance; never affects the requested zoom |
| `geometry` | the area showing that date, clipped to the AOI — real footprint boundaries, not tile edges |

`gdf.attrs` records `release_id`, `release_date` (the **publication** date),
`release_title`, `as_of` and `zooms`.

Source metadata remains attached when two footprints have the same capture
date: only footprints with identical source metadata are dissolved together.

`as_of` accepts either a date, selecting the catalogue release with the greatest
date less than or equal to it, or an exact `release_id` from
`esri_wayback_releases()`.

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

This is the only release selector on `download`. `date=` cannot substitute for
it: a release is a mosaic of many capture dates, so asking for one date masks
out everything captured on any other, costs roughly one request per tile to
re-derive each tile's release history, and cannot reach a tile whose capture
metadata is missing at all.

`esri_wayback_release_id` requires `provider="esri"` and `date=None` (the
default), and rejects a non-default `date_match`; those combinations are errors
rather than ambiguous requests. Release downloads record
`esri_wayback_release_id`, `esri_wayback_catalogue_date` and
`esri_wayback_release_title` in the raster tags instead of `target_date` and
`date_match`, and count any tile with missing capture metadata in
`tiles_capture_date_unknown`.

### How availability is resolved

`zoom` sets how finely date boundaries are traced. Zoom 15–17 is usually the right trade-off; the date *list* barely changes above that, only the precision of the polygons.

**There is no `method` option.** Each provider is resolved as finely as it allows:

- **Esri** returns real capture footprints, so date boundaries follow actual imagery seams. Note these are *not* zoom-independent: Esri publishes metadata per scale (`min(13, 23 - zoom)`), so a different zoom queries a different metadata layer and can return different footprints.
- **Google** has nothing finer to offer — dbRoot reports dates per tile — so its `geometry` is a union of tile extents. Your AOI's outline survives the clip; internal date boundaries are tile-shaped.

`gdf.attrs["method"]` reports which ran (`"region-query"` or `"per-tile"`), so a
result can say how it was obtained. It is not selectable.

`coverage` is measured as area on both paths, so neither depends on the tile
grid. That grid is a transport detail: it bounds the request cost and narrows
the release list, but it never shapes the answer.

### `download`

```python
download(
    aoi: shapely.Polygon | shapely.MultiPolygon | shapely.GeometryCollection,
    zoom: int,
    date: date | str | None = None,
    *,
    date_match: Literal["closest", "exact", "before", "after"] = "closest",
    provider: Literal["google", "esri"] = "google",
    esri_wayback_release_id: str | None = None,
    cache_dir: str | os.PathLike[str] | None = DEFAULT_CACHE_DIR,
    max_tiles: int = 1_000,
) -> rasterio.DatasetReader
```

Returns an open, in-memory **3-band uint8 RGB** `rasterio.DatasetReader`
covering the AOI's bounding box, snapped to tile pixels.

The CRS is each provider's **native tile grid** — `EPSG:4326` for Google, `EPSG:3857` for Esri — so no resampling happens on the way out. Reproject with `rasterio.warp` if you need something else.

`date_match` controls per-tile fallback when a tile has nothing on the target date: `"closest"` (default), `"exact"`, `"before"`, `"after"`. Because matching is per tile, a mosaic can mix dates; check the tags:

```python
ds.tags()
# {'selection_mode': 'capture-date', 'dates': '1993-07-10',
#  'target_date': '1993-07-10', 'date_match': 'closest', 'zoom': '18',
#  'provider': 'google', 'tiles_total': '15', 'tiles_missing': '0',
#  'tiles_capture_date_unknown': '0'}
```

Esri downloads additionally include `esri_source_metadata` when source
metadata is available. It is compact JSON containing the distinct provider,
description, native resolution, positional accuracy and map-level values used
by the mosaic.

Tiles with no imagery are left black and excluded by the dataset mask, so use `ds.dataset_mask()` or `ds.read(masked=True)` to ignore gaps.

To save the in-memory result without changing its pixels or georeferencing:

```python
from rasterio.shutil import copy as rio_copy

with old_imagery.download(aoi, 17, target, date_match="exact") as src:
    rio_copy(src, "mosaic.tif", driver="GTiff")
```

`max_tiles` limits the tile grid spanned by the AOI's bounding box. `download`
holds the RGB mosaic and its mask in memory: 1,000 full tiles require about
250 MiB for those raw buffers, plus the in-memory GeoTIFF and decoding
overhead. Raise the guard only after estimating the resulting memory and
request cost.

### Concurrency and failures

There is no `max_workers`. Every parallel section here is network-bound, so the
pool is sized by what the *service* tolerates and by how much work there is —
never more threads than tasks, and never more than the per-provider cap (16 for
Google, 10 for Esri). The caps are deliberately fixed rather than adaptive:
finding a service's limit means exceeding it, and Google's terms prohibit bulk
feeds. See `src/old_imagery/_concurrency.py` for the measurements behind them
and their (stated) uncertainty.

Transient HTTP failures are retried. Failures for an individual tile, packet,
or Wayback release are treated as missing data so one bad response does not
usually abort the whole call. A failure while loading the initial Google
dbRoot or Esri capabilities document raises `old_imagery.RequestFailed`.

## Caching

Responses are cached on disk, in the location each platform expects:

| platform | default | |
| --- | --- | --- |
| Linux/BSD | `$XDG_CACHE_HOME/old-imagery` | falls back to `~/.cache/old-imagery` |
| macOS | `~/Library/Caches/old-imagery` | |
| Windows | `%LOCALAPPDATA%\old-imagery\Cache` | `LOCALAPPDATA`, not `APPDATA` — a cache must not roam between machines |

Read `old_imagery.DEFAULT_CACHE_DIR` to see the resolved path. Override it with
`$OLD_IMAGERY_CACHE_DIR` before importing the package, or per call with the
`cache_dir` argument; pass `cache_dir=None` to disable. Keyhole assets are addressed by
epoch and therefore immutable, so cache entries never go stale; only dbRoot and
the Esri capabilities document are re-fetched, daily and weekly respectively.
There is no automatic size limit or eviction policy, so long-running workflows
should monitor or periodically remove this directory.

## Notes and limitations

- **Antimeridian.** AOIs must lie within longitude −180…180. Split geometries that cross it and query each half.
- **Esri is slow.** Wayback exposes no bulk per-tile date query, so an availability call issues ~195 metadata queries plus one footprint fetch per capture date, and takes tens of seconds on a cold cache. Footprint payloads grow with AOI area — one sampled footprint had 3,520 vertices — so large AOIs are slower still. Google is far quicker.
- **Zoom limits.** `availability`, `download` and `esri_mosaic_as_of` reject zooms above **21 for Google** and **20 for Esri Wayback** — the deepest levels at which each service actually publishes imagery, per [upstream's docs](https://github.com/Mbucari/GEHistoricalImagery/blob/master/docs/availability.md). Deeper levels return well-formed tiles carrying no imagery while costing 4× the requests per level, so they raise rather than fail quietly. The caps are readable as `old_imagery.MAX_IMAGERY_ZOOM`.
- **Undated imagery.** Google tiles sometimes carry a provider's undated default imagery. It is excluded from `availability` but is used by `download` as a last-resort fallback, in which case it contributes nothing to the `dates` tag.
- **Missing Esri capture metadata.** Capture-date searches omit an Esri imagery version when its metadata service does not provide a usable capture date. Exact release downloads retain its pixels and count the tile in `tiles_capture_date_unknown`.
- Only `availability` and `download` are ported. Upstream's `info`, `dump`, DXF output, terrain meshes, and the non-time-machine databases (Mars, Moon, Sky) are not.

## Development

```bash
uv sync && uv run pre-commit install
```

`uv sync` creates `.venv`, installs the project editable, and installs the
`dev` dependency group — development tooling lives in PEP 735
[`[dependency-groups]`](https://peps.python.org/pep-0735/) rather than in
extras, so it is never published and cannot be pulled off PyPI by a consumer.
With pip instead (25.1+ for `--group`):

```bash
pip install -e . --group dev && pre-commit install
```

The only published extra is `examples`, which the notebook needs.

```bash
pytest                  # offline suite, no network
pytest -m network       # live tests against Google and Esri
```

Run the network tests sparingly and against small AOIs — they hit the live services.

The offline suite includes upstream's own pre-computed quadtree subindex fixtures (`test/LibGoogleEarthTest/*IndexDictionary.json`), so a passing run means this port agrees with the C# implementation node-for-node.

`tests/test_examples.py` covers the notebook two ways: offline it checks that
every cell parses, that the notebook only uses exported names, that all three
public functions are still demonstrated, and that no outputs are committed;
under `-m network` it executes the notebook end to end (about 100 s on a cold
cache). Commit it with outputs cleared — executed outputs would embed fetched
Google and Esri imagery in the repository.

### Linting and types

`ruff` (lint + format) and `mypy` run on every commit via [pre-commit](https://github.com/angusmcb/old-imagery/blob/main/.pre-commit-config.yaml), and again in CI:

```bash
pre-commit run --all-files
```

The mypy hook runs from your dev install rather than an isolated environment, so the `dev` group is a prerequisite. It is a system hook and resolves `mypy` from `PATH`, so either activate the venv or run the hooks through `uv run pre-commit run --all-files`. mypy covers `src/` and `tools/`, not `tests/`.

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
