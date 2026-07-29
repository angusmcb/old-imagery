# Test fixtures

## `RootIndexDictionary.json`, `SubIndexDictionary.json`

Copied verbatim from [GEHistoricalImagery](https://github.com/Mbucari/GEHistoricalImagery),
`test/LibGoogleEarthTest/`, and **copyright Mbucari and contributors, GPL-3.0-or-later**.
They are redistributed here under that licence, as is the rest of this project — see
[NOTICE](../../NOTICE).

They map quadtree paths to the subindex values the C# implementation computes.
Using upstream's own fixtures unmodified is deliberate: it means a passing
`tests/test_keyhole.py` run is evidence that this port agrees with the C#
implementation node-for-node, rather than agreeing only with itself.

Do not regenerate these from this codebase — that would make the check circular.
Re-copy them from upstream if they ever need updating.

## `wayback_capabilities_sample.xml`

A trimmed WMTS capabilities response from Esri's World Imagery Wayback service,
captured from

    https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml

and cut down to a handful of layers. Used to test release parsing offline. This
is Esri service metadata describing available layers — no imagery.
