"""old-imagery — historical aerial imagery from Google Earth and Esri Wayback.

Copyright (C) 2026 the old-imagery contributors

This program is free software: you can redistribute it and/or modify it under
the terms of version 3 of the GNU General Public License as published by the
Free Software Foundation. It is distributed WITHOUT ANY WARRANTY, without even
the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License (LICENSE) for details.

Ported from Mbucari/GEHistoricalImagery (GPL-3.0-or-later); see NOTICE for
attribution. The combined old-imagery work is GPL-3.0-only.
This library retrieves imagery but does not license it — imagery remains the
copyright of Google, Esri or their providers, and their terms of service apply.

A Python port of the protocol layer of Mbucari/GEHistoricalImagery, exposing
two functions for use from Python rather than a command line tool:

>>> import old_imagery
>>> from shapely.geometry import box
>>> aoi = box(-122.404, 37.792, -122.396, 37.798)
>>> dates = old_imagery.availability(aoi, zoom=18)                 # doctest: +SKIP
>>> img = old_imagery.download(aoi, zoom=18, date="1993-07-10")    # doctest: +SKIP

The ``date`` argument and every date this package reports is an **image capture
date**, never a provider's publication or release date. Esri Wayback snapshots
are selected separately and explicitly by stable release identifier or an
as-of catalogue date.
"""

from ._dbroot import Database, DatedTile, DbRoot
from ._http import DEFAULT_CACHE_DIR, CachedHttpClient, NotFound, RequestFailed
from ._keyhole import KeyholeTile
from .api import ESRI_REGION_QUERY_MIN_TILES, MAX_IMAGERY_ZOOM, availability, download

__version__ = "0.1.0"

__all__ = [
    "availability",
    "download",
    "ESRI_REGION_QUERY_MIN_TILES",
    "MAX_IMAGERY_ZOOM",
    "KeyholeTile",
    "DbRoot",
    "Database",
    "DatedTile",
    "CachedHttpClient",
    "RequestFailed",
    "NotFound",
    "DEFAULT_CACHE_DIR",
    "__version__",
]
