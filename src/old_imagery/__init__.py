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
three functions for use from Python rather than a command line tool:

>>> import old_imagery
>>> from shapely.geometry import box
>>> aoi = box(-122.404, 37.792, -122.396, 37.798)
>>> dates = old_imagery.availability(aoi, zoom=18)                 # doctest: +SKIP
>>> img = old_imagery.download(aoi, zoom=18, date="1993-07-10")    # doctest: +SKIP

``availability`` and ``download`` answer "which capture dates exist here?" and
"give me the pixels for one of them". The third function answers a different
question -- what a published Esri Wayback snapshot *displays*, and how old each
part of it is:

>>> seams = old_imagery.esri_mosaic_as_of(aoi, 18, "2020-06-01")   # doctest: +SKIP

The ``date`` argument and every date this package reports is an **image capture
date**, never a provider's publication or release date. The one date that means
publication is ``esri_mosaic_as_of``'s ``as_of_date``, which selects a release
from Esri's catalogue and is named to say so.
"""

from ._http import DEFAULT_CACHE_DIR, NotFound, RequestFailed
from .api import (
    MAX_IMAGERY_ZOOM,
    DateMatch,
    Provider,
    availability,
    download,
    esri_mosaic_as_of,
)

__version__ = "0.1.0"

# Deliberately narrow: the two documented functions, the constants and option
# types their signatures name, the exceptions they raise, and the version.
#
# The protocol layer (DbRoot, Database, DatedTile, KeyholeTile,
# CachedHttpClient) is reachable from the underscore modules but is not
# exported and is not a supported interface -- it tracks Google's and Esri's
# wire formats, so it changes when they do. Anything here that proves genuinely
# useful can be promoted into a named advanced module later; promoting is
# cheap, and withdrawing something callers already import is not.
__all__ = [
    # public API
    "availability",
    "download",
    "esri_mosaic_as_of",
    # option types
    "Provider",
    "DateMatch",
    # constants
    "MAX_IMAGERY_ZOOM",
    "DEFAULT_CACHE_DIR",
    # exceptions
    "RequestFailed",
    "NotFound",
    "__version__",
]
