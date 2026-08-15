"""
aod_utils.py
------------
Shared MODIS MAIAC AOD extraction utilities used by both
fetch_satellite_data.py (bulk historical) and app/app.py (on-demand single-date).

Public API
----------
  latlon_to_tile_pixel(lat, lon)  -> TilePixel(h, v, row, col)
  tile_id(h, v)                   -> "hHHvVV"
  earthaccess_login()             -> None  (idempotent; reads .env)
  find_granule_for_date(tile, date_str, lookback_days)
      -> (granule_object, actual_date_str) | (None, None)
  download_granule(granule, cache_dir, retries) -> Path
  extract_aod_at_point(hdf_path, lat, lon) -> float | None
      QA-filtered mean AOD over all valid Terra+Aqua orbits at the
      nearest pixel to (lat, lon).  Returns None if no valid data.

HDF reader notes
----------------
  MCD19A2 is HDF4/HDF-EOS.  We open it with two backends in priority order:

  1. netCDF4  — works on Windows (local) where the wheel is built with HDF4.
               Raises "[Errno -128] ... feature not turned on" on the Linux pip
               wheel used by Streamlit Cloud.

  2. pyhdf   — pure HDF4 SD API; works everywhere as long as libhdf4 is
               available (packages.txt installs it on Streamlit Cloud).
               Returns raw int16; we apply scale_factor + add_offset from
               the SDS attributes and build an equivalent masked array.

  _open_hdf4(path) encapsulates this try/fallback and returns identical
  (aod_arr, qa_arr) shapes/dtypes to both callers:
    aod_arr: np.ma.MaskedArray float64 (n_orbits, 1200, 1200)
    qa_arr:  np.ndarray uint16         (n_orbits, 1200, 1200)
"""

import math
import os
import time
import pathlib
import datetime
from typing import NamedTuple

import numpy as np
import netCDF4 as nc4

# ---------------------------------------------------------------------------
# MODIS sinusoidal projection constants
# ---------------------------------------------------------------------------
RE           = 6371007.181        # authalic sphere radius (m)
TILE_DEG     = 10.0               # degrees per tile
TILE_M       = TILE_DEG * math.pi / 180.0 * RE
PIX_PER_TILE = 1200
PIX_M        = TILE_M / PIX_PER_TILE   # ~926.6 m

# AOD_QA cloud-mask bits 0-2: keep 001 (Clear) and 011 (Possibly cloudy)
KEEP_CLOUD_STATES = {0b001, 0b011}


# ---------------------------------------------------------------------------
# HDF4 reader — nc4 primary, pyhdf fallback
# ---------------------------------------------------------------------------

def _open_hdf4(
    hdf_path: pathlib.Path,
) -> tuple["np.ma.MaskedArray", "np.ndarray"]:
    """
    Open an MCD19A2 HDF4 granule and return:
      aod_arr : float64 masked array  (n_orbits, 1200, 1200)  — scaled, fill masked
      qa_arr  : uint16  ndarray       (n_orbits, 1200, 1200)  — raw QA bits

    Tries netCDF4 first (works on Windows / any build with HDF4 support).
    Falls back to pyhdf if netCDF4 raises the "feature not turned on" error
    that occurs on the Streamlit Cloud Linux pip wheel.

    Raises RuntimeError for any other open failure so callers get a clear message.
    """
    path_str = str(hdf_path)

    # ---- attempt 1: netCDF4 ----
    try:
        ds = nc4.Dataset(path_str)
        try:
            aod_arr = ds.variables["Optical_Depth_055"][:]   # masked float64
            qa_arr  = ds.variables["AOD_QA"][:].data         # uint16
        finally:
            ds.close()
        return aod_arr, qa_arr
    except Exception as nc4_err:
        # Only fall through if this looks like the "HDF4 not compiled in" error;
        # re-raise anything else (file-not-found, corrupt file, etc.).
        err_str = str(nc4_err).lower()
        if "feature" not in err_str and "not turned on" not in err_str and "errno -128" not in err_str:
            raise RuntimeError(
                f"Failed to open HDF granule {hdf_path.name}: {nc4_err}"
            ) from nc4_err
        # Fall through to pyhdf

    # ---- attempt 2: pyhdf ----
    try:
        from pyhdf.SD import SD as _SD, SDC as _SDC  # type: ignore
    except ImportError as imp_err:
        raise RuntimeError(
            f"netCDF4 cannot open HDF4 on this platform ({nc4_err}) and pyhdf "  # noqa: F821
            f"is not installed.  Add pyhdf to requirements.txt and libhdf4-dev "
            f"to packages.txt: {imp_err}"
        ) from imp_err

    try:
        f = _SD(path_str, _SDC.READ)
    except Exception as e:
        raise RuntimeError(
            f"Failed to open HDF granule {hdf_path.name} via pyhdf: {e}"
        ) from e

    try:
        aod_sds = f.select("Optical_Depth_055")
        qa_sds  = f.select("AOD_QA")

        aod_raw = np.array(aod_sds[:], dtype=np.float64)   # (n_orbits, 1200, 1200)
        qa_arr  = np.array(qa_sds[:],  dtype=np.uint16)

        attrs      = aod_sds.attributes()
        scale      = float(attrs.get("scale_factor", 1.0))
        offset     = float(attrs.get("add_offset",   0.0))
        fill_val   = int(  attrs.get("_FillValue",   -28672))

        # Apply scale; mask fill values to match netCDF4 masked-array output
        fill_mask = (aod_raw == fill_val)
        aod_phys  = np.where(fill_mask, np.nan, aod_raw * scale + offset)
        aod_arr   = np.ma.masked_invalid(aod_phys)
    finally:
        f.end()

    return aod_arr, qa_arr


class TilePixel(NamedTuple):
    h: int
    v: int
    row: int
    col: int


def latlon_to_tile_pixel(lat_deg: float, lon_deg: float) -> TilePixel:
    """Convert WGS-84 lat/lon to MODIS sinusoidal tile + pixel (row, col)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    x = RE * lon * math.cos(lat)
    y = RE * lat
    h = int((x / TILE_M) + 18)
    v = int(9 - (y / TILE_M))
    x_origin = (h - 18) * TILE_M
    y_origin = (9 - v)  * TILE_M
    col = int((x - x_origin) / PIX_M)
    row = int((y_origin - y) / PIX_M)
    return TilePixel(h=h, v=v, row=row, col=col)


def tile_id(h: int, v: int) -> str:
    return f"h{h:02d}v{v:02d}"


# ---------------------------------------------------------------------------
# Earthdata authentication
# ---------------------------------------------------------------------------

def earthaccess_login() -> None:
    """
    Login to NASA Earthdata using environment-strategy credentials.

    On Streamlit Cloud there is no .env file and no cached ~/.netrc; credentials
    are expected to already be in os.environ (placed there by app.py's bootstrap
    block via st.secrets).  Locally, app.py's load_dotenv call also pre-populates
    os.environ, so we just call earthaccess with strategy="environment" directly.

    Raises RuntimeError if the required env-vars are missing or login fails, so
    callers get a real error message rather than a silent empty result.
    """
    import earthaccess as _ea

    user = os.environ.get("EARTHDATA_USERNAME", "")
    pwd  = os.environ.get("EARTHDATA_PASSWORD", "")
    if not user or not pwd:
        raise RuntimeError(
            "EARTHDATA_USERNAME / EARTHDATA_PASSWORD not found in os.environ. "
            "Set them in Streamlit Cloud Secrets or in your local .env file."
        )
    _ea.login(strategy="environment")


# ---------------------------------------------------------------------------
# Granule search — find the most recent granule on or before a target date
# ---------------------------------------------------------------------------

def find_granule_for_date(
    tile: str,
    target_date_str: str,
    lookback_days: int = 7,
) -> tuple[object | None, str | None]:
    """
    Search for the most recent MCD19A2.061 granule for `tile` within
    `lookback_days` of `target_date_str` (inclusive).

    Returns (granule_object, date_str_of_granule) or (None, None).

    Strategy: search CMR for the full lookback window, filter filenames
    to the correct tile, pick the latest date <= target_date.
    """
    import logging as _logging
    import earthaccess as _ea

    _log = _logging.getLogger(__name__)

    target = datetime.date.fromisoformat(target_date_str)
    window_start = target - datetime.timedelta(days=lookback_days - 1)

    h = int(tile[1:3])
    v = int(tile[4:6])
    lat_min = 90 - (v + 1) * 10
    lat_max = 90 - v * 10
    lon_min = (h - 18) * 10
    lon_max = lon_min + 10

    _log.info(
        "[aod] CMR search: tile=%s temporal=(%s, %s) bbox=(%s,%s,%s,%s)",
        tile, window_start.isoformat(), target.isoformat(),
        lon_min, lat_min, lon_max, lat_max,
    )
    print(
        f"[aod] CMR search: tile={tile} temporal=({window_start.isoformat()}, "
        f"{target.isoformat()}) bbox=({lon_min},{lat_min},{lon_max},{lat_max})"
    )

    try:
        results = _ea.search_data(
            short_name="MCD19A2",
            version="061",
            temporal=(window_start.isoformat(), target.isoformat()),
            bounding_box=(lon_min, lat_min, lon_max, lat_max),
            count=1000,
        )
    except Exception as exc:
        raise RuntimeError(f"CMR granule search failed: {exc}") from exc

    _log.info("[aod] CMR returned %d raw results for tile=%s", len(results), tile)
    print(f"[aod] CMR returned {len(results)} raw results for tile={tile}")

    # Build date->granule map filtered to this tile
    by_date: dict[str, object] = {}
    for g in results:
        for url in (g.data_links() or []):
            fname = url.split("/")[-1]
            if tile not in fname:
                continue
            parts = fname.split(".")
            if len(parts) < 3:
                continue
            jd_str = parts[1]   # e.g. "A2023001"
            try:
                yr  = int(jd_str[1:5])
                doy = int(jd_str[5:8])
                d   = datetime.date(yr, 1, 1) + datetime.timedelta(days=doy - 1)
                if window_start <= d <= target:
                    by_date[d.isoformat()] = g
            except (ValueError, IndexError):
                pass
            break   # one URL per granule is enough

    if not by_date:
        return None, None

    # Pick latest date in the window
    best_date = max(by_date.keys())
    return by_date[best_date], best_date


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_granule(
    granule,
    cache_dir: pathlib.Path,
    retries: int = 3,
) -> pathlib.Path:
    """Download one granule to cache_dir. Skip if already present. Returns local path.
    Raises RuntimeError if the granule has no URLs or all download attempts fail.
    """
    import earthaccess as _ea

    urls = granule.data_links()
    if not urls:
        raise RuntimeError("Granule has no data URLs — cannot download.")
    filename = urls[0].split("/")[-1]
    dest = cache_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            files = _ea.download([granule], local_path=str(cache_dir))
            if files:
                p = pathlib.Path(files[0])
                if p.exists() and p.stat().st_size > 0:
                    return p
        except Exception as exc:
            last_exc = exc
            wait = 15 * (attempt + 1)
            if attempt < retries - 1:
                time.sleep(wait)
    raise RuntimeError(
        f"Granule download failed after {retries} attempts: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Point AOD extraction
# ---------------------------------------------------------------------------

def extract_aod_at_point(
    hdf_path: pathlib.Path,
    lat_deg: float,
    lon_deg: float,
) -> float | None:
    """
    Open one MCD19A2 HDF granule and return the QA-filtered mean AOD_055
    across all valid Terra+Aqua orbits at the pixel nearest to (lat_deg, lon_deg).

    Returns a float (>=0) on success, or None if no valid data exists at that pixel.
    """
    tp = latlon_to_tile_pixel(lat_deg, lon_deg)

    # Verify granule covers this tile
    parts = hdf_path.name.split(".")
    if len(parts) >= 3:
        fname_tile = parts[2]   # e.g. "h24v06"
        try:
            fh, fv = int(fname_tile[1:3]), int(fname_tile[4:6])
            if fh != tp.h or fv != tp.v:
                return None
        except (ValueError, IndexError):
            return None

    if not (0 <= tp.row < PIX_PER_TILE and 0 <= tp.col < PIX_PER_TILE):
        return None

    try:
        aod_arr, qa_arr = _open_hdf4(hdf_path)
    except RuntimeError:
        return None

    valid_aods: list[float] = []
    n_orbits = aod_arr.shape[0]
    for orbit_idx in range(n_orbits):
        val = aod_arr[orbit_idx, tp.row, tp.col]
        if np.ma.is_masked(val):
            continue
        aod_f = float(val)
        if aod_f < -0.05:
            continue
        qa_val = int(qa_arr[orbit_idx, tp.row, tp.col])
        if qa_val == 0:
            continue
        if (qa_val & 0b111) not in KEEP_CLOUD_STATES:
            continue
        valid_aods.append(aod_f)

    if not valid_aods:
        return None
    return round(sum(valid_aods) / len(valid_aods), 4)
