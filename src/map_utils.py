"""
map_utils.py
------------
Regional PM2.5 map pipeline: extract AOD across a downsampled tile grid,
fetch weather for the grid, run the model on every valid cell, and return
a list of dicts ready for Plotly rendering.

Public API
----------
  REGIONS          -- dict of region metadata (name, tile_id, centre, etc.)
  build_region_map(region_key, date_str, model_bundle, progress_cb=None)
      -> (cells, exposure, advisory)
         cells: list[dict]  each dict: {lat, lon, aod, pm25, band, band_label, band_color}
         Returns ([], {available:False}, {}) if granule unavailable (cloud/data gap).
         Raises RuntimeError on auth, download, or HDF failures.
  load_cached_map(region_key, date_str) -> list[dict] | None
  save_cached_map(region_key, date_str, cells)

Grid design
-----------
  Tile is 1200×1200 pixels. We sample every GRID_STEP pixels so the
  downsampled grid is ~50×50 cells.  Each cell represents a ~22 km × 22 km
  area.  The AOD value is the QA-filtered mean over all orbits at that pixel.
  Cells where no orbit has valid QA-filtered data are omitted (cloud/gap).

Weather
-------
  We build a coarse 1°-resolution weather grid over the tile, fetch all
  points in a single batch Open-Meteo request, then assign each map cell
  its nearest weather point (Euclidean distance in degrees — good enough
  at this scale).
"""

import csv
import datetime
import json
import logging
import math
import pathlib
import os
import sys
import tempfile
import time

logger = logging.getLogger(__name__)

import requests
import numpy as np
import netCDF4 as nc4
from global_land_mask import globe as _globe

# ---------------------------------------------------------------------------
# Shared geometry from aod_utils
# ---------------------------------------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from aod_utils import (
    RE, TILE_M, PIX_M, PIX_PER_TILE,
    KEEP_CLOUD_STATES,
    latlon_to_tile_pixel, tile_id,
    earthaccess_login, find_granule_for_date, download_granule,
)
from population_utils import (
    load_population_grid, compute_exposure, exposure_headline,
    ensure_population_grid,
)
from advisory_utils import build_advisory_context, generate_advisory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_STEP = 24          # sample 1 pixel every GRID_STEP -> 50x50 grid

# Land-mask: exclude over-water cells — model was trained on land monitors only
def _is_land(lat: float, lon: float) -> bool:
    return bool(_globe.is_land(lat, lon))

# Step for the weather grid in degrees
WEATHER_DEG_STEP = 1.0  # degrees between weather query points

MAP_CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed" / "map_cache"

# Health bands — MUST match app.py CATEGORIES thresholds
HEALTH_BANDS = [
    (  0,  12,  "Good",                        "#22c55e"),
    ( 12,  35,  "Moderate",                    "#84cc16"),
    ( 35,  55,  "USG",                         "#eab308"),
    ( 55, 150,  "Unhealthy",                   "#f97316"),
    (150, 250,  "Very Unhealthy",              "#ef4444"),
    (250, float("inf"), "Hazardous",           "#7c3aed"),
]

WEATHER_VARS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "precipitation_sum",
    "surface_pressure_mean",
    "shortwave_radiation_sum",
]

# ---------------------------------------------------------------------------
# Region registry
# ---------------------------------------------------------------------------
# Each region maps to one MODIS tile.
# centre_lat/lon is the map zoom centre; label shown in the UI.
# n_land is computed at module load (fast in-memory land-mask lookup).
REGIONS = {
    "h24v06": {
        "label":      "Delhi / Karachi region (NW India + Pakistan)",
        "tile":       "h24v06",
        "centre_lat": 26.0,
        "centre_lon": 72.0,
        "zoom":       4.5,
        "validated_cities": [
            {"name": "Delhi",   "lat": 28.6139, "lon": 77.2090},
            {"name": "Karachi", "lat": 24.8607, "lon": 67.0011},
        ],
        # Approximate bounding box for the tile (lat_min, lat_max, lon_min, lon_max)
        # Used to scope the no-data population estimate
        "bbox": (20.0, 30.5, 66.0, 82.0),
    },
    "h24v07": {
        "label":      "Mumbai / Central India region",
        "tile":       "h24v07",
        "centre_lat": 17.5,
        "centre_lon": 74.0,
        "zoom":       4.5,
        "validated_cities": [
            {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
        ],
        "bbox": (10.0, 20.5, 66.0, 82.0),
    },
}

# Populate n_land for each region (counted once at import time).
# pixel_to_latlon is defined below — we use a forward reference via a helper
# that we fill in after the function is defined.
def _count_land_cells(tile: str, step: int = GRID_STEP) -> int:
    h = int(tile[1:3]); v = int(tile[4:6])
    return sum(
        1
        for r in range(0, PIX_PER_TILE, step)
        for c in range(0, PIX_PER_TILE, step)
        if _is_land(*pixel_to_latlon(h, v, r, c))
    )

# ---------------------------------------------------------------------------
# Sinusoidal -> lat/lon inverse transform
# ---------------------------------------------------------------------------

def pixel_to_latlon(h_tile: int, v_tile: int, row: int, col: int) -> tuple[float, float]:
    """
    Convert a MODIS sinusoidal tile pixel (row, col) to WGS-84 (lat, lon).
    Uses the pixel-centre coordinate.
    """
    x_origin = (h_tile - 18) * TILE_M
    y_origin = (9 - v_tile)  * TILE_M
    # Pixel centre
    x = x_origin + (col + 0.5) * PIX_M
    y = y_origin - (row + 0.5) * PIX_M
    lat = math.degrees(y / RE)
    cos_lat = math.cos(y / RE)
    if abs(cos_lat) < 1e-10:
        lon = 0.0
    else:
        lon = math.degrees(x / (RE * cos_lat))
    return lat, lon


# n_land is now safe to compute (pixel_to_latlon is defined above).
for _rk, _rv in REGIONS.items():
    _rv["n_land"] = _count_land_cells(_rv["tile"])

# ---------------------------------------------------------------------------
# AOD grid extraction from one HDF granule
# ---------------------------------------------------------------------------

def extract_aod_grid(
    hdf_path: pathlib.Path,
    h_tile: int,
    v_tile: int,
    step: int = GRID_STEP,
) -> list[dict]:
    """
    Open one MCD19A2 HDF granule and extract QA-filtered AOD at every
    `step`-th pixel across both dimensions.

    Returns list of dicts:
      {row, col, lat, lon, aod}  — only pixels with at least one valid orbit.
    """
    try:
        ds = nc4.Dataset(str(hdf_path))
    except Exception as e:
        raise RuntimeError(f"Failed to open HDF granule {hdf_path.name}: {e}") from e

    cells: list[dict] = []
    try:
        aod_arr = ds.variables["Optical_Depth_055"][:]    # (2,1200,1200) masked float64
        qa_arr  = ds.variables["AOD_QA"][:].data          # (2,1200,1200) uint16 raw
        n_orbits = aod_arr.shape[0]

        row_indices = range(0, PIX_PER_TILE, step)
        col_indices = range(0, PIX_PER_TILE, step)

        for row in row_indices:
            for col in col_indices:
                lat, lon = pixel_to_latlon(h_tile, v_tile, row, col)
                if not _is_land(lat, lon):
                    continue   # skip ocean / sea cells

                valid_aods: list[float] = []
                for orbit in range(n_orbits):
                    val = aod_arr[orbit, row, col]
                    if np.ma.is_masked(val):
                        continue
                    aod_f = float(val)
                    if aod_f < -0.05:
                        continue
                    qa_val = int(qa_arr[orbit, row, col])
                    if qa_val == 0:
                        continue
                    if (qa_val & 0b111) not in KEEP_CLOUD_STATES:
                        continue
                    valid_aods.append(aod_f)

                if not valid_aods:
                    continue   # cloud / no data — skip cell

                mean_aod = round(sum(valid_aods) / len(valid_aods), 4)
                cells.append({
                    "row": row,
                    "col": col,
                    "lat": round(lat, 4),
                    "lon": round(lon, 4),
                    "aod": mean_aod,
                })
    finally:
        ds.close()

    return cells

# ---------------------------------------------------------------------------
# Weather fetch — batch Open-Meteo for the whole grid
# ---------------------------------------------------------------------------

def _build_weather_grid(
    h_tile: int,
    v_tile: int,
    deg_step: float = WEATHER_DEG_STEP,
) -> list[tuple[float, float]]:
    """
    Build a coarse lat/lon grid over the tile at `deg_step` resolution.
    Returns list of (lat, lon) weather query points.
    """
    x_origin = (h_tile - 18) * TILE_M
    y_origin = (9 - v_tile)  * TILE_M
    # Approximate tile lat/lon extent from corners
    lat_max, _ = pixel_to_latlon(h_tile, v_tile, 0,               0)
    lat_min, _ = pixel_to_latlon(h_tile, v_tile, PIX_PER_TILE - 1, 0)
    _, lon_min  = pixel_to_latlon(h_tile, v_tile, PIX_PER_TILE // 2, 0)
    _, lon_max  = pixel_to_latlon(h_tile, v_tile, PIX_PER_TILE // 2, PIX_PER_TILE - 1)

    # Floor/ceil to degree grid
    lat_min_g = math.floor(lat_min / deg_step) * deg_step
    lat_max_g = math.ceil( lat_max / deg_step) * deg_step
    lon_min_g = math.floor(lon_min / deg_step) * deg_step
    lon_max_g = math.ceil( lon_max / deg_step) * deg_step

    points: list[tuple[float, float]] = []
    lat = lat_min_g
    while lat <= lat_max_g + 1e-9:
        lon = lon_min_g
        while lon <= lon_max_g + 1e-9:
            points.append((round(lat, 4), round(lon, 4)))
            lon += deg_step
        lat += deg_step
    return points


def fetch_weather_batch(
    points: list[tuple[float, float]],
    date_str: str,
    max_per_request: int = 100,
) -> dict[tuple[float, float], dict]:
    """
    Fetch weather for all (lat, lon) points in `points` using Open-Meteo batch.
    Returns {(lat, lon): {var: value, ...}}.
    Splits into chunks of max_per_request to stay under API limits.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    result: dict[tuple[float, float], dict] = {}

    for chunk_start in range(0, len(points), max_per_request):
        chunk = points[chunk_start : chunk_start + max_per_request]
        params: list[tuple] = []
        for lat, lon in chunk:
            params.append(("latitude",  lat))
            params.append(("longitude", lon))
        params += [
            ("start_date", date_str),
            ("end_date",   date_str),
            ("daily",      ",".join(WEATHER_VARS)),
            ("timezone",   "auto"),
        ]
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            responses = r.json()
        except Exception as e:
            print(f"  [weather batch error] {e}")
            continue

        # Response can be a single dict or a list depending on count
        if isinstance(responses, dict):
            responses = [responses]

        for i, resp in enumerate(responses):
            if i >= len(chunk):
                break
            pt = chunk[i]
            daily = resp.get("daily") or {}
            wx: dict = {}
            for var in WEATHER_VARS:
                vals = daily.get(var) or [None]
                wx[var] = vals[0] if vals else None
            result[pt] = wx

        if chunk_start + max_per_request < len(points):
            time.sleep(0.5)   # brief pause between chunks

    return result


def assign_weather_to_cells(
    cells: list[dict],
    weather_map: dict[tuple[float, float], dict],
) -> None:
    """
    For each cell, find the nearest weather grid point and attach weather vars.
    Modifies cells in-place.
    """
    wx_points = list(weather_map.keys())
    if not wx_points:
        for cell in cells:
            for var in WEATHER_VARS:
                cell[var] = None
        return

    for cell in cells:
        clat, clon = cell["lat"], cell["lon"]
        # Nearest by Euclidean distance in degrees (fine at ~10° scale)
        best = min(wx_points, key=lambda p: (p[0] - clat) ** 2 + (p[1] - clon) ** 2)
        wx = weather_map[best]
        for var in WEATHER_VARS:
            cell[var] = wx.get(var)

# ---------------------------------------------------------------------------
# PM2.5 prediction and band assignment
# ---------------------------------------------------------------------------

SEASON_ENCODE = {"winter": 0, "spring": 1, "monsoon": 2, "autumn": 3}

def _season(month: int) -> str:
    if month in (12, 1, 2):   return "winter"
    if month in (3, 4, 5):    return "spring"
    if month in (6, 7, 8, 9): return "monsoon"
    return "autumn"


def _pm25_to_band(pm25: float) -> tuple[str, str]:
    """Return (band_label, band_color)."""
    for lo, hi, label, color in HEALTH_BANDS:
        if lo <= pm25 < hi:
            return label, color
    return HEALTH_BANDS[-1][2], HEALTH_BANDS[-1][3]


def predict_grid(cells: list[dict], model_bundle: dict, date: datetime.date) -> list[dict]:
    """
    Run the model on every cell.  Adds 'pm25', 'band_label', 'band_color' to each.
    Returns the list (modified in place and returned for convenience).
    """
    features = model_bundle["features"]
    model    = model_bundle["model"]

    month   = date.month
    doy     = date.timetuple().tm_yday
    dow     = date.weekday()
    seas    = SEASON_ENCODE[_season(month)]

    DEFAULT_WX = {
        "temperature_2m_mean":       25.0,
        "relative_humidity_2m_mean": 50.0,
        "wind_speed_10m_mean":        3.0,
        "precipitation_sum":          0.0,
        "surface_pressure_mean":   1010.0,
        "shortwave_radiation_sum":   15.0,
    }

    rows = []
    for cell in cells:
        row = [
            cell["aod"],
            month, doy, dow, seas,
            cell.get("temperature_2m_mean")       or DEFAULT_WX["temperature_2m_mean"],
            cell.get("relative_humidity_2m_mean") or DEFAULT_WX["relative_humidity_2m_mean"],
            cell.get("wind_speed_10m_mean")        or DEFAULT_WX["wind_speed_10m_mean"],
            cell.get("precipitation_sum")          or DEFAULT_WX["precipitation_sum"],
            cell.get("surface_pressure_mean")      or DEFAULT_WX["surface_pressure_mean"],
            cell.get("shortwave_radiation_sum")    or DEFAULT_WX["shortwave_radiation_sum"],
        ]
        rows.append(row)

    if not rows:
        return cells

    X    = np.array(rows, dtype=np.float64)
    preds = np.expm1(model.predict(X))
    preds = np.clip(preds, 0, None)

    for cell, pm25 in zip(cells, preds):
        pm25_f = float(pm25)
        band_label, band_color = _pm25_to_band(pm25_f)
        cell["pm25"]       = round(pm25_f, 1)
        cell["band_label"] = band_label
        cell["band_color"] = band_color

    return cells

# ---------------------------------------------------------------------------
# Cache  (JSON, keyed by region+date)
# ---------------------------------------------------------------------------

def _cache_path(region_key: str, date_str: str) -> pathlib.Path:
    MAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return MAP_CACHE_DIR / f"{region_key}_{date_str}.json"


def load_cached_map(region_key: str, date_str: str) -> tuple[list[dict], dict, dict] | None:
    """
    Load cached map data.
    Returns (cells, exposure, advisory) tuple, or None if no cache.
    Backward compatible: old caches (plain list) return empty exposure+advisory.
    """
    p = _cache_path(region_key, date_str)
    if p.exists():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return raw, {"available": False}, {}
            elif isinstance(raw, dict) and "cells" in raw:
                return (raw["cells"],
                        raw.get("exposure", {"available": False}),
                        raw.get("advisory", {}))
        except Exception:
            pass
    return None


def save_cached_map(region_key: str, date_str: str,
                    payload: list[dict] | dict) -> None:
    """Save map cache.  payload is a dict with cells, exposure, and optional advisory."""
    if isinstance(payload, list):
        payload = {"cells": payload, "exposure": {"available": False}, "advisory": {}}
    _cache_path(region_key, date_str).write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# Use the system temp dir — guaranteed writable on every platform (local and cloud).
HDF_TMP = pathlib.Path(tempfile.gettempdir()) / "skywatch_map_hdf"


def build_region_map(
    region_key: str,
    date_str: str,
    model_bundle: dict,
    progress_cb=None,   # callable(str) for status messages
) -> tuple[list[dict], dict, dict]:
    """
    Full pipeline: find granule -> download -> extract AOD grid ->
    fetch weather -> predict -> cache -> return (cells, exposure, advisory).

    Returns ([], {"available": False}, {}) when no granule is available for the date
    (genuine cloud gap or archive data gap — NOT an error).

    Raises RuntimeError for hard failures (auth, download, HDF open) so the caller
    can surface the real error message instead of silently showing "No data".
    """
    def log(msg: str) -> None:
        print(msg)
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

    region = REGIONS[region_key]
    tile   = region["tile"]
    h_tile = int(tile[1:3])
    v_tile = int(tile[4:6])
    date   = datetime.date.fromisoformat(date_str)

    log(f"[map] Region: {region['label']}  |  Date: {date_str}")

    # 1. Find + download granule — let exceptions propagate so app.py can display them
    log("[map] Authenticating with NASA Earthdata...")
    earthaccess_login()   # raises RuntimeError with a clear message on failure

    log("[map] Searching for MCD19A2 granule (exact date)...")
    granule, g_date = find_granule_for_date(tile, date_str, lookback_days=1)
    if granule is None:
        log("[map] No granule found for this date (cloud gap or data missing).")
        return [], {"available": False}, {}
    log(f"[map] Found granule for {g_date}. Downloading...")

    HDF_TMP.mkdir(parents=True, exist_ok=True)
    hdf_path = download_granule(granule, HDF_TMP)   # raises RuntimeError on failure
    log(f"[map] Downloaded: {hdf_path.name} ({hdf_path.stat().st_size // 1024:,} KB)")

    # 2. Extract AOD grid
    log(f"[map] Extracting AOD grid (step={GRID_STEP}, ~50x50 cells)...")
    cells = extract_aod_grid(hdf_path, h_tile, v_tile, step=GRID_STEP)
    n_land = region.get("n_land", (PIX_PER_TILE // GRID_STEP) ** 2)
    log(f"[map] Valid cells (land, non-cloudy): {len(cells):,} / {n_land} land cells")

    # Delete HDF to keep disk footprint low
    try:
        hdf_path.unlink()
        log("[map] HDF file deleted.")
    except OSError:
        pass

    if not cells:
        log("[map] All cells cloudy — no valid AOD for this date.")
        return [], {"available": False}, {}

    # 3. Weather for the grid
    log("[map] Building weather grid and fetching from Open-Meteo (batch)...")
    wx_points = _build_weather_grid(h_tile, v_tile, WEATHER_DEG_STEP)
    log(f"[map] Fetching weather for {len(wx_points)} grid points...")
    weather_map = fetch_weather_batch(wx_points, date_str)
    log(f"[map] Weather received for {len(weather_map)}/{len(wx_points)} points.")
    assign_weather_to_cells(cells, weather_map)

    # 4. Predict
    log(f"[map] Predicting PM2.5 for {len(cells):,} cells...")
    cells = predict_grid(cells, model_bundle, date)
    log(f"[map] Prediction complete.  PM2.5 range: "
        f"{min(c['pm25'] for c in cells):.1f} – {max(c['pm25'] for c in cells):.1f} ug/m3")

    # 5. Exposure computation
    pop_grid = load_population_grid()
    if pop_grid is not None:
        bbox = region.get("bbox")
        exposure = compute_exposure(cells, pop_grid, region_bbox=bbox)
        log(f"[map] Exposure computed: {exposure_headline(exposure)}")
    else:
        exposure = {"available": False}
        log("[map] Population grid not available — skipping exposure.")

    # 6. Advisory generation
    log("[map] Generating health advisory...")
    try:
        adv_ctx = build_advisory_context(cells, exposure, region_key, date_str,
                                         is_forecast=False)
        advisory = generate_advisory(adv_ctx)
        log(f"[map] Advisory source: {advisory['source']}")
    except Exception as e:
        log(f"[map] Advisory generation failed: {e}")
        advisory = {}

    # 7. Cache (cells + exposure + advisory)
    cache_payload = {"cells": cells, "exposure": exposure, "advisory": advisory}
    save_cached_map(region_key, date_str, cache_payload)
    log(f"[map] Cached to {_cache_path(region_key, date_str).name}")

    return cells, exposure, advisory
