"""
forecast_utils.py
-----------------
FORECAST mode: "tomorrow's air, today's warning."

Strategy
--------
* Weather  — Open-Meteo *forecast* API (same variables as archive).
  Supports today+1 and today+2 (and further, though skill degrades).
* AOD      — no forecast exists for satellite aerosol.  We use the most
  recent QA-valid AOD per grid cell found within the last AOD_LOOKBACK_DAYS
  days.  This is the "persistence" assumption: recent aerosol conditions
  are the best proxy we have.  Clearly labelled in the UI.
* Pipeline — reuses map_utils.extract_aod_grid / predict_grid / cache
  infrastructure verbatim.  Only the weather fetch is swapped.

Public API
----------
  is_forecast_date(date_str)          -> bool
  fetch_forecast_weather_batch(points, date_str) -> same shape as fetch_weather_batch
  build_forecast_map(region_key, date_str, model_bundle, progress_cb)
      -> (cells, exposure) | ([], {"available": False})
  load_forecast_cache(region_key, date_str)  -> (cells, exposure) | None
      Only returns a hit if cached on the same calendar day (forecasts go stale).

Uncertainty note (shown in UI):
  "Forecast uses latest available satellite AOD (persistence assumption,
   up to {AOD_LOOKBACK_DAYS} days old) + forecast weather — uncertainty is
   substantially higher than same-day estimates."
"""

import datetime
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Bootstrap path so we can import sibling modules
# ---------------------------------------------------------------------------
_SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

from map_utils import (
    REGIONS, GRID_STEP, WEATHER_VARS, WEATHER_DEG_STEP,
    MAP_CACHE_DIR, HDF_TMP,
    _build_weather_grid, assign_weather_to_cells,
    extract_aod_grid, predict_grid,
    pixel_to_latlon, _is_land, PIX_PER_TILE,
    save_cached_map,
)
from aod_utils import (
    earthaccess_login, find_granule_for_date, download_granule,
)
from population_utils import (
    load_population_grid, compute_exposure, exposure_headline,
)
from advisory_utils import build_advisory_context, generate_advisory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
AOD_LOOKBACK_DAYS = 5    # search this many days back for a valid satellite pass
FORECAST_CACHE_DIR = MAP_CACHE_DIR   # same folder, different key prefix

# How far ahead we support (days from today)
MAX_FORECAST_DAYS = 2


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def is_forecast_date(date_str: str) -> bool:
    """Return True if date_str is today or in the future."""
    return datetime.date.fromisoformat(date_str) >= datetime.date.today()


def forecast_label(date_str: str) -> str:
    """Return 'Tomorrow', 'In 2 days', or 'Today' prefix for UI."""
    delta = (datetime.date.fromisoformat(date_str) - datetime.date.today()).days
    if delta == 1:
        return "Tomorrow"
    elif delta == 2:
        return "In 2 days"
    elif delta == 0:
        return "Today"
    return f"In {delta} days"


# ---------------------------------------------------------------------------
# Forecast weather — Open-Meteo forecast API
# ---------------------------------------------------------------------------

def fetch_forecast_weather_batch(
    points: list[tuple[float, float]],
    date_str: str,
    max_per_request: int = 100,   # kept for API compatibility, not used (single-point calls)
) -> dict[tuple[float, float], dict]:
    """
    Fetch forecast weather for all (lat, lon) points using Open-Meteo forecast API.
    Returns {(lat, lon): {var: value, ...}}.

    The Open-Meteo forecast endpoint does not support multi-location batching,
    so we deduplicate points to a coarser 1° grid and make individual requests.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    result: dict[tuple[float, float], dict] = {}

    # Deduplicate: round to 1° grid to minimise requests (~100–200 for a tile)
    deg = 1.0
    unique: dict[tuple[float, float], list[tuple[float, float]]] = {}
    for pt in points:
        key = (round(pt[0] / deg) * deg, round(pt[1] / deg) * deg)
        unique.setdefault(key, []).append(pt)

    for i, (grid_pt, orig_pts) in enumerate(unique.items()):
        params = {
            "latitude":   grid_pt[0],
            "longitude":  grid_pt[1],
            "start_date": date_str,
            "end_date":   date_str,
            "daily":      ",".join(WEATHER_VARS),
            "timezone":   "auto",
        }
        try:
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            daily = r.json().get("daily") or {}
            wx: dict = {}
            for var in WEATHER_VARS:
                vals = daily.get(var) or [None]
                wx[var] = vals[0] if vals else None
            for pt in orig_pts:
                result[pt] = wx
        except Exception as e:
            print(f"  [forecast weather error] {grid_pt}: {e}")
            continue

        if i % 20 == 19:
            time.sleep(0.3)   # brief pause every 20 requests

    return result


# ---------------------------------------------------------------------------
# Single-point forecast weather (for City estimate tab)
# ---------------------------------------------------------------------------

def fetch_forecast_weather_point(lat: float, lon: float, date_str: str) -> dict | None:
    """
    Fetch forecast weather for a single (lat, lon) point.
    Returns {var: value} dict or None on failure.
    """
    result = fetch_forecast_weather_batch([(lat, lon)], date_str, max_per_request=1)
    return result.get((lat, lon))


# ---------------------------------------------------------------------------
# AOD persistence: build a grid of recent AOD values for a tile
# ---------------------------------------------------------------------------

def build_persisted_aod_grid(
    region_key: str,
    forecast_date_str: str,
    progress_cb=None,
) -> tuple[list[dict], str | None]:
    """
    Find the most recent satellite granule within the last AOD_LOOKBACK_DAYS
    days before `forecast_date_str` and extract AOD for all land cells.

    Returns (cells_with_aod, actual_aod_date_str) or ([], None).
    `cells_with_aod` has the same structure as extract_aod_grid output:
      [{row, col, lat, lon, aod}, ...]
    """
    def log(msg: str) -> None:
        print(msg)
        if progress_cb:
            progress_cb(msg)

    region = REGIONS[region_key]
    tile   = region["tile"]
    h_tile = int(tile[1:3])
    v_tile = int(tile[4:6])

    # We look back from (forecast_date - 1) since forecast_date has no sat data yet
    lookback_end = (datetime.date.fromisoformat(forecast_date_str)
                    - datetime.timedelta(days=1)).isoformat()

    log(f"[forecast] Searching for recent AOD (up to {AOD_LOOKBACK_DAYS} days before {forecast_date_str})...")
    earthaccess_login()   # raises RuntimeError with a clear message on failure

    granule, aod_date = find_granule_for_date(tile, lookback_end, lookback_days=AOD_LOOKBACK_DAYS)
    if granule is None:
        log("[forecast] No recent granule found — cannot build AOD persistence grid.")
        return [], None

    log(f"[forecast] Using AOD from {aod_date} (persistence assumption). Downloading...")
    HDF_TMP.mkdir(parents=True, exist_ok=True)
    hdf_path = download_granule(granule, HDF_TMP)   # raises RuntimeError on failure
    log(f"[forecast] Downloaded: {hdf_path.name} ({hdf_path.stat().st_size // 1024:,} KB)")

    cells = extract_aod_grid(hdf_path, h_tile, v_tile, step=GRID_STEP)
    log(f"[forecast] Persisted AOD cells: {len(cells):,}")

    try:
        hdf_path.unlink()
    except OSError:
        pass

    return cells, aod_date


# ---------------------------------------------------------------------------
# Forecast cache: keyed by region+date+build_date so stale entries are ignored
# ---------------------------------------------------------------------------

def _forecast_cache_path(region_key: str, date_str: str) -> pathlib.Path:
    FORECAST_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return FORECAST_CACHE_DIR / f"{region_key}_{date_str}_forecast.json"


def load_forecast_cache(region_key: str, date_str: str) -> tuple[list[dict], dict, dict] | None:
    """
    Load forecast cache only if it was built today.
    Returns (cells, exposure, advisory) or None.
    """
    p = _forecast_cache_path(region_key, date_str)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        # Stale check: only use if built today
        built_on = raw.get("built_on", "")
        if built_on != datetime.date.today().isoformat():
            return None
        return (raw.get("cells", []),
                raw.get("exposure", {"available": False}),
                raw.get("advisory", {}))
    except Exception:
        return None


def _save_forecast_cache(region_key: str, date_str: str,
                         cells: list[dict], exposure: dict,
                         aod_date: str | None, advisory: dict) -> None:
    payload = {
        "cells":     cells,
        "exposure":  exposure,
        "advisory":  advisory,
        "aod_date":  aod_date,
        "built_on":  datetime.date.today().isoformat(),
    }
    _forecast_cache_path(region_key, date_str).write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main forecast pipeline
# ---------------------------------------------------------------------------

def build_forecast_map(
    region_key: str,
    date_str: str,
    model_bundle: dict,
    progress_cb=None,
) -> tuple[list[dict], dict, str | None]:
    """
    Build a forecast PM2.5 map for a future date.

    Steps:
      1. Persisted AOD: find the most recent satellite granule (<= today-1),
         download and extract the AOD grid.
      2. Forecast weather: fetch Open-Meteo forecast for the target date.
      3. Predict PM2.5 using the same model.
      4. Compute population exposure.
      5. Cache (with today's build date; goes stale after midnight).

    Returns (cells, exposure, aod_date_str).
    aod_date_str is the date the AOD came from (for provenance label).
    Returns ([], {"available": False}, None) on failure.
    """
    def log(msg: str) -> None:
        print(msg)
        if progress_cb:
            progress_cb(msg)

    region = REGIONS[region_key]
    tile   = region["tile"]
    h_tile = int(tile[1:3])
    v_tile = int(tile[4:6])
    date   = datetime.date.fromisoformat(date_str)

    log(f"[forecast] Building forecast map — {region['label']}  |  Target: {date_str}")

    # 1. Persisted AOD grid
    cells, aod_date = build_persisted_aod_grid(region_key, date_str, progress_cb=progress_cb)
    if not cells:
        log("[forecast] No persisted AOD available — cannot build forecast map.")
        return [], {"available": False}, None

    # 2. Forecast weather
    log(f"[forecast] Fetching forecast weather for {date_str} from Open-Meteo...")
    wx_points   = _build_weather_grid(h_tile, v_tile, WEATHER_DEG_STEP)
    weather_map = fetch_forecast_weather_batch(wx_points, date_str)
    log(f"[forecast] Weather received for {len(weather_map)}/{len(wx_points)} points.")
    assign_weather_to_cells(cells, weather_map)

    # 3. Predict
    log(f"[forecast] Predicting PM2.5 for {len(cells):,} cells...")
    cells = predict_grid(cells, model_bundle, date)
    log(f"[forecast] Done. PM2.5 range: "
        f"{min(c['pm25'] for c in cells):.1f} – {max(c['pm25'] for c in cells):.1f} µg/m³")

    # 4. Exposure
    pop_grid = load_population_grid()
    if pop_grid is not None:
        bbox     = region.get("bbox")
        exposure = compute_exposure(cells, pop_grid, region_bbox=bbox)
        log(f"[forecast] Exposure: {exposure_headline(exposure, suffix='')}")
    else:
        exposure = {"available": False}

    # 5. Advisory
    log("[forecast] Generating health advisory...")
    try:
        adv_ctx  = build_advisory_context(cells, exposure, region_key, date_str,
                                          is_forecast=True)
        advisory = generate_advisory(adv_ctx)
        log(f"[forecast] Advisory source: {advisory['source']}")
    except Exception as e:
        log(f"[forecast] Advisory generation failed: {e}")
        advisory = {}

    # 6. Cache
    _save_forecast_cache(region_key, date_str, cells, exposure, aod_date, advisory)
    log("[forecast] Cached (valid today only).")

    return cells, exposure, aod_date, advisory
