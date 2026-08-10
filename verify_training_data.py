"""
verify_training_data.py
-----------------------
Picks 5 random rows from data/processed/training_data.csv and independently
re-fetches / re-derives each stored value from its primary source:

  PM2.5       -- OpenAQ v3 /sensors/{id}/days  (same sensor, same calendar day)
  Weather     -- Open-Meteo archive API         (same lat/lon/date, 6 variables)
  AOD         -- NASA MCD19A2 via earthaccess   (same tile/pixel/date)

Reports a pass/fail table with stored vs. re-derived values and tolerance used.

Usage
-----
  python verify_training_data.py [--rows N] [--seed S]

Defaults: 5 rows, seed=42.  Pick one row per city to ensure variety.
"""

import argparse
import csv
import datetime
import math
import os
import pathlib
import random
import sys
import time

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap paths + credentials
# ---------------------------------------------------------------------------
REPO = pathlib.Path(__file__).resolve().parent
load_dotenv(REPO / ".env", override=True)
for _k in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "OPENAQ_API_KEY"):
    _v = os.getenv(_k, "")
    if _v:
        os.environ[_k] = _v

sys.path.insert(0, str(REPO / "src"))
from aod_utils import (
    latlon_to_tile_pixel, tile_id,
    earthaccess_login, find_granule_for_date,
    download_granule, extract_aod_at_point,
)

TRAINING_CSV  = REPO / "data" / "processed" / "training_data.csv"
OPENAQ_BASE   = "https://api.openaq.org/v3"
OPENAQ_KEY    = os.getenv("OPENAQ_API_KEY", "")
OPENAQ_HEADERS = {"X-API-Key": OPENAQ_KEY} if OPENAQ_KEY else {}
OM_ARCHIVE    = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_VARS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "precipitation_sum",
    "surface_pressure_mean",
    "shortwave_radiation_sum",
]

# Tolerances for match/mismatch decision
PM25_TOL_ABS    = 2.0    # ug/m3  — rounding in stored avg
WEATHER_REL_TOL = 0.02   # 2 %    — floating-point / unit rounding
AOD_TOL_ABS     = 0.005  # AOD units — multi-orbit average rounding

# ---------------------------------------------------------------------------
# Sample selection: one random row per city (or N rows total if N < #cities)
# ---------------------------------------------------------------------------

def load_rows(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Return ≤n rows: stratified by city first, then random fill."""
    rng = random.Random(seed)
    by_city: dict[str, list[dict]] = {}
    for r in rows:
        by_city.setdefault(r["city"], []).append(r)

    sample: list[dict] = []
    cities = sorted(by_city.keys())
    # One per city first
    for city in cities:
        if len(sample) >= n:
            break
        sample.append(rng.choice(by_city[city]))

    # Fill remainder from global pool (excluding already chosen)
    chosen_keys = {(r["date"], r["location_id"]) for r in sample}
    pool = [r for r in rows if (r["date"], r["location_id"]) not in chosen_keys]
    remaining = n - len(sample)
    if remaining > 0:
        sample.extend(rng.sample(pool, min(remaining, len(pool))))

    return sample

# ---------------------------------------------------------------------------
# Re-fetch: PM2.5 from OpenAQ
# ---------------------------------------------------------------------------

def _openaq_get(url: str, params: dict) -> dict | None:
    try:
        r = requests.get(url, params=params, headers=OPENAQ_HEADERS, timeout=20)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def refetch_pm25(location_id: str, date_str: str) -> float | None:
    """
    Re-fetch the daily PM2.5 average for this location_id on date_str.
    Queries /locations/{id} to get the PM2.5 sensor IDs, then polls each
    sensor's /days endpoint with a ±7-day window (API quirk) and filters locally.
    Returns the mean across all sensors that report data on that day, or None.
    """
    # Get PM2.5 sensor IDs for this location
    data = _openaq_get(f"{OPENAQ_BASE}/locations/{location_id}", {})
    if not data:
        return None
    locs = data.get("results") or []
    if not locs:
        return None

    sensor_ids = [
        s["id"]
        for s in (locs[0].get("sensors") or [])
        if (s.get("parameter") or {}).get("name") == "pm25"
    ]
    if not sensor_ids:
        return None

    target  = datetime.date.fromisoformat(date_str)
    d_from  = (target - datetime.timedelta(days=7)).isoformat()
    d_to    = (target + datetime.timedelta(days=1)).isoformat()

    vals: list[float] = []
    for sid in sensor_ids:
        time.sleep(0.3)
        d = _openaq_get(f"{OPENAQ_BASE}/sensors/{sid}/days", {
            "date_from": d_from, "date_to": d_to, "limit": 1000,
        })
        if not d:
            continue
        for rec in (d.get("results") or []):
            period   = rec.get("period") or {}
            dt_local = (period.get("datetimeFrom") or {}).get("local", "")
            if dt_local[:10] != date_str:
                continue
            val = rec.get("value")
            if val is None:
                val = (rec.get("summary") or {}).get("avg")
            if val is not None:
                vals.append(float(val))

    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)

# ---------------------------------------------------------------------------
# Re-fetch: weather from Open-Meteo
# ---------------------------------------------------------------------------

def refetch_weather(lat: float, lon: float, date_str: str) -> dict | None:
    try:
        r = requests.get(OM_ARCHIVE, params={
            "latitude": lat, "longitude": lon,
            "start_date": date_str, "end_date": date_str,
            "daily": ",".join(WEATHER_VARS), "timezone": "auto",
        }, timeout=15)
        r.raise_for_status()
        daily = r.json().get("daily", {})
        return {v: (daily.get(v, [None])[0]) for v in WEATHER_VARS}
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Re-fetch: AOD from MCD19A2 granule
# ---------------------------------------------------------------------------

HDF_TMP = pathlib.Path(os.environ.get("TEMP", "/tmp")) / "skywatch_verify_cache"

def refetch_aod(lat: float, lon: float, date_str: str) -> tuple[float | None, str | None]:
    """
    Find the MCD19A2 granule for exactly date_str (0-day lookback window),
    download, extract, delete.  Returns (aod, granule_date) or (None, None).
    We use a 1-day window so we only accept the granule for that exact date.
    """
    HDF_TMP.mkdir(parents=True, exist_ok=True)
    tp  = latlon_to_tile_pixel(lat, lon)
    tid = tile_id(tp.h, tp.v)

    try:
        earthaccess_login()
    except Exception as e:
        return None, f"login failed: {e}"

    granule, g_date = find_granule_for_date(tid, date_str, lookback_days=1)
    if granule is None:
        return None, None

    hdf_path = download_granule(granule, HDF_TMP)
    if hdf_path is None:
        return None, g_date

    try:
        aod = extract_aod_at_point(hdf_path, lat, lon)
    finally:
        try:
            hdf_path.unlink()
        except OSError:
            pass

    return aod, g_date

# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

PASS  = "PASS"
FAIL  = "FAIL"
SKIP  = "SKIP"   # could not re-fetch

def _pct(a: float, b: float) -> float:
    """Relative difference |a-b| / max(|b|, 1e-9)."""
    return abs(a - b) / max(abs(b), 1e-9)

def compare_pm25(stored: float, refetched: float | None) -> tuple[str, str]:
    if refetched is None:
        return SKIP, "no data returned"
    diff = abs(stored - refetched)
    status = PASS if diff <= PM25_TOL_ABS else FAIL
    return status, f"stored={stored:.2f}  re-fetched={refetched:.2f}  |diff|={diff:.2f}"

def compare_weather(stored: dict, refetched: dict | None) -> list[tuple[str, str, str]]:
    results = []
    if refetched is None:
        for v in WEATHER_VARS:
            results.append((v, SKIP, "fetch failed"))
        return results
    for v in WEATHER_VARS:
        sv = stored.get(v)
        rv = refetched.get(v)
        if sv is None or rv is None:
            results.append((v, SKIP, f"stored={sv}  re-fetched={rv}"))
            continue
        sv, rv = float(sv), float(rv)
        rel = _pct(sv, rv)
        status = PASS if rel <= WEATHER_REL_TOL else FAIL
        results.append((v, status, f"stored={sv:.3f}  re-fetched={rv:.3f}  rel={rel*100:.2f}%"))
    return results

def compare_aod(stored: float, refetched: float | None, g_date: str | None) -> tuple[str, str]:
    if refetched is None:
        note = f"no granule on exact date (g_date={g_date})"
        return SKIP, note
    diff = abs(stored - refetched)
    status = PASS if diff <= AOD_TOL_ABS else FAIL
    return status, f"stored={stored:.4f}  re-extracted={refetched:.4f}  |diff|={diff:.4f}"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify training_data.csv spot-check")
    parser.add_argument("--rows", type=int, default=5,  help="Number of rows to verify")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 70)
    print("skywatch-aq  |  training_data.csv verification")
    print(f"Rows: {args.rows}  |  Seed: {args.seed}")
    print("=" * 70)

    rows   = load_rows(TRAINING_CSV)
    sample = pick_sample(rows, args.rows, args.seed)
    print(f"\nSelected {len(sample)} rows (stratified by city):")
    for r in sample:
        print(f"  [{r['city']:<8}] {r['date']}  loc={r['location_id']}  "
              f"({r['location_name'][:40]})")

    # Track summary
    totals: dict[str, int] = {PASS: 0, FAIL: 0, SKIP: 0}

    for idx, row in enumerate(sample, 1):
        city  = row["city"]
        date  = row["date"]
        lat   = float(row["latitude"])
        lon   = float(row["longitude"])
        loc_id = row["location_id"]

        print(f"\n{'-'*70}")
        print(f"Row {idx}/{len(sample)}: [{city}]  {date}  loc={loc_id}  "
              f"({row['location_name'][:45]})")
        print(f"{'-'*70}")

        # ---- PM2.5 ----
        print(f"\n  [PM2.5]  Stored: {float(row['pm25_daily_avg']):.2f} ug/m3")
        print(f"           Re-fetching from OpenAQ v3...")
        rf_pm25 = refetch_pm25(loc_id, date)
        pm25_status, pm25_note = compare_pm25(float(row["pm25_daily_avg"]), rf_pm25)
        print(f"           {pm25_status:4s} | {pm25_note}")
        totals[pm25_status] += 1

        # ---- Weather ----
        print(f"\n  [Weather] Re-fetching from Open-Meteo (lat={lat}, lon={lon}, {date})...")
        rf_wx = refetch_weather(lat, lon, date)
        wx_comparisons = compare_weather({v: row[v] for v in WEATHER_VARS}, rf_wx)
        short_labels = {
            "temperature_2m_mean":        "temp_mean",
            "relative_humidity_2m_mean":  "rh_mean",
            "wind_speed_10m_mean":        "wind_mean",
            "precipitation_sum":          "precip_sum",
            "surface_pressure_mean":      "pressure_mean",
            "shortwave_radiation_sum":    "sw_rad_sum",
        }
        for var, wx_status, wx_note in wx_comparisons:
            lbl = short_labels.get(var, var)
            print(f"           {wx_status:4s} | {lbl:<16} | {wx_note}")
            totals[wx_status] += 1

        # ---- AOD ----
        stored_aod = float(row["aod_055"])
        print(f"\n  [AOD]    Stored: {stored_aod:.4f}")
        print(f"           Re-extracting from MCD19A2 granule for {date}...")
        rf_aod, g_date = refetch_aod(lat, lon, date)
        aod_status, aod_note = compare_aod(stored_aod, rf_aod, g_date)
        print(f"           {aod_status:4s} | {aod_note}")
        totals[aod_status] += 1

    # ---- Summary ----
    total = sum(totals.values())
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  PASS: {totals[PASS]:3d} / {total}")
    print(f"  FAIL: {totals[FAIL]:3d} / {total}")
    print(f"  SKIP: {totals[SKIP]:3d} / {total}  (source returned no data)")
    if totals[FAIL] == 0:
        print("\n  All verifiable checks PASSED.")
    else:
        print(f"\n  {totals[FAIL]} check(s) FAILED — review output above.")
    print()


if __name__ == "__main__":
    main()
