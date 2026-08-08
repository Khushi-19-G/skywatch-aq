"""
fetch_satellite_data.py
-----------------------
Fetches NASA MODIS MAIAC daily AOD (MCD19A2 v061) for the ground-monitor
locations collected in fetch_ground_data.py.

Approach
--------
AppEEARS does NOT carry MCD19A2. Instead we use earthaccess to:
  1. Identify the unique MODIS sinusoidal tiles that cover Delhi and Karachi.
  2. Search + download MCD19A2 HDF granules for each tile-day in the
     requested date range.
  3. For each granule, map every monitor coordinate to a pixel (row, col),
     read Optical_Depth_055 for both orbits (Terra + Aqua), apply QA,
     apply the scale factor (0.001), and record valid readings.
  4. Average Terra + Aqua per location per day -> one AOD value.
  5. Write data/raw/satellite_aod_raw.csv (all per-orbit readings) and
     data/processed/satellite_aod.csv (daily averages, one row per
     location-day that has at least one valid overpass).

MCD19A2 HDF structure (confirmed by probe)
------------------------------------------
  Variables: Optical_Depth_055  shape (2, 1200, 1200)  int16 scaled
             AOD_QA             shape (2, 1200, 1200)  uint16 bitmask
  Dim-0 (size 2) = Terra overpass [0], Aqua overpass [1]
  scale_factor = 0.001, add_offset = 0.0
  fill_value (raw int16) = -28672 (masked automatically by netCDF4)
  valid_range raw: [-100, 6000]  ->  physical AOD: [-0.1, 6.0]

AOD_QA bitmask (bits 0-2 = Cloud Mask):
  001 = Clear, 011 = Possibly cloudy, 101 = Cloudy/snow
  We keep only pixels where cloud mask bits 0-2 != 0 (not undefined)
  and bits 0-2 in {0b001, 0b011} (Clear or Possibly cloudy).
  We discard cloudy (101) and undefined (000).

Sinusoidal grid coordinate transform
-------------------------------------
  RE = 6371007.181 m
  Tile size = 10 deg * pi/180 * RE metres
  Pixel size = tile_size_m / 1200
  For tile (h, v):
    x_origin = (h - 18) * tile_size_m
    y_origin = (9  - v) * tile_size_m
  lat/lon -> row/col:
    x = RE * lon_rad * cos(lat_rad)
    y = RE * lat_rad
    col = (x - x_origin) / pixel_size_m
    row = (y_origin - y) / pixel_size_m
"""

import os
import csv
import math
import time
import pathlib
import datetime
from typing import NamedTuple

import numpy as np
import netCDF4 as nc4
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv(override=True)
# earthaccess reads OS env vars directly; push dotenv values in.
for _k in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
    _v = os.getenv(_k, "")
    if _v:
        os.environ[_k] = _v

import earthaccess  # noqa: E402 — import after env vars are set

DATE_FROM = "2023-01-01"
DATE_TO   = "2025-07-31"

GROUND_CSVS = {
    "Delhi":   "data/raw/ground_pm25_delhi.csv",
    "Karachi": "data/raw/ground_pm25_karachi.csv",
}

RAW_OUT      = "data/raw/satellite_aod_raw.csv"
PROCESSED_OUT = "data/processed/satellite_aod.csv"

HDF_CACHE_DIR = pathlib.Path("data/raw/_hdf_cache")

# MODIS sinusoidal constants
RE = 6371007.181   # authalic sphere radius (m)
TILE_DEG = 10.0    # degrees per tile
TILE_M   = TILE_DEG * math.pi / 180.0 * RE
PIX_PER_TILE = 1200
PIX_M = TILE_M / PIX_PER_TILE   # ~926.6 m

# AOD_QA cloud-mask bits 0-2: keep 001 (clear) and 011 (possibly cloudy)
KEEP_CLOUD_STATES = {0b001, 0b011}

# Delay between earthaccess downloads to avoid hammering the server
DOWNLOAD_DELAY_S = 0.5


# ---------------------------------------------------------------------------
# MODIS sinusoidal helpers
# ---------------------------------------------------------------------------

class TilePixel(NamedTuple):
    h: int
    v: int
    row: int
    col: int


def latlon_to_tile_pixel(lat_deg: float, lon_deg: float) -> TilePixel:
    """Convert WGS-84 lat/lon to MODIS sinusoidal tile + pixel row/col."""
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
# I/O helpers
# ---------------------------------------------------------------------------

def load_monitor_locations(city: str, csv_path: str) -> list[dict]:
    """Return unique monitor locations from a ground PM2.5 CSV."""
    seen: set[str] = set()
    locs: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lid = row["location_id"]
            if lid in seen:
                continue
            seen.add(lid)
            locs.append({
                "city":        city,
                "location_id": lid,
                "latitude":    float(row["latitude"]),
                "longitude":   float(row["longitude"]),
                "location_name": row["location_name"],
            })
    return locs


def daterange(start: str, end: str):
    """Yield date strings from start to end inclusive."""
    d = datetime.date.fromisoformat(start)
    e = datetime.date.fromisoformat(end)
    while d <= e:
        yield d.isoformat()
        d += datetime.timedelta(days=1)


# ---------------------------------------------------------------------------
# Earthdata authentication + granule search/download
# ---------------------------------------------------------------------------

def _earthaccess_login() -> None:
    earthaccess.login(strategy="environment")


def search_granules_for_tile_month(tile: str, year: int, month: int) -> dict[str, object]:
    """
    Search for all MCD19A2.061 granules for a tile within one calendar month.
    Returns a dict mapping date_str -> granule object.
    """
    import calendar
    h = int(tile[1:3])
    v = int(tile[4:6])
    lat_min = 90 - (v + 1) * 10
    lat_max = 90 - v * 10
    lon_min = (h - 18) * 10
    lon_max = lon_min + 10
    last_day = calendar.monthrange(year, month)[1]
    start = f"{year:04d}-{month:02d}-01"
    end   = f"{year:04d}-{month:02d}-{last_day:02d}"
    results = earthaccess.search_data(
        short_name="MCD19A2",
        version="061",
        temporal=(start, end),
        bounding_box=(lon_min, lat_min, lon_max, lat_max),
        count=40,          # up to 31 days x 1 tile = <=31; 40 is safe
    )
    # Index by date string parsed from filename (A2023001 -> 2023-01-01)
    by_date: dict[str, object] = {}
    for g in results:
        urls = g.data_links() or []
        for url in urls:
            fname = url.split("/")[-1]
            if tile not in fname:
                continue
            parts = fname.split(".")
            if len(parts) < 3:
                continue
            jd_str = parts[1]          # e.g. "A2023001"
            try:
                yr  = int(jd_str[1:5])
                doy = int(jd_str[5:8])
                d   = (datetime.date(yr, 1, 1) + datetime.timedelta(days=doy - 1)).isoformat()
                by_date[d] = g
            except (ValueError, IndexError):
                pass
            break
    return by_date


def download_granule(granule, cache_dir: pathlib.Path, retries: int = 3) -> pathlib.Path | None:
    """Download a granule to cache_dir (skip if already present). Returns path."""
    urls = granule.data_links()
    if not urls:
        return None
    filename = urls[0].split("/")[-1]
    dest = cache_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    for attempt in range(retries):
        try:
            files = earthaccess.download([granule], local_path=str(cache_dir))
            if files:
                p = pathlib.Path(files[0])
                if p.exists() and p.stat().st_size > 0:
                    return p
        except Exception as exc:
            wait = 15 * (attempt + 1)
            print(f"    [download error attempt {attempt+1}/{retries}] {type(exc).__name__}: {str(exc)[:120]}")
            if attempt < retries - 1:
                print(f"    retrying in {wait}s...")
                time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# HDF pixel extraction
# ---------------------------------------------------------------------------

def extract_aod_pixels(
    hdf_path: pathlib.Path,
    locations: list[dict],
) -> list[dict]:
    """
    Open one MCD19A2 HDF granule and extract Optical_Depth_055 + QA at
    each monitor coordinate.  Returns list of raw-orbit row dicts.
    """
    rows: list[dict] = []

    # Parse date + tile from filename: MCD19A2.AYYYYDOY.hHHvVV.061.*.hdf
    parts = hdf_path.name.split(".")
    # parts[1] = A2023001 -> julian day
    try:
        jd_str = parts[1]   # e.g. "A2023001"
        year   = int(jd_str[1:5])
        doy    = int(jd_str[5:8])
        date   = (datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)).isoformat()
        tile   = parts[2]   # e.g. "h24v06"
        h_tile = int(tile[1:3])
        v_tile = int(tile[4:6])
    except (IndexError, ValueError) as exc:
        print(f"    [filename parse error] {hdf_path.name}: {exc}")
        return rows

    try:
        ds = nc4.Dataset(str(hdf_path))
    except Exception as exc:
        print(f"    [HDF open error] {hdf_path.name}: {exc}")
        return rows

    try:
        v055 = ds.variables["Optical_Depth_055"]   # (2, 1200, 1200) float64 auto-scaled
        vqa  = ds.variables["AOD_QA"]               # (2, 1200, 1200) uint16 raw

        # Read full arrays once (avoids repeated small reads)
        aod_arr = v055[:]       # masked float64 array
        qa_arr  = vqa[:].data   # uint16, no masking

        for loc in locations:
            tp = latlon_to_tile_pixel(loc["latitude"], loc["longitude"])
            if tp.h != h_tile or tp.v != v_tile:
                continue   # this granule doesn't cover this location
            if not (0 <= tp.row < PIX_PER_TILE and 0 <= tp.col < PIX_PER_TILE):
                continue   # out of bounds

            for orbit_idx in range(2):
                orbit_name = "Terra" if orbit_idx == 0 else "Aqua"

                aod_val = aod_arr[orbit_idx, tp.row, tp.col]
                if np.ma.is_masked(aod_val):
                    continue   # fill / no data

                aod_float = float(aod_val)
                if aod_float < -0.05:   # negative AOD beyond small noise -> invalid
                    continue

                # QA check: cloud mask in bits 0-2
                qa_val = int(qa_arr[orbit_idx, tp.row, tp.col])
                if qa_val == 0:
                    continue   # QA fill (no retrieval)
                cloud_bits = qa_val & 0b111
                if cloud_bits not in KEEP_CLOUD_STATES:
                    continue   # cloudy or undefined

                rows.append({
                    "date":          date,
                    "location_id":   loc["location_id"],
                    "location_name": loc["location_name"],
                    "city":          loc["city"],
                    "latitude":      loc["latitude"],
                    "longitude":     loc["longitude"],
                    "orbit":         orbit_name,
                    "aod_055_raw":   round(aod_float, 4),
                    "qa_value":      qa_val,
                })

    finally:
        ds.close()

    return rows


# ---------------------------------------------------------------------------
# Post-processing: average Terra+Aqua -> one daily value per location
# ---------------------------------------------------------------------------

def aggregate_daily(raw_rows: list[dict]) -> list[dict]:
    """Average Terra+Aqua overpasses per (date, location_id) -> daily rows."""
    groups: dict[tuple, list[float]] = {}
    meta: dict[tuple, dict] = {}

    for r in raw_rows:
        key = (r["date"], r["location_id"])
        groups.setdefault(key, []).append(r["aod_055_raw"])
        if key not in meta:
            meta[key] = {
                "date":          r["date"],
                "location_id":   r["location_id"],
                "location_name": r["location_name"],
                "city":          r["city"],
                "latitude":      r["latitude"],
                "longitude":     r["longitude"],
            }

    daily: list[dict] = []
    for key, vals in groups.items():
        row = dict(meta[key])
        row["aod_055"] = round(sum(vals) / len(vals), 4)
        daily.append(row)

    daily.sort(key=lambda r: (r["city"], r["location_id"], r["date"]))
    return daily


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_raw_csv(rows: list[dict], path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"  (no raw rows to write -> {path})")
        return
    fields = ["date", "location_id", "location_name", "city",
              "latitude", "longitude", "orbit", "aod_055_raw", "qa_value"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {len(rows):,} raw-orbit rows -> {path}")


def write_daily_csv(rows: list[dict], path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"  (no daily rows to write -> {path})")
        return
    fields = ["date", "location_id", "location_name", "city",
              "latitude", "longitude", "aod_055"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {len(rows):,} daily rows -> {path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(daily_rows: list[dict], locations: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    total_location_days = 0
    for city in ["Delhi", "Karachi"]:
        city_locs = [l for l in locations if l["city"] == city]
        city_rows = [r for r in daily_rows if r["city"] == city]
        if not city_rows or not city_locs:
            print(f"\n[{city}] No data.")
            continue

        dates = [r["date"] for r in city_rows]
        aods  = [r["aod_055"] for r in city_rows]
        mean_aod = sum(aods) / len(aods)

        # Count total possible location-days
        all_dates = list(daterange(DATE_FROM, DATE_TO))
        n_locs     = len(city_locs)
        possible   = n_locs * len(all_dates)
        pct_valid  = 100.0 * len(city_rows) / possible if possible else 0.0
        total_location_days += len(city_rows)

        print(f"\n[{city}]")
        print(f"  Locations with >=1 valid reading : {len({r['location_id'] for r in city_rows})}/{n_locs}")
        print(f"  Total daily AOD rows             : {len(city_rows):,}")
        print(f"  Date range                       : {min(dates)} to {max(dates)}")
        print(f"  Location-days with valid AOD     : {len(city_rows):,} / {possible:,}  ({pct_valid:.1f}%)")
        print(f"  Mean AOD_055                     : {mean_aod:.3f}")

    print(f"\n  Total daily AOD rows (both cities): {total_location_days:,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("skywatch-aq  |  MCD19A2 MAIAC AOD fetcher")
    print(f"Period: {DATE_FROM}  to  {DATE_TO}")
    print("=" * 60)

    # Auth
    print("\nAuthenticating with NASA Earthdata...")
    _earthaccess_login()
    print("  Authenticated OK.")

    # Load all monitor locations
    all_locations: list[dict] = []
    for city, csv_path in GROUND_CSVS.items():
        locs = load_monitor_locations(city, csv_path)
        print(f"  {city}: {len(locs)} unique monitor locations loaded.")
        all_locations.extend(locs)

    # Determine required tiles
    tile_to_locs: dict[str, list[dict]] = {}
    for loc in all_locations:
        tp = latlon_to_tile_pixel(loc["latitude"], loc["longitude"])
        tid = tile_id(tp.h, tp.v)
        tile_to_locs.setdefault(tid, []).append(loc)

    print(f"\nModules to fetch: {sorted(tile_to_locs.keys())}")
    for tid, locs in tile_to_locs.items():
        print(f"  Tile {tid}: {len(locs)} location(s)")

    HDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Build list of (year, month) tuples to process
    start_d = datetime.date.fromisoformat(DATE_FROM)
    end_d   = datetime.date.fromisoformat(DATE_TO)
    months: list[tuple[int, int]] = []
    d = start_d.replace(day=1)
    while d <= end_d:
        months.append((d.year, d.month))
        # advance one month
        if d.month == 12:
            d = d.replace(year=d.year + 1, month=1)
        else:
            d = d.replace(month=d.month + 1)

    all_dates_set = set(daterange(DATE_FROM, DATE_TO))
    all_dates     = list(daterange(DATE_FROM, DATE_TO))

    # --- Resume support: load any rows already saved in RAW_OUT ---
    raw_rows: list[dict] = []
    already_done: set[tuple] = set()   # (date, location_id) pairs already extracted
    raw_out_path = pathlib.Path(RAW_OUT)
    if raw_out_path.exists() and raw_out_path.stat().st_size > 0:
        with open(raw_out_path, newline="", encoding="utf-8") as rf:
            for row in csv.DictReader(rf):
                raw_rows.append(row)
                already_done.add((row["date"], row["location_id"]))
        print(f"\nResume: loaded {len(raw_rows):,} existing raw rows "
              f"({len({r['date'] for r in raw_rows})} dates already done).")
    else:
        print(f"\nStarting fresh (no existing {RAW_OUT}).")

    # Determine which dates still need processing
    dates_done_set: set[str] = {r["date"] for r in raw_rows}

    print(f"\nProcessing {len(months)} months x {len(tile_to_locs)} tile(s)  "
          f"({len(all_dates)} days total, {len(dates_done_set)} already done)...")

    new_rows: list[dict] = []

    for month_num, (year, month) in enumerate(months, 1):
        mo_str = f"{year:04d}-{month:02d}"
        # Check whether any day in this month still needs work
        import calendar as _cal
        last_day = _cal.monthrange(year, month)[1]
        mo_dates = {f"{year:04d}-{month:02d}-{d:02d}" for d in range(1, last_day + 1)}
        pending  = mo_dates & all_dates_set - dates_done_set
        if not pending:
            continue   # entire month already processed

        print(f"  Month {month_num}/{len(months)} : {mo_str}  "
              f"({len(pending)} days pending, {len(new_rows):,} new rows so far)")

        for tid, locs in tile_to_locs.items():
            # One CMR search for the whole month
            granule_map = search_granules_for_tile_month(tid, year, month)
            if not granule_map:
                print(f"    Tile {tid}: no granules found for {mo_str}")
                continue

            print(f"    Tile {tid}: {len(granule_map)} granules found")

            for date_str, granule in sorted(granule_map.items()):
                if date_str not in pending:
                    continue   # already done or out of range
                hdf_path = download_granule(granule, HDF_CACHE_DIR)
                if hdf_path is None:
                    continue
                orbit_rows = extract_aod_pixels(hdf_path, locs)
                new_rows.extend(orbit_rows)
                dates_done_set.add(date_str)

            time.sleep(DOWNLOAD_DELAY_S)

        # Append new rows to raw CSV incrementally after each month
        if new_rows:
            raw_out_path.parent.mkdir(parents=True, exist_ok=True)
            write_mode = "a" if raw_out_path.exists() and raw_out_path.stat().st_size > 0 else "w"
            fields = ["date", "location_id", "location_name", "city",
                      "latitude", "longitude", "orbit", "aod_055_raw", "qa_value"]
            with open(raw_out_path, write_mode, newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if write_mode == "w":
                    w.writeheader()
                w.writerows(new_rows)
            raw_rows.extend(new_rows)
            new_rows = []

    all_raw = raw_rows   # already extended with new_rows above
    print(f"\nExtraction complete. {len(all_raw):,} total raw orbit readings.")

    # Rewrite raw CSV cleanly (dedup + sort)
    write_raw_csv(all_raw, RAW_OUT)

    # Aggregate to daily
    daily_rows = aggregate_daily(all_raw)
    write_daily_csv(daily_rows, PROCESSED_OUT)

    # Summary
    print_summary(daily_rows, all_locations)
    print("\nDone.")


if __name__ == "__main__":
    main()
