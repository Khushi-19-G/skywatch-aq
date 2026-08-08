"""
fetch_ground_data.py
--------------------
Fetches daily PM2.5 measurements from the OpenAQ v3 API for monitoring
stations near Delhi and Karachi, then saves them as CSVs in data/raw/.

API reference: https://api.openaq.org/v3
  - GET /locations          → find stations by coordinate + radius
  - GET /sensors/{id}/days  → daily aggregates for a sensor
"""

import os
import time
import csv
import pathlib
from datetime import date

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()
_API_KEY = os.getenv("OPENAQ_API_KEY", "")

BASE_URL = "https://api.openaq.org/v3"
PM25_PARAMETER_ID = 2          # OpenAQ parameter_id for pm25
MAX_RADIUS_M = 25_000          # API hard cap is 25 000 m

DATE_FROM = "2023-01-01"
DATE_TO   = "2025-07-31"

PAGE_LIMIT = 1000              # rows per page for /days
DELAY_BETWEEN_REQUESTS = 1.1   # seconds — stay well under rate limit
RETRY_WAIT = 30                # seconds to wait on a 429

CITIES = [
    {
        "name":     "Delhi",
        "coords":   "28.6139,77.2090",
        "out_csv":  "data/raw/ground_pm25_delhi.csv",
    },
    {
        "name":     "Karachi",
        "coords":   "24.8607,67.0011",
        "out_csv":  "data/raw/ground_pm25_karachi.csv",
    },
    {
        "name":     "Mumbai",
        "coords":   "19.0760,72.8777",
        "out_csv":  "data/raw/ground_pm25_mumbai.csv",
    },
]

HEADERS = {"X-API-Key": _API_KEY} if _API_KEY else {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict, retries: int = 3) -> dict | None:
    """GET with retry-on-429 and basic error handling. Never logs the key."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        except requests.RequestException as exc:
            print(f"    [network error] {exc}  (attempt {attempt + 1}/{retries})")
            time.sleep(5)
            continue

        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            wait = RETRY_WAIT * (attempt + 1)
            print(f"    [429 rate-limit] waiting {wait}s before retry...")
            time.sleep(wait)
            continue
        # Any other error - log status + body snippet, return None
        print(f"    [HTTP {r.status_code}] {r.text[:120]}")
        return None

    print(f"    [failed after {retries} attempts] {url}")
    return None


def _found_count(meta: dict) -> int:
    """meta['found'] can be an int or the string '>1000' — return a usable int."""
    raw = meta.get("found", 0)
    if isinstance(raw, str):
        return int(raw.lstrip(">")) + 1   # treat '>N' as N+1 → triggers pagination
    return int(raw)


def fetch_pm25_locations(city_name: str, coords: str) -> list[dict]:
    """
    Query /locations for PM2.5 stations within MAX_RADIUS_M of coords.
    Returns a list of dicts with keys: id, name, latitude, longitude, pm25_sensor_ids.
    """
    print(f"\n[{city_name}] Searching PM2.5 locations within {MAX_RADIUS_M/1000:.0f} km...")
    data = _get(
        f"{BASE_URL}/locations",
        {
            "coordinates":   coords,
            "radius":        MAX_RADIUS_M,
            "parameters_id": PM25_PARAMETER_ID,
            "limit":         100,
        },
    )
    if not data:
        print(f"  No response from /locations for {city_name}.")
        return []

    results = data.get("results") or []
    meta    = data.get("meta", {})
    found   = meta.get("found", len(results))
    print(f"  API reports {found} PM2.5 location(s) within radius.")

    locations = []
    for loc in results:
        coords_obj = loc.get("coordinates") or {}
        pm25_sensors = [
            s["id"]
            for s in (loc.get("sensors") or [])
            if (s.get("parameter") or {}).get("name") == "pm25"
        ]
        if not pm25_sensors:
            continue
        locations.append(
            {
                "id":              loc["id"],
                "name":            loc.get("name", "unknown"),
                "latitude":        coords_obj.get("latitude"),
                "longitude":       coords_obj.get("longitude"),
                "pm25_sensor_ids": pm25_sensors,
                "datetime_last":   ((loc.get("datetimeLast") or {}).get("utc", "") or "")[:10],
            }
        )

    print(f"  {len(locations)} location(s) have at least one PM2.5 sensor.")
    for loc in locations:
        print(
            f"    id={loc['id']:>8}  last={loc['datetime_last']}  "
            f"sensors={loc['pm25_sensor_ids']}  {loc['name'][:55]}"
        )
    return locations


def fetch_sensor_days(
    sensor_id: int,
    location: dict,
) -> list[dict]:
    """
    Fetch all daily aggregates for sensor_id between DATE_FROM and DATE_TO.
    Handles pagination (page-based). Returns list of row dicts.
    """
    rows: list[dict] = []
    page = 1

    while True:
        time.sleep(DELAY_BETWEEN_REQUESTS)
        data = _get(
            f"{BASE_URL}/sensors/{sensor_id}/days",
            {
                "date_from": DATE_FROM,
                "date_to":   DATE_TO,
                "limit":     PAGE_LIMIT,
                "page":      page,
            },
        )
        if not data:
            break

        results = data.get("results") or []
        meta    = data.get("meta") or {}

        for rec in results:
            period    = rec.get("period") or {}
            dt_from   = (period.get("datetimeFrom") or {}).get("utc", "")
            # Use local date (datetimeFrom local) as the calendar day label
            dt_local  = (period.get("datetimeFrom") or {}).get("local", dt_from)
            day_label = dt_local[:10] if dt_local else dt_from[:10]

            summary  = rec.get("summary") or {}
            avg_val  = rec.get("value")            # top-level daily avg
            if avg_val is None:
                avg_val = summary.get("avg")

            coverage = rec.get("coverage") or {}
            meas_cnt = coverage.get("observedCount", 0)

            if avg_val is None:
                continue

            rows.append(
                {
                    "date":            day_label,
                    "location_id":     location["id"],
                    "location_name":   location["name"],
                    "latitude":        location["latitude"],
                    "longitude":       location["longitude"],
                    "pm25_daily_avg":  round(float(avg_val), 4),
                    "measurement_count": int(meas_cnt),
                }
            )

        # Pagination: stop if fewer rows than limit returned
        if len(results) < PAGE_LIMIT:
            break

        # Double-check against found
        found = _found_count(meta)
        if len(rows) >= found:
            break

        page += 1

    return rows


def save_csv(rows: list[dict], path: str) -> None:
    """Write rows to CSV, creating parent directories if needed."""
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"  (no rows to write → {path})")
        return
    fields = ["date", "location_id", "location_name",
               "latitude", "longitude", "pm25_daily_avg", "measurement_count"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows):,} rows → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_city(city: dict) -> list[dict]:
    """End-to-end fetch for one city. Returns all collected rows."""
    name   = city["name"]
    coords = city["coords"]
    out    = city["out_csv"]

    locations = fetch_pm25_locations(name, coords)
    if not locations:
        print(f"  [{name}] No PM2.5 locations found — skipping.")
        return []

    all_rows: list[dict] = []

    for loc in locations:
        loc_rows: list[dict] = []

        for sid in loc["pm25_sensor_ids"]:
            print(
                f"\n  [{name}] Fetching sensor {sid} for "
                f"'{loc['name'][:45]}' ({DATE_FROM} to {DATE_TO})..."
            )
            sensor_rows = fetch_sensor_days(sid, loc)
            if not sensor_rows:
                print(f"    -> no data for sensor {sid}, skipping.")
            else:
                print(f"    -> {len(sensor_rows):,} day records fetched.")
                loc_rows.extend(sensor_rows)

        if loc_rows:
            # De-duplicate by (date, location_id) - keep first occurrence
            seen: set[tuple] = set()
            for row in loc_rows:
                key = (row["date"], row["location_id"])
                if key not in seen:
                    seen.add(key)
                    all_rows.append(row)
        else:
            print(f"  [{name}] No data returned for location '{loc['name'][:45]}'.")

    # Sort by location then date before saving
    all_rows.sort(key=lambda r: (r["location_id"], r["date"]))
    save_csv(all_rows, out)
    return all_rows


def print_summary(city_name: str, rows: list[dict]) -> None:
    if not rows:
        print(f"\n[{city_name}] No data collected.")
        return
    dates  = [r["date"] for r in rows]
    avgs   = [r["pm25_daily_avg"] for r in rows]
    grand_avg = sum(avgs) / len(avgs)
    print(
        f"\n[{city_name} Summary]\n"
        f"  Total rows   : {len(rows):,}\n"
        f"  Date range   : {min(dates)} to {max(dates)}\n"
        f"  Mean PM2.5   : {grand_avg:.1f} ug/m3"
    )


def main() -> None:
    print("=" * 60)
    print("skywatch-aq  |  OpenAQ v3 ground PM2.5 fetcher")
    print(f"Period: {DATE_FROM}  to  {DATE_TO}")
    print(f"API key loaded: {'yes' if _API_KEY else 'NO - requests may be rate-limited'}")
    print("=" * 60)

    all_city_rows: dict[str, list[dict]] = {}

    for city in CITIES:
        city_rows = process_city(city)
        all_city_rows[city["name"]] = city_rows

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for city in CITIES:
        print_summary(city["name"], all_city_rows[city["name"]])

    print("\nDone.")


if __name__ == "__main__":
    main()
