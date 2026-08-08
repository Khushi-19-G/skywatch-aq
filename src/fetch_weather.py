"""
fetch_weather.py
----------------
Fetches daily weather for Delhi and Karachi city centres from the
Open-Meteo Historical Weather API (free, no API key).

Variables requested:
  temperature_2m_mean        (deg C)
  relative_humidity_2m_mean  (%)
  wind_speed_10m_mean        (km/h)
  precipitation_sum          (mm)
  surface_pressure_mean      (hPa)
  boundary_layer_height      -- not available in Open-Meteo historical archive;
                                replaced by shortwave_radiation_sum (MJ/m2)
                                as a sunlight/cloud proxy useful for AOD work.

API docs: https://open-meteo.com/en/docs/historical-weather-api
"""

import csv
import pathlib
import time
import requests

DATE_FROM = "2023-01-01"
DATE_TO   = "2025-07-31"

OUT_CSV = "data/processed/weather.csv"

CITIES = {
    "Delhi":   {"latitude": 28.6139, "longitude": 77.2090, "timezone": "Asia/Kolkata"},
    "Karachi": {"latitude": 24.8607, "longitude": 67.0011, "timezone": "Asia/Karachi"},
    "Mumbai":  {"latitude": 19.0760, "longitude": 72.8777, "timezone": "Asia/Kolkata"},
}

DAILY_VARS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "precipitation_sum",
    "surface_pressure_mean",
    "shortwave_radiation_sum",   # sunlight proxy; BLH not in historical API
]

API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_city(city: str, meta: dict) -> list[dict]:
    """Fetch all daily records for one city. Returns list of row dicts."""
    params = {
        "latitude":        meta["latitude"],
        "longitude":       meta["longitude"],
        "start_date":      DATE_FROM,
        "end_date":        DATE_TO,
        "daily":           ",".join(DAILY_VARS),
        "timezone":        meta["timezone"],
    }
    r = requests.get(API_URL, params=params, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Open-Meteo {city}: HTTP {r.status_code} — {r.text[:200]}")
    data = r.json()
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    if not dates:
        raise RuntimeError(f"Open-Meteo {city}: no dates in response")

    rows = []
    for i, date in enumerate(dates):
        row = {"date": date, "city": city}
        for var in DAILY_VARS:
            raw = daily.get(var, [])
            row[var] = raw[i] if i < len(raw) else None
        rows.append(row)
    return rows


def main() -> None:
    print("Fetching Open-Meteo historical weather...")
    print(f"  Period  : {DATE_FROM} to {DATE_TO}")
    print(f"  Cities  : {list(CITIES)}")
    print(f"  Variables: {DAILY_VARS}")

    all_rows: list[dict] = []
    for city, meta in CITIES.items():
        print(f"\n  Fetching {city}...")
        rows = fetch_city(city, meta)
        all_rows.extend(rows)
        print(f"    {len(rows):,} days fetched")
        time.sleep(0.5)

    # Save
    pathlib.Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    fields = ["date", "city"] + DAILY_VARS
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {len(all_rows):,} rows -> {OUT_CSV}")

    # Quick check for nulls
    for city in CITIES:
        city_rows = [r for r in all_rows if r["city"] == city]
        null_counts = {v: sum(1 for r in city_rows if r[v] is None) for v in DAILY_VARS}
        if any(null_counts.values()):
            print(f"  [{city}] null counts: { {k:v for k,v in null_counts.items() if v} }")
        else:
            print(f"  [{city}] all {len(city_rows)} rows complete, no nulls")


if __name__ == "__main__":
    main()
