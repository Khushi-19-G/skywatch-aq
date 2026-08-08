"""
build_dataset.py
----------------
Merges satellite AOD, ground PM2.5, and Open-Meteo weather data,
adds time features, audits the matched dataset, and writes:
  - data/processed/training_data.csv
  - data/processed/aod_vs_pm25.png
"""

import csv
import math
import pathlib
import datetime
import statistics
import collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AOD_CSV       = "data/processed/satellite_aod.csv"
DELHI_CSV     = "data/raw/ground_pm25_delhi.csv"
KARACHI_CSV   = "data/raw/ground_pm25_karachi.csv"
MUMBAI_CSV    = "data/raw/ground_pm25_mumbai.csv"
WEATHER_CSV   = "data/processed/weather.csv"
TRAINING_CSV  = "data/processed/training_data.csv"
SCATTER_PNG   = "data/processed/aod_vs_pm25.png"

WEATHER_VARS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "precipitation_sum",
    "surface_pressure_mean",
    "shortwave_radiation_sum",
]

# ---------------------------------------------------------------------------
# South-Asia season labels
# ---------------------------------------------------------------------------
def season(month: int) -> str:
    if month in (12, 1, 2):   return "winter"
    if month in (3, 4, 5):    return "spring"
    if month in (6, 7, 8, 9): return "monsoon"
    return "autumn"

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_ground(paths: list[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    """(date, location_id) -> ground row (with city field)."""
    index: dict[tuple[str, str], dict] = {}
    for city, path in paths:
        for row in load_csv(path):
            key = (row["date"], row["location_id"])
            index[key] = {**row, "city": city}
    return index


def load_aod() -> dict[tuple[str, str], dict]:
    """(date, location_id) -> aod row."""
    index: dict[tuple[str, str], dict] = {}
    for row in load_csv(AOD_CSV):
        key = (row["date"], row["location_id"])
        index[key] = row
    return index


def load_weather() -> dict[tuple[str, str], dict]:
    """(date, city) -> weather row."""
    index: dict[tuple[str, str], dict] = {}
    for row in load_csv(WEATHER_CSV):
        key = (row["date"], row["city"])
        index[key] = row
    return index

# ---------------------------------------------------------------------------
# Build joined dataset
# ---------------------------------------------------------------------------

def build_joined(
    ground:  dict[tuple, dict],
    aod:     dict[tuple, dict],
    weather: dict[tuple, dict],
) -> list[dict]:
    joined: list[dict] = []
    weather_miss = 0

    for key, grow in ground.items():
        if key not in aod:
            continue
        arow = aod[key]

        city     = grow["city"]
        wrow     = weather.get((key[0], city))
        if wrow is None:
            weather_miss += 1

        date_obj = datetime.date.fromisoformat(key[0])
        month    = date_obj.month

        record: dict = {
            "date":              key[0],
            "location_id":       key[1],
            "location_name":     grow["location_name"],
            "city":              city,
            "latitude":          float(grow["latitude"]),
            "longitude":         float(grow["longitude"]),
            "pm25_daily_avg":    float(grow["pm25_daily_avg"]),
            "measurement_count": int(grow["measurement_count"]),
            "aod_055":           float(arow["aod_055"]),
            "month":             month,
            "day_of_year":       date_obj.timetuple().tm_yday,
            "day_of_week":       date_obj.weekday(),
            "season":            season(month),
        }
        # Attach weather vars (None if missing)
        for var in WEATHER_VARS:
            raw = wrow.get(var) if wrow else None
            record[var] = float(raw) if raw not in (None, "") else None

        joined.append(record)

    joined.sort(key=lambda r: (r["city"], r["location_id"], r["date"]))
    if weather_miss:
        print(f"  Warning: {weather_miss} rows had no weather match")
    return joined

# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def describe(values: list[float], label: str, indent: int = 4) -> None:
    if not values:
        print(f"{' '*indent}{label}: no data")
        return
    pad = " " * indent
    mu  = statistics.mean(values)
    med = statistics.median(values)
    sd  = statistics.stdev(values) if len(values) > 1 else 0.0
    print(f"{pad}{label}: mean={mu:.2f}  median={med:.2f}  "
          f"min={min(values):.2f}  max={max(values):.2f}  sd={sd:.2f}  n={len(values)}")


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2: return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx  = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy  = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0: return float("nan")
    return num / (dx * dy)


def spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2: return float("nan")
    def ranks(v):
        s = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and v[s[j]] == v[s[j + 1]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[s[k]] = avg
            i = j + 1
        return r
    return pearson(ranks(xs), ranks(ys))

# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit(joined: list[dict]) -> None:
    print("\n" + "=" * 68)
    print("DATASET AUDIT")
    print("=" * 68)
    print(f"Total matched rows (all cities): {len(joined):,}")

    by_city = collections.defaultdict(list)
    for r in joined: by_city[r["city"]].append(r)

    for city in sorted(by_city.keys()):
        rows = by_city.get(city, [])
        if not rows:
            print(f"\n[{city}] no matched rows"); continue

        dates  = sorted({r["date"] for r in rows})
        locs   = {r["location_id"] for r in rows}
        pm25s  = [r["pm25_daily_avg"] for r in rows]
        aods   = [r["aod_055"] for r in rows]
        seasons = collections.Counter(r["season"] for r in rows)

        print(f"\n[{city}]")
        print(f"    Matched rows     : {len(rows):,}")
        print(f"    Unique locations : {len(locs)}")
        print(f"    Date range       : {dates[0]} to {dates[-1]}  ({len(dates)} unique dates)")
        describe(pm25s, "PM2.5 (ug/m3)")
        describe(aods,  "AOD_055")
        print(f"    Season breakdown : "
              + "  ".join(f"{s}={n}" for s, n in sorted(seasons.items())))

        # Correlation table: each feature vs PM2.5
        features = [("aod_055", aods)]
        for var in WEATHER_VARS:
            vals = [r[var] for r in rows if r[var] is not None]
            pm_  = [r["pm25_daily_avg"] for r in rows if r[var] is not None]
            features.append((var, vals, pm_))

        print(f"\n    {'Feature':<32}  Pearson-r   Spearman-r   n")
        print(f"    {'-'*32}  ----------  ----------  ----")
        # AOD first
        rp = pearson(aods, pm25s); rs = spearman(aods, pm25s)
        print(f"    {'aod_055':<32}  {rp:+.4f}    {rs:+.4f}    {len(pm25s):,}")
        for var in WEATHER_VARS:
            vals = [r[var] for r in rows if r[var] is not None]
            pm_  = [r["pm25_daily_avg"] for r in rows if r[var] is not None]
            if not vals: continue
            rp = pearson(vals, pm_); rs = spearman(vals, pm_)
            print(f"    {var:<32}  {rp:+.4f}    {rs:+.4f}    {len(pm_):,}")

    print()

# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

CITY_COLORS = {"Delhi": "#e05c2a", "Karachi": "#2a7ae0", "Mumbai": "#2aae6e"}

def scatter_plot(joined: list[dict], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7f8fa")
    ax.grid(True, color="#e5e7eb", linewidth=0.7, zorder=0)

    by_city = collections.defaultdict(list)
    for r in joined: by_city[r["city"]].append(r)

    for city, rows in sorted(by_city.items()):
        ax.scatter([r["aod_055"] for r in rows],
                   [r["pm25_daily_avg"] for r in rows],
                   color=CITY_COLORS.get(city, "grey"),
                   alpha=0.30, s=14, linewidths=0, label=city, zorder=3)

    ax.set_xlabel("MODIS MAIAC AOD at 0.55 um", fontsize=11)
    ax.set_ylabel("Ground PM2.5 (ug/m3)", fontsize=11)
    city_list = ", ".join(sorted(by_city.keys()))
    ax.set_title(f"Satellite AOD vs. Ground-level PM2.5\n{city_list}  2023–2025", fontsize=12)

    handles = [mpatches.Patch(color=CITY_COLORS.get(c, "grey"), label=c)
               for c in sorted(by_city.keys())]
    ax.legend(handles=handles, fontsize=10, framealpha=0.9)

    y_pos = 0.97
    for city, rows in sorted(by_city.items()):
        aods  = [r["aod_055"]        for r in rows]
        pm25s = [r["pm25_daily_avg"] for r in rows]
        rp    = pearson(aods, pm25s)
        ax.text(0.02, y_pos, f"{city}  r = {rp:+.3f}",
                transform=ax.transAxes, fontsize=9,
                color=CITY_COLORS.get(city, "grey"),
                verticalalignment="top")
        y_pos -= 0.05

    plt.tight_layout()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Scatter plot saved -> {out_path}")

# ---------------------------------------------------------------------------
# Save training CSV
# ---------------------------------------------------------------------------

FIELDS = [
    "date", "location_id", "location_name", "city",
    "latitude", "longitude",
    "pm25_daily_avg", "measurement_count",
    "aod_055",
    "month", "day_of_year", "day_of_week", "season",
] + WEATHER_VARS

def save_training(joined: list[dict], path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(joined)
    print(f"  Training data saved -> {path}  ({len(joined):,} rows, {len(FIELDS)} cols)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading CSVs...")
    ground  = load_ground([("Delhi", DELHI_CSV), ("Karachi", KARACHI_CSV), ("Mumbai", MUMBAI_CSV)])
    aod     = load_aod()
    weather = load_weather()
    print(f"  Ground  rows indexed : {len(ground):,}")
    print(f"  AOD     rows indexed : {len(aod):,}")
    print(f"  Weather rows indexed : {len(weather):,}")

    print("\nJoining on (date, location_id) + weather on (date, city)...")
    joined = build_joined(ground, aod, weather)
    print(f"  Matched rows : {len(joined):,}")

    audit(joined)
    save_training(joined, TRAINING_CSV)
    scatter_plot(joined, SCATTER_PNG)
    print("\nDone.")


if __name__ == "__main__":
    main()
