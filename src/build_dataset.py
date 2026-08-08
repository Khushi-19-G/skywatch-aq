"""
build_dataset.py
----------------
Merges satellite AOD (data/processed/satellite_aod.csv) with ground PM2.5
readings, adds time features, audits the matched dataset, and writes:
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
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AOD_CSV       = "data/processed/satellite_aod.csv"
DELHI_CSV     = "data/raw/ground_pm25_delhi.csv"
KARACHI_CSV   = "data/raw/ground_pm25_karachi.csv"
TRAINING_CSV  = "data/processed/training_data.csv"
SCATTER_PNG   = "data/processed/aod_vs_pm25.png"

# ---------------------------------------------------------------------------
# South-Asia season labels
# ---------------------------------------------------------------------------
def season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8, 9):
        return "monsoon"
    return "autumn"          # Oct–Nov

# ---------------------------------------------------------------------------
# Load CSVs
# ---------------------------------------------------------------------------

def load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_ground(paths: list[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    """
    Returns a dict keyed by (date, location_id) -> ground row,
    with a 'city' field added.
    """
    index: dict[tuple[str, str], dict] = {}
    for city, path in paths:
        for row in load_csv(path):
            key = (row["date"], row["location_id"])
            index[key] = {**row, "city": city}
    return index


def load_aod() -> dict[tuple[str, str], dict]:
    """Returns a dict keyed by (date, location_id) -> aod row."""
    index: dict[tuple[str, str], dict] = {}
    for row in load_csv(AOD_CSV):
        key = (row["date"], row["location_id"])
        index[key] = row
    return index

# ---------------------------------------------------------------------------
# Build joined dataset
# ---------------------------------------------------------------------------

def build_joined(
    ground: dict[tuple, dict],
    aod: dict[tuple, dict],
) -> list[dict]:
    joined: list[dict] = []
    for key, grow in ground.items():
        if key not in aod:
            continue
        arow = aod[key]

        date_obj = datetime.date.fromisoformat(key[0])
        month      = date_obj.month
        doy        = date_obj.timetuple().tm_yday
        dow        = date_obj.weekday()   # 0=Mon … 6=Sun

        joined.append({
            "date":             key[0],
            "location_id":      key[1],
            "location_name":    grow["location_name"],
            "city":             grow["city"],
            "latitude":         float(grow["latitude"]),
            "longitude":        float(grow["longitude"]),
            "pm25_daily_avg":   float(grow["pm25_daily_avg"]),
            "measurement_count": int(grow["measurement_count"]),
            "aod_055":          float(arow["aod_055"]),
            "month":            month,
            "day_of_year":      doy,
            "day_of_week":      dow,
            "season":           season(month),
        })

    joined.sort(key=lambda r: (r["city"], r["location_id"], r["date"]))
    return joined

# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def describe(values: list[float], label: str, indent: int = 4) -> None:
    if not values:
        print(f"{' '*indent}{label}: no data")
        return
    pad = " " * indent
    mn  = min(values)
    mx  = max(values)
    mu  = statistics.mean(values)
    med = statistics.median(values)
    sd  = statistics.stdev(values) if len(values) > 1 else 0.0
    print(f"{pad}{label}: mean={mu:.2f}  median={med:.2f}  "
          f"min={mn:.2f}  max={mx:.2f}  sd={sd:.2f}  n={len(values)}")


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson r; returns nan if degenerate."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx  = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy  = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho via rank correlation."""
    n = len(xs)
    if n < 2:
        return float("nan")
    def ranks(v):
        s = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and v[s[j]] == v[s[j + 1]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[s[k]] = avg_rank
            i = j + 1
        return r
    return pearson(ranks(xs), ranks(ys))

# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit(joined: list[dict]) -> None:
    print("\n" + "=" * 64)
    print("DATASET AUDIT")
    print("=" * 64)
    print(f"Total matched rows (both cities): {len(joined):,}")

    by_city = collections.defaultdict(list)
    for r in joined:
        by_city[r["city"]].append(r)

    for city in ["Delhi", "Karachi"]:
        rows = by_city.get(city, [])
        if not rows:
            print(f"\n[{city}] — no matched rows")
            continue

        dates   = sorted({r["date"] for r in rows})
        locs    = {r["location_id"] for r in rows}
        pm25s   = [r["pm25_daily_avg"] for r in rows]
        aods    = [r["aod_055"] for r in rows]
        r_p     = pearson(aods, pm25s)
        r_s     = spearman(aods, pm25s)

        seasons = collections.Counter(r["season"] for r in rows)

        print(f"\n[{city}]")
        print(f"    Matched rows            : {len(rows):,}")
        print(f"    Unique locations        : {len(locs)}")
        print(f"    Date range              : {dates[0]} to {dates[-1]}")
        print(f"    Unique dates            : {len(dates)}")
        describe(pm25s, "PM2.5 (ug/m3)")
        describe(aods,  "AOD_055")
        print(f"    Pearson  r(AOD, PM2.5)  : {r_p:+.3f}")
        print(f"    Spearman r(AOD, PM2.5)  : {r_s:+.3f}")
        print(f"    Season breakdown        : "
              + "  ".join(f"{s}={n}" for s, n in sorted(seasons.items())))

    print()

# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

CITY_COLORS = {"Delhi": "#e05c2a", "Karachi": "#2a7ae0"}

def scatter_plot(joined: list[dict], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7f8fa")
    ax.grid(True, color="#e5e7eb", linewidth=0.7, zorder=0)

    by_city = collections.defaultdict(list)
    for r in joined:
        by_city[r["city"]].append(r)

    for city, rows in by_city.items():
        aods  = [r["aod_055"]        for r in rows]
        pm25s = [r["pm25_daily_avg"] for r in rows]
        ax.scatter(aods, pm25s,
                   color=CITY_COLORS.get(city, "grey"),
                   alpha=0.35, s=18, linewidths=0,
                   label=city, zorder=3)

    ax.set_xlabel("MODIS MAIAC AOD at 0.55 µm", fontsize=11)
    ax.set_ylabel("Ground PM2.5 (µg/m³)", fontsize=11)
    ax.set_title("Satellite AOD vs. Ground-level PM2.5\n"
                 "Delhi & Karachi · 2023-2025", fontsize=12)

    handles = [
        mpatches.Patch(color=CITY_COLORS[c], label=c)
        for c in ["Delhi", "Karachi"] if c in by_city
    ]
    ax.legend(handles=handles, fontsize=10, framealpha=0.9)

    # Annotate Pearson r per city
    y_pos = 0.97
    for city, rows in sorted(by_city.items()):
        aods  = [r["aod_055"]        for r in rows]
        pm25s = [r["pm25_daily_avg"] for r in rows]
        r_p   = pearson(aods, pm25s)
        ax.text(0.02, y_pos, f"{city}  r = {r_p:+.3f}",
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
# Save CSV
# ---------------------------------------------------------------------------

FIELDS = [
    "date", "location_id", "location_name", "city",
    "latitude", "longitude",
    "pm25_daily_avg", "measurement_count",
    "aod_055",
    "month", "day_of_year", "day_of_week", "season",
]

def save_training(joined: list[dict], path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(joined)
    print(f"  Training data saved -> {path}  ({len(joined):,} rows)")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading CSVs...")
    ground = load_ground([
        ("Delhi",   DELHI_CSV),
        ("Karachi", KARACHI_CSV),
    ])
    aod = load_aod()
    print(f"  Ground rows indexed : {len(ground):,}")
    print(f"  AOD rows indexed    : {len(aod):,}")

    print("\nJoining on (date, location_id)...")
    joined = build_joined(ground, aod)
    print(f"  Matched rows        : {len(joined):,}")

    audit(joined)
    save_training(joined, TRAINING_CSV)
    scatter_plot(joined, SCATTER_PNG)
    print("\nDone.")


if __name__ == "__main__":
    main()
