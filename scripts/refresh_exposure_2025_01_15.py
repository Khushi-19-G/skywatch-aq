"""
Refresh the h24v06 2025-01-15 cache to include the population-exposure dict,
then print the per-band population table and the app headline line.

Run from repo root:
    python scripts/refresh_exposure_2025_01_15.py
"""
import sys
import pathlib
import json

# ── repo path setup ──────────────────────────────────────────────────────────
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from population_utils import (
    ensure_population_grid,
    load_population_grid,
    compute_exposure,
    exposure_headline,
    _BAND_SEVERITY,
)
from map_utils import REGIONS, _cache_path

# ── config ───────────────────────────────────────────────────────────────────
REGION_KEY = "h24v06"
DATE_STR   = "2025-01-15"
CACHE_FILE = _cache_path(REGION_KEY, DATE_STR)

# ── 1. ensure population grid is present ─────────────────────────────────────
print("Checking population grid …")
npz = ensure_population_grid(progress_cb=lambda m: print(" ", m))
print(f"Population grid: {npz}")

# ── 2. load population grid ───────────────────────────────────────────────────
pop_grid = load_population_grid()
if pop_grid is None:
    sys.exit("ERROR: population grid could not be loaded.")
total_grid = pop_grid["pop"].sum() / 1e6
print(f"Grid loaded — total pop in grid: {total_grid:.1f} M\n")

# ── 3. load existing cache ────────────────────────────────────────────────────
print(f"Loading cache: {CACHE_FILE.name}")
raw = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
if isinstance(raw, dict) and "cells" in raw:
    cells    = raw["cells"]
    advisory = raw.get("advisory", {})
elif isinstance(raw, list):
    cells    = raw
    advisory = {}
else:
    sys.exit("Unexpected cache format.")
print(f"Cells loaded: {len(cells):,}\n")

# ── 4. compute exposure ───────────────────────────────────────────────────────
bbox = REGIONS[REGION_KEY].get("bbox")
exposure = compute_exposure(cells, pop_grid, region_bbox=bbox)

# ── 5. print per-band table ───────────────────────────────────────────────────
print("=" * 60)
print(f"  Population-exposure table — {REGION_KEY}  {DATE_STR}")
print("=" * 60)
by_band  = exposure.get("by_band", {})
no_data  = exposure.get("no_data_pop", 0.0)
total_ex = exposure.get("total_land_pop", 0.0)

band_order = [b for b in _BAND_SEVERITY if b in by_band]
for band in band_order:
    pop_m = by_band[band]
    bar   = "#" * int(pop_m * 2)
    print(f"  {band:<30s}  {pop_m:6.2f} M  {bar}")

print("-" * 60)
print(f"  Total exposed (known band)  {total_ex:6.2f} M")
print(f"  No-data (cloud/missing)     {no_data:6.2f} M")
print("=" * 60)

# ── 6. show headline the app would render ────────────────────────────────────
headline = exposure_headline(exposure)
print(f"\n  APP HEADLINE:  {headline}\n")

# ── 7. re-save cache with exposure ───────────────────────────────────────────
payload = {"cells": cells, "exposure": exposure, "advisory": advisory}
CACHE_FILE.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
new_size = CACHE_FILE.stat().st_size / 1024
print(f"Cache updated: {CACHE_FILE.name}  ({new_size:.0f} KB)")
