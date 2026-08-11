"""
population_utils.py
-------------------
Population-exposure layer for the SkyWatch-AQ regional map.

Provides per-health-band population exposure counts: "X million people
estimated to be breathing [band]+ air today."

DATA SOURCE
-----------
WorldPop Global Mosaic 2020 — country-level population counts at ~1 km
resolution (100 m resampled to 1 km), aggregated here to 0.25-degree
(~28 km) grid cells to match our map resolution.

Files downloaded per country (on first use, ~3–12 MB each, stored in
data/population/):
  IND — India   : https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/IND/ind_ppp_2020_1km_Aggregated.tif
  PAK — Pakistan: https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/PAK/pak_ppp_2020_1km_Aggregated.tif
  BGD — Bangladesh: https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/BGD/bgd_ppp_2020_1km_Aggregated.tif
  AFG — Afghanistan: https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/AFG/afg_ppp_2020_1km_Aggregated.tif
  NPL — Nepal   : https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/NPL/npl_ppp_2020_1km_Aggregated.tif

License: Creative Commons Attribution 4.0 International (CC BY 4.0)
  Citation: WorldPop (www.worldpop.org) - School of Geography and
  Environmental Science, University of Southampton; Department of
  Geography and Geosciences, University of Louisville; Departement de
  Geographie, Universite de Namur; Center for International Earth Science
  Information Network (CIESIN), Columbia University (2018). Global High
  Resolution Population Denominators Project - Funded by The Bill and
  Melinda Gates Foundation (OPP1134076).

After first download, grids are merged and saved as a compact regional
numpy archive: data/population/population_grid_0p25deg.npz
  Keys: 'pop'  (2-D float32 array, rows=lats, cols=lons)
        'lats' (1-D float64 centre latitudes, ascending)
        'lons' (1-D float64 centre longitudes, ascending)

Size of resulting .npz for the South Asia bounding box: ~1–3 MB.
The .npz is committed to the repo so the deployed app never needs to
re-download; country TIFFs are in data/population/ (gitignored, ~50 MB).

PUBLIC API
----------
  ensure_population_grid()        -> pathlib.Path  (build/download if absent)
  load_population_grid()          -> dict with keys pop/lats/lons
  compute_exposure(cells, grid)   -> dict:
      {
        "by_band":   {band_label: population_millions},
        "no_data_pop": population_millions,          # cloud-covered land
        "total_land_pop": population_millions,
      }
  exposure_headline(exposure)     -> str  (one-line summary for UI)
"""

import math
import pathlib
import sys
import time
from typing import Optional

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SRC_DIR  = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
POP_DIR   = _REPO_ROOT / "data" / "population"
NPZ_PATH  = POP_DIR / "population_grid_0p25deg.npz"

# ---------------------------------------------------------------------------
# WorldPop country TIFFs covering South Asia
# Bounding box for coverage: lat 5–37°N, lon 60–97°E
# ---------------------------------------------------------------------------
GRID_DEG  = 0.25   # 0.25-degree (~28 km) resolution for aggregated grid

# Countries included — those whose extent overlaps lat 5–37, lon 60–97
WORLDPOP_URLS = {
    "IND": "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/IND/ind_ppp_2020_1km_Aggregated.tif",
    "PAK": "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/PAK/pak_ppp_2020_1km_Aggregated.tif",
    "BGD": "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/BGD/bgd_ppp_2020_1km_Aggregated.tif",
    "AFG": "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/AFG/afg_ppp_2020_1km_Aggregated.tif",
    "NPL": "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/NPL/npl_ppp_2020_1km_Aggregated.tif",
}

# Bounding box for the aggregated regional grid
LAT_MIN, LAT_MAX =  5.0, 37.0
LON_MIN, LON_MAX = 60.0, 97.0


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_tif(url: str, dest: pathlib.Path, progress_cb=None) -> bool:
    """Download a file with streaming, show progress, return success."""
    try:
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if progress_cb and total:
                    pct = 100 * downloaded / total
                    progress_cb(f"  {dest.name}: {pct:.0f}% ({downloaded // 1024 // 1024} MB)")
        return True
    except Exception as e:
        if progress_cb:
            progress_cb(f"  Failed to download {url}: {e}")
        print(f"  [population] Download failed for {url}: {e}")
        return False


# ---------------------------------------------------------------------------
# Build the aggregated 0.25° grid from country TIFFs
# ---------------------------------------------------------------------------

def _build_grid_from_tifs(tif_paths: list[pathlib.Path], progress_cb=None) -> Optional[np.ndarray]:
    """
    Read each country TIFF with rasterio, mosaic, aggregate to 0.25° grid.
    Returns (pop_array, lats, lons) or None on failure.
    """
    try:
        import rasterio
        from rasterio.merge import merge as rio_merge
        import numpy as np
    except ImportError as e:
        if progress_cb:
            progress_cb(f"  [population] rasterio not available: {e}")
        print(f"  [population] rasterio import error: {e}")
        return None

    # Build lat/lon edges for 0.25° grid
    lons_edges = np.arange(LON_MIN, LON_MAX + GRID_DEG, GRID_DEG)
    lats_edges = np.arange(LAT_MIN, LAT_MAX + GRID_DEG, GRID_DEG)
    n_lat = len(lats_edges) - 1
    n_lon = len(lons_edges) - 1
    pop_grid = np.zeros((n_lat, n_lon), dtype=np.float64)

    for tif_path in tif_paths:
        if not tif_path.exists():
            continue
        try:
            with rasterio.open(str(tif_path)) as ds:
                if progress_cb:
                    progress_cb(f"  Aggregating {tif_path.name} ...")
                print(f"  [population] Processing {tif_path.name} "
                      f"({ds.width}x{ds.height}, CRS={ds.crs})")

                # Read full raster (already at ~1km = ~0.00833°)
                arr = ds.read(1, masked=True).astype(np.float64)
                arr = np.where(arr.mask | (arr.data < 0), 0.0, arr.data)

                # Get pixel coordinates
                transform = ds.transform
                nrows, ncols = arr.shape
                # Upper-left corner + pixel size
                west  = transform.c
                north = transform.f
                dx    = transform.a   # pixel width  (positive)
                dy    = transform.e   # pixel height (negative for north-up)

                # Clip to our bounding box
                # Row/col range that falls in [LAT_MIN, LAT_MAX] x [LON_MIN, LON_MAX]
                col_start = max(0, math.floor((LON_MIN - west) / dx))
                col_end   = min(ncols, math.ceil((LON_MAX - west) / dx))
                row_start = max(0, math.floor((north - LAT_MAX) / (-dy)))
                row_end   = min(nrows, math.ceil((north - LAT_MIN) / (-dy)))

                if col_start >= col_end or row_start >= row_end:
                    continue  # country doesn't overlap our bbox

                sub_arr = arr[row_start:row_end, col_start:col_end]
                sub_rows, sub_cols = sub_arr.shape

                # Accumulate into 0.25° grid
                for r in range(sub_rows):
                    # Lat of pixel centre
                    lat_c = north + dy * (row_start + r + 0.5)
                    if lat_c < LAT_MIN or lat_c > LAT_MAX:
                        continue
                    gi = int((lat_c - LAT_MIN) / GRID_DEG)
                    if gi < 0 or gi >= n_lat:
                        continue
                    for c in range(sub_cols):
                        lon_c = west + dx * (col_start + c + 0.5)
                        if lon_c < LON_MIN or lon_c > LON_MAX:
                            continue
                        gj = int((lon_c - LON_MIN) / GRID_DEG)
                        if gj < 0 or gj >= n_lon:
                            continue
                        pop_grid[gi, gj] += sub_arr[r, c]
        except Exception as e:
            print(f"  [population] Error reading {tif_path}: {e}")
            continue

    # Centre coordinates
    lats = np.array([LAT_MIN + (i + 0.5) * GRID_DEG for i in range(n_lat)])
    lons = np.array([LON_MIN + (j + 0.5) * GRID_DEG for j in range(n_lon)])

    return pop_grid.astype(np.float32), lats, lons


# ---------------------------------------------------------------------------
# Public: ensure NPZ is present (download + build if missing)
# ---------------------------------------------------------------------------

def ensure_population_grid(progress_cb=None) -> pathlib.Path:
    """
    Ensure data/population/population_grid_0p25deg.npz exists.
    If not, download country TIFFs from WorldPop and build it.
    Returns path to NPZ.
    """
    def log(msg: str) -> None:
        print(msg)
        if progress_cb:
            progress_cb(msg)

    if NPZ_PATH.exists():
        return NPZ_PATH

    log("[population] Building population grid...")
    POP_DIR.mkdir(parents=True, exist_ok=True)

    tif_paths = []
    for iso, url in WORLDPOP_URLS.items():
        dest = POP_DIR / f"{iso.lower()}_ppp_2020_1km.tif"
        tif_paths.append(dest)
        if dest.exists():
            log(f"[population] Found cached TIF: {dest.name}")
        else:
            log(f"[population] Downloading {iso} population (~3–12 MB)...")
            ok = _download_tif(url, dest, progress_cb=log)
            if not ok:
                log(f"[population] WARNING: could not download {iso} TIF, skipping.")

    available = [p for p in tif_paths if p.exists()]
    if not available:
        log("[population] No TIFs available — population exposure disabled.")
        # Write empty placeholder so we don't retry every load
        lats = np.array([LAT_MIN + (i + 0.5) * GRID_DEG
                         for i in range(int((LAT_MAX - LAT_MIN) / GRID_DEG))])
        lons = np.array([LON_MIN + (j + 0.5) * GRID_DEG
                         for j in range(int((LON_MAX - LON_MIN) / GRID_DEG))])
        pop = np.zeros((len(lats), len(lons)), dtype=np.float32)
        np.savez_compressed(str(NPZ_PATH), pop=pop, lats=lats, lons=lons)
        return NPZ_PATH

    log(f"[population] Aggregating {len(available)} TIF(s) to 0.25° grid...")
    result = _build_grid_from_tifs(available, progress_cb=log)
    if result is None:
        log("[population] Grid build failed — writing empty placeholder.")
        lats = np.array([LAT_MIN + (i + 0.5) * GRID_DEG
                         for i in range(int((LAT_MAX - LAT_MIN) / GRID_DEG))])
        lons = np.array([LON_MIN + (j + 0.5) * GRID_DEG
                         for j in range(int((LON_MAX - LON_MIN) / GRID_DEG))])
        pop = np.zeros((len(lats), len(lons)), dtype=np.float32)
        np.savez_compressed(str(NPZ_PATH), pop=pop, lats=lats, lons=lons)
        return NPZ_PATH

    pop_grid, lats, lons = result
    np.savez_compressed(str(NPZ_PATH), pop=pop_grid, lats=lats, lons=lons)
    total_pop = float(pop_grid.sum())
    log(f"[population] Grid saved: {NPZ_PATH.name} "
        f"({pop_grid.shape[0]}×{pop_grid.shape[1]} cells, "
        f"total population {total_pop / 1e6:.1f} M)")
    return NPZ_PATH


# ---------------------------------------------------------------------------
# Public: load grid (cached in-process)
# ---------------------------------------------------------------------------

_grid_cache: Optional[dict] = None


def load_population_grid() -> Optional[dict]:
    """
    Load the population grid NPZ.
    Returns dict with keys 'pop' (2-D float32), 'lats', 'lons',
    or None if the grid is unavailable / all-zero (disabled).
    """
    global _grid_cache
    if _grid_cache is not None:
        return _grid_cache

    if not NPZ_PATH.exists():
        ensure_population_grid()

    if not NPZ_PATH.exists():
        return None

    try:
        data = np.load(str(NPZ_PATH))
        grid = {
            "pop":  data["pop"],   # (n_lat, n_lon) float32
            "lats": data["lats"],  # (n_lat,) ascending
            "lons": data["lons"],  # (n_lon,) ascending
        }
        if grid["pop"].sum() == 0:
            return None  # placeholder / disabled
        _grid_cache = grid
        return grid
    except Exception as e:
        print(f"  [population] Failed to load NPZ: {e}")
        return None


# ---------------------------------------------------------------------------
# Core exposure computation
# ---------------------------------------------------------------------------

def _grid_pop_for_cell(lat: float, lon: float, grid: dict) -> float:
    """
    Return estimated population in the ~0.25° map cell centred at (lat, lon).
    Uses nearest-neighbour lookup against the population grid.
    Returns 0.0 if outside bounds.
    """
    lats = grid["lats"]
    lons = grid["lons"]
    pop  = grid["pop"]

    # Nearest index
    i = int(round((lat - lats[0]) / GRID_DEG))
    j = int(round((lon - lons[0]) / GRID_DEG))

    if 0 <= i < pop.shape[0] and 0 <= j < pop.shape[1]:
        return float(pop[i, j])
    return 0.0


def compute_exposure(cells: list[dict], grid: dict,
                     region_bbox: tuple[float, float, float, float] | None = None,
                     ) -> dict:
    """
    Given a list of map cells (each with 'lat', 'lon', 'band_label') and
    a population grid, compute population exposed per health band.

    Also estimates population in cloud-covered / no-data cells: we sum all
    population grid points inside the region bounding box that are NOT
    matched by a data cell, and report that as "no estimate available".

    Parameters
    ----------
    cells       : list of dicts from build_region_map / load_cached_map
    grid        : dict from load_population_grid()
    region_bbox : optional (lat_min, lat_max, lon_min, lon_max) to constrain
                  the no-data population estimate.  When None, derived from
                  the data cell extents ± GRID_DEG.

    Returns
    -------
    dict:
      {
        "available":      True,
        "by_band":        {band_label: float},   # millions, per band
        "no_data_pop":    float,                  # millions in cloud gaps
        "total_land_pop": float,                  # millions (data + no-data)
      }
    or {"available": False} if grid is unavailable.
    """
    if grid is None:
        return {"available": False}

    lats_g = grid["lats"]   # (n_lat,) ascending
    lons_g = grid["lons"]   # (n_lon,) ascending
    pop    = grid["pop"]    # (n_lat, n_lon) float32

    # ---- 1. Per-band exposure (data cells) ----------------------------------
    # Convert each data cell (lat, lon) -> population grid indices, vectorised.
    by_band: dict[str, float] = {}

    if cells:
        c_lats = np.array([c["lat"] for c in cells], dtype=np.float64)
        c_lons = np.array([c["lon"] for c in cells], dtype=np.float64)
        # Nearest-neighbour index into population grid
        i_idx = np.clip(np.round((c_lats - lats_g[0]) / GRID_DEG).astype(int),
                        0, pop.shape[0] - 1)
        j_idx = np.clip(np.round((c_lons - lons_g[0]) / GRID_DEG).astype(int),
                        0, pop.shape[1] - 1)
        pops = pop[i_idx, j_idx].astype(np.float64)   # (n_cells,)

        for k, cell in enumerate(cells):
            band = cell.get("band_label", "Unknown")
            by_band[band] = by_band.get(band, 0.0) + float(pops[k])

    by_band_m = {k: v / 1e6 for k, v in by_band.items()}
    total_data_pop = sum(by_band.values())

    # ---- 2. No-data (cloud gap) population -----------------------------------
    # Region bounding box: use provided bbox or derive from data extents.
    if region_bbox is not None:
        lat_min, lat_max, lon_min, lon_max = region_bbox
    elif cells:
        lat_min = float(c_lats.min()) - GRID_DEG
        lat_max = float(c_lats.max()) + GRID_DEG
        lon_min = float(c_lons.min()) - GRID_DEG
        lon_max = float(c_lons.max()) + GRID_DEG
    else:
        lat_min, lat_max = LAT_MIN, LAT_MAX
        lon_min, lon_max = LON_MIN, LON_MAX

    # Indices into population grid for the bbox
    gi_lo = max(0, int(np.searchsorted(lats_g, lat_min)))
    gi_hi = min(pop.shape[0], int(np.searchsorted(lats_g, lat_max, side="right")))
    gj_lo = max(0, int(np.searchsorted(lons_g, lon_min)))
    gj_hi = min(pop.shape[1], int(np.searchsorted(lons_g, lon_max, side="right")))

    # Build a boolean "has data" mask over the population sub-grid
    has_data = np.zeros((gi_hi - gi_lo, gj_hi - gj_lo), dtype=bool)
    if cells:
        # Mark each data cell's nearest pop-grid cell as covered
        i_sub = np.clip(i_idx - gi_lo, 0, has_data.shape[0] - 1)
        j_sub = np.clip(j_idx - gj_lo, 0, has_data.shape[1] - 1)
        # Only mark cells that are actually inside the bbox
        in_bbox = ((i_idx >= gi_lo) & (i_idx < gi_hi) &
                   (j_idx >= gj_lo) & (j_idx < gj_hi))
        has_data[i_sub[in_bbox], j_sub[in_bbox]] = True

    sub_pop = pop[gi_lo:gi_hi, gj_lo:gj_hi].astype(np.float64)
    no_data_pop = float(sub_pop[~has_data].sum())
    total_land_pop = total_data_pop + no_data_pop

    return {
        "available":      True,
        "by_band":        by_band_m,
        "no_data_pop":    no_data_pop / 1e6,
        "total_land_pop": total_land_pop / 1e6,
    }


# ---------------------------------------------------------------------------
# Headline string
# ---------------------------------------------------------------------------

# Health band severity order (most severe first for headline selection)
_BAND_SEVERITY = [
    "Hazardous",
    "Very Unhealthy",
    "Unhealthy",
    "USG",
    "Moderate",
    "Good",
]


def exposure_headline(exposure: dict, suffix: str = "today") -> str:
    """
    Return the one-line headline metric string for display in the app.
    E.g.: "~28.3 M people estimated in Unhealthy+ air today"

    Pass suffix="" to omit the trailing time word (used when a forecast_prefix
    already carries the temporal context, e.g. "Tomorrow: ~X M people...").

    Returns empty string if exposure data is unavailable.
    """
    if not exposure.get("available"):
        return ""

    by_band = exposure["by_band"]
    tail = f" {suffix}" if suffix else ""
    # Find most severe band with non-trivial population (>0.05 M)
    for band in _BAND_SEVERITY:
        pop = by_band.get(band, 0.0)
        if pop >= 0.05:
            # Cumulative from this band upward (more severe)
            idx = _BAND_SEVERITY.index(band)
            cum = sum(by_band.get(b, 0.0) for b in _BAND_SEVERITY[:idx + 1])
            return f"~{cum:.1f} M people estimated in {band}+ air{tail}"

    # All in Good
    good_pop = by_band.get("Good", 0.0)
    if good_pop > 0:
        return f"~{good_pop:.1f} M people in Good air quality{tail}"
    return ""


# ---------------------------------------------------------------------------
# CLI: build the grid (run this script directly to pre-generate the NPZ)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def _cli_progress(msg: str) -> None:
        print(msg)

    print("=== Building population grid ===")
    path = ensure_population_grid(progress_cb=_cli_progress)
    grid = load_population_grid()
    if grid is None:
        print("Grid unavailable (all zeros or load failed).")
        sys.exit(1)

    pop = grid["pop"]
    lats = grid["lats"]
    lons = grid["lons"]
    print(f"\nGrid shape: {pop.shape}")
    print(f"Lat range:  {lats[0]:.2f} – {lats[-1]:.2f}")
    print(f"Lon range:  {lons[0]:.2f} – {lons[-1]:.2f}")
    print(f"Total pop:  {pop.sum() / 1e6:.1f} M")
    print(f"NPZ size:   {path.stat().st_size // 1024} KB")

    # Spot check a few known cities
    checks = [
        ("Delhi",    28.61, 77.21),
        ("Mumbai",   19.08, 72.88),
        ("Karachi",  24.86, 67.00),
        ("Lahore",   31.55, 74.34),
        ("Dhaka",    23.81, 90.41),
    ]
    print("\nCity spot-checks (0.25° cell population):")
    for name, lat, lon in checks:
        p = _grid_pop_for_cell(lat, lon, grid)
        print(f"  {name:12s} ({lat:.2f}, {lon:.2f}): {p / 1e6:.2f} M")
