"""
app.py — skywatch-aq Streamlit health-alert app
------------------------------------------------
v3 — adds Regional Map mode:
  Tab 1 "City estimate"  — existing single-city point-estimate flow.
  Tab 2 "Regional map"   — paints PM2.5 across a full MODIS tile (~50x50
    grid, ~22 km resolution) using map_utils.py, rendered as a Plotly
    scatter_mapbox with health-band colouring and hover detail.
"""

import math
import os
import datetime
import pathlib
import tempfile
import time

import requests
import numpy as np
import joblib
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap: load .env and push creds into os.environ so earthaccess finds them
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env", override=True)
for _k in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD"):
    _v = os.getenv(_k, "")
    if _v:
        os.environ[_k] = _v

# Bridge Streamlit Cloud secrets into env vars (no-op locally)

try:
    for _k in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "OPENAQ_API_KEY", "HF_TOKEN"):
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass

st.caption(f"debug: ED_user={bool(os.environ.get('EARTHDATA_USERNAME'))} ED_pass={bool(os.environ.get('EARTHDATA_PASSWORD'))}")


# Add src/ to path for aod_utils + map_utils
import sys
sys.path.insert(0, str(_REPO_ROOT / "src"))
from aod_utils import (
    latlon_to_tile_pixel, tile_id,
    earthaccess_login, find_granule_for_date,
    download_granule, extract_aod_at_point,
)
from map_utils import (
    REGIONS, build_region_map, load_cached_map,
    HEALTH_BANDS as MAP_HEALTH_BANDS,
)
from population_utils import (
    load_population_grid, compute_exposure, exposure_headline,
    ensure_population_grid,
    _BAND_SEVERITY as POP_BAND_SEVERITY,
)
from forecast_utils import (
    is_forecast_date, forecast_label,
    fetch_forecast_weather_point,
    build_forecast_map, load_forecast_cache,
    AOD_LOOKBACK_DAYS,
)
from advisory_utils import build_advisory_context, generate_advisory
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = _REPO_ROOT / "models" / "pm25_model.pkl"

# ---------------------------------------------------------------------------
# Known cities  {display_name: (lat, lon, typical_aod, openaq_coords_or_None)}
# openaq_coords is the "lat,lon" string to search OpenAQ /locations
# ---------------------------------------------------------------------------
CITIES = {
    "Delhi, India":       (28.6139, 77.2090, 0.65, "28.6139,77.2090"),
    "Mumbai, India":      (19.0760, 72.8777, 0.52, "19.0760,72.8777"),
    "Karachi, Pakistan":  (24.8607, 67.0011, 0.45, "24.8607,67.0011"),
    "Lahore, Pakistan":   (31.5497, 74.3436, 0.70, None),
    "Dhaka, Bangladesh":  (23.8103, 90.4125, 0.60, None),
    "Kolkata, India":     (22.5726, 88.3639, 0.58, None),
    "Dhanbad, India":     (23.7957, 86.4304, 0.55, None),
    "Custom location":    (None,    None,    0.50, None),
}

# ---------------------------------------------------------------------------
# Season helper
# ---------------------------------------------------------------------------
def season(month: int) -> str:
    if month in (12, 1, 2):   return "winter"
    if month in (3, 4, 5):    return "spring"
    if month in (6, 7, 8, 9): return "monsoon"
    return "autumn"

SEASON_ENCODE = {"winter": 0, "spring": 1, "monsoon": 2, "autumn": 3}

# ---------------------------------------------------------------------------
# Region confidence
# ---------------------------------------------------------------------------
# Anchor cities used for training/validation, with 500 km radius
_VALIDATED_ANCHORS = [
    ("Delhi",   28.6139, 77.2090),
    ("Mumbai",  19.0760, 72.8777),
    ("Karachi", 24.8607, 67.0011),
]
_VALIDATED_RADIUS_KM = 500.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two WGS-84 points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def region_confidence(lat: float, lon: float) -> tuple[bool, str]:
    """
    Returns (is_validated, badge_text).
    Validated if within _VALIDATED_RADIUS_KM of any training anchor.
    """
    for name, alat, alon in _VALIDATED_ANCHORS:
        dist = _haversine_km(lat, lon, alat, alon)
        if dist <= _VALIDATED_RADIUS_KM:
            return True, f"Validated region (within {dist:.0f} km of {name})"
    # Find nearest anchor for context
    nearest = min(_VALIDATED_ANCHORS, key=lambda a: _haversine_km(lat, lon, a[1], a[2]))
    dist = _haversine_km(lat, lon, nearest[1], nearest[2])
    return False, f"Lower confidence — outside the validated region ({dist:.0f} km from {nearest[0]}); estimates are indicative only."

# ---------------------------------------------------------------------------
# AQI / health categories  (PM2.5 µg/m³)
# ---------------------------------------------------------------------------
CATEGORIES = [
    (  0,  12,  "Good",                        "#22c55e",
     "Air quality is satisfactory. No health risk for the general population."),
    ( 12,  35,  "Moderate",                    "#84cc16",
     "Air quality is acceptable. Sensitive individuals may experience minor symptoms."),
    ( 35,  55,  "Unhealthy for Sensitive Groups", "#eab308",
     "Children, elderly and those with respiratory or heart conditions "
     "should limit prolonged outdoor exertion."),
    ( 55, 150,  "Unhealthy",                   "#f97316",
     "Everyone may begin to experience health effects. "
     "Reduce outdoor activities, especially strenuous exercise."),
    (150, 250,  "Very Unhealthy",              "#ef4444",
     "Health alert: everyone should avoid prolonged outdoor exertion. "
     "Sensitive groups must stay indoors."),
    (250, 9999, "Hazardous",                   "#7c3aed",
     "Health emergency. Entire population is likely to be affected. "
     "Stay indoors with windows closed."),
]

def categorise(pm25: float) -> tuple[str, str, str]:
    for lo, hi, label, color, advice in CATEGORIES:
        if lo <= pm25 < hi:
            return label, color, advice
    return "Hazardous", "#7c3aed", CATEGORIES[-1][4]

# ---------------------------------------------------------------------------
# Open-Meteo weather fetch
# ---------------------------------------------------------------------------
WEATHER_VARS = [
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "precipitation_sum",
    "surface_pressure_mean",
    "shortwave_radiation_sum",
]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather(lat: float, lon: float, date_str: str) -> dict | None:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": date_str,
        "end_date":   date_str,
        "daily":      ",".join(WEATHER_VARS),
        "timezone":   "auto",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        daily = r.json().get("daily", {})
        return {v: (daily.get(v, [None])[0]) for v in WEATHER_VARS}
    except Exception:
        return None

# ---------------------------------------------------------------------------
# AUTO-FETCH SATELLITE AOD  (Part 1)
# ---------------------------------------------------------------------------

AOD_CACHE_DIR = pathlib.Path(tempfile.gettempdir()) / "skywatch_aod_cache"

@st.cache_data(ttl=86400, show_spinner=False)   # cache 24 h — one download per city/date
def fetch_aod_auto(lat: float, lon: float, date_str: str) -> tuple[float | None, str | None]:
    """
    Search MCD19A2.061 for the most recent valid granule within a 7-day
    lookback of `date_str`.  Download the granule, extract QA-filtered AOD
    at (lat, lon), delete the file, return (aod_value, actual_date).
    Returns (None, None) on any failure or cloud cover.
    """
    AOD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Determine tile
    tp = latlon_to_tile_pixel(lat, lon)
    tid = tile_id(tp.h, tp.v)

    # Login (idempotent)
    try:
        earthaccess_login()
    except Exception:
        return None, None

    # Find granule
    granule, granule_date = find_granule_for_date(tid, date_str, lookback_days=7)
    if granule is None:
        return None, None

    # Download
    hdf_path = download_granule(granule, AOD_CACHE_DIR)
    if hdf_path is None:
        return None, None

    # Extract
    try:
        aod = extract_aod_at_point(hdf_path, lat, lon)
    finally:
        try:
            hdf_path.unlink()
        except OSError:
            pass

    return aod, granule_date

# ---------------------------------------------------------------------------
# REALITY CHECK — OpenAQ ground truth  (Part 2)
# ---------------------------------------------------------------------------
OPENAQ_API_KEY  = os.getenv("OPENAQ_API_KEY", "")
OPENAQ_BASE     = "https://api.openaq.org/v3"
OPENAQ_HEADERS  = {"X-API-Key": OPENAQ_API_KEY} if OPENAQ_API_KEY else {}
PM25_PARAM_ID   = 2
OPENAQ_RADIUS_M = 25_000

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ground_truth(openaq_coords: str, date_str: str) -> float | None:
    """
    Fetch the average PM2.5 reading across all OpenAQ monitors near
    openaq_coords on date_str.  Returns a float or None.
    Never used in the prediction.

    Note: the OpenAQ v3 /sensors/{id}/days endpoint only returns results when
    a multi-day range is provided; we query a ±7-day window and filter locally.
    """
    # 1. Find locations
    try:
        r = requests.get(
            f"{OPENAQ_BASE}/locations",
            params={
                "coordinates":   openaq_coords,
                "radius":        OPENAQ_RADIUS_M,
                "parameters_id": PM25_PARAM_ID,
                "limit":         100,
            },
            headers=OPENAQ_HEADERS,
            timeout=20,
        )
        r.raise_for_status()
        locations = r.json().get("results") or []
    except Exception:
        return None

    if not locations:
        return None

    # Gather PM2.5 sensor IDs
    sensor_ids: list[int] = []
    for loc in locations:
        for s in (loc.get("sensors") or []):
            if (s.get("parameter") or {}).get("name") == "pm25":
                sensor_ids.append(s["id"])

    if not sensor_ids:
        return None

    # 2. Query /sensors/{id}/days with a ±7-day window, filter to target date locally.
    # The API requires a multi-day range to return data; single-day queries return empty.
    import datetime as _dt
    target     = _dt.date.fromisoformat(date_str)
    date_from  = (target - _dt.timedelta(days=7)).isoformat()
    date_to    = (target + _dt.timedelta(days=1)).isoformat()   # inclusive buffer

    day_values: list[float] = []

    for sid in sensor_ids[:30]:   # cap to avoid hammering API
        try:
            time.sleep(0.3)
            r = requests.get(
                f"{OPENAQ_BASE}/sensors/{sid}/days",
                params={
                    "date_from": date_from,
                    "date_to":   date_to,
                    "limit":     1000,
                },
                headers=OPENAQ_HEADERS,
                timeout=15,
            )
            if r.status_code != 200:
                continue
            results = r.json().get("results") or []
            for rec in results:
                # Filter to the target date (local calendar day)
                period   = rec.get("period") or {}
                dt_local = (period.get("datetimeFrom") or {}).get("local", "")
                if not dt_local[:10] == date_str:
                    continue
                val = rec.get("value")
                if val is None:
                    val = (rec.get("summary") or {}).get("avg")
                if val is not None:
                    day_values.append(float(val))
        except Exception:
            continue

    if not day_values:
        return None
    return round(sum(day_values) / len(day_values), 1)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict(bundle: dict, aod: float, date: datetime.date, weather: dict) -> float:
    m = date.month
    row = [
        aod,
        m,
        date.timetuple().tm_yday,
        date.weekday(),
        SEASON_ENCODE[season(m)],
        weather.get("temperature_2m_mean")       or 25.0,
        weather.get("relative_humidity_2m_mean") or 50.0,
        weather.get("wind_speed_10m_mean")        or 3.0,
        weather.get("precipitation_sum")          or 0.0,
        weather.get("surface_pressure_mean")      or 1010.0,
        weather.get("shortwave_radiation_sum")    or 15.0,
    ]
    X = np.array([row], dtype=np.float64)
    return float(max(0.0, math.expm1(bundle["model"].predict(X)[0])))

# ---------------------------------------------------------------------------
# Map figure builder
# ---------------------------------------------------------------------------

def _cells_to_figure(cells: list[dict], region_key: str, date_str: str) -> go.Figure | None:
    """Convert a list of map cells to a Plotly figure."""
    if not cells:
        return None

    region = REGIONS[region_key]

    # Build per-band traces so legend entries match health bands
    fig = go.Figure()

    band_order = [b[2] for b in MAP_HEALTH_BANDS]
    band_colors = {b[2]: b[3] for b in MAP_HEALTH_BANDS}

    # Group cells by band
    by_band: dict[str, list[dict]] = {b: [] for b in band_order}
    for cell in cells:
        by_band.setdefault(cell["band_label"], []).append(cell)

    for band_label in band_order:
        band_cells = by_band.get(band_label, [])
        if not band_cells:
            continue
        lats  = [c["lat"]  for c in band_cells]
        lons  = [c["lon"]  for c in band_cells]
        texts = [
            f"<b>{c['band_label']}</b><br>"
            f"PM2.5: {c['pm25']:.1f} ug/m3<br>"
            f"AOD: {c['aod']:.3f}<br>"
            f"({c['lat']:.2f}N, {c['lon']:.2f}E)"
            for c in band_cells
        ]
        fig.add_trace(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode="markers",
            marker=dict(
                size=7,
                color=band_colors[band_label],
                opacity=0.82,
            ),
            text=texts,
            hovertemplate="%{text}<extra></extra>",
            name=band_label,
            legendgroup=band_label,
        ))

    # Overlay validated city markers
    for city in region["validated_cities"]:
        fig.add_trace(go.Scattermapbox(
            lat=[city["lat"]],
            lon=[city["lon"]],
            mode="markers+text",
            marker=dict(size=14, color="#1f2328", symbol="circle"),
            text=[city["name"]],
            textposition="top right",
            textfont=dict(size=12, color="#1f2328"),
            hovertemplate=f"<b>{city['name']}</b> (validated city)<extra></extra>",
            name=city["name"],
            legendgroup=city["name"],
        ))

    n_cells = len(cells)
    # n_land was computed at map_utils import time and stored in REGIONS.
    n_land  = REGIONS.get(region_key, {}).get("n_land", 2500)
    pct     = 100 * n_cells / max(n_land, 1)
    title_text = (
        f"Estimated air quality — {region['label']}, {date_str}<br>"
        f"<sup>{n_cells:,} of {n_land:,} land cells have satellite data ({pct:.0f}%); "
        f"grey gaps = cloud cover or ocean (no estimate). "
        f"Every coloured cell is a model estimate — most locations have no ground monitor.</sup>"
    )

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=region["centre_lat"], lon=region["centre_lon"]),
            zoom=region["zoom"],
        ),
        legend=dict(
            title=dict(text="Health band"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e5e7eb",
            borderwidth=1,
        ),
        title=dict(text=title_text, font=dict(size=13)),
        margin=dict(l=0, r=0, t=70, b=0),
        height=580,
        dragmode="pan",          # default interaction: pan (not select/zoom-box)
    )
    return fig


# ---------------------------------------------------------------------------
# Population-exposure UI helper
# ---------------------------------------------------------------------------

# Band colors for the exposure breakdown bars
_BAND_COLORS_MAP = {b[2]: b[3] for b in MAP_HEALTH_BANDS}


def _render_exposure(exposure: dict, forecast_prefix: str = "") -> None:
    """
    Render the population-exposure headline metric row above the map.
    Shows: headline stat, per-band breakdown table/bars, unknown-coverage line.
    forecast_prefix — if non-empty, prepended to the headline (e.g. "Tomorrow: ").
    """
    if not exposure.get("available"):
        return

    by_band = exposure.get("by_band", {})
    no_data = exposure.get("no_data_pop", 0.0)

    # For forecast prefix, drop "today" from the headline stem
    headline = exposure_headline(exposure, suffix="" if forecast_prefix else "today")
    if not headline:
        return

    if forecast_prefix:
        headline = f"{forecast_prefix}: {headline}"

    # --- Headline ---
    border_color = "#6366f1" if forecast_prefix else "#f97316"
    bg_color     = "#eef2ff" if forecast_prefix else "#fff7ed"
    st.markdown(
        f'<div style="background:{bg_color};border-left:5px solid {border_color};'
        f'padding:10px 16px;border-radius:6px;margin-bottom:10px;">'
        f'<span style="font-size:1.1em;font-weight:600;color:#1f2328;">🏙️ {headline}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # --- Breakdown expander ---
    with st.expander("Population exposure breakdown by health band", expanded=False):
        # Table + simple bar representation
        band_order_display = [b for b in POP_BAND_SEVERITY if b in by_band and by_band[b] >= 0.01]

        if band_order_display:
            max_pop = max(by_band.get(b, 0.0) for b in band_order_display)
            rows_html = ""
            for band in band_order_display:
                pop_m = by_band.get(band, 0.0)
                color = _BAND_COLORS_MAP.get(band, "#aaaaaa")
                bar_w = int(200 * pop_m / max(max_pop, 0.01))
                rows_html += (
                    f'<tr>'
                    f'<td style="padding:4px 10px 4px 0;white-space:nowrap;">'
                    f'<span style="background:{color};color:white;padding:2px 8px;'
                    f'border-radius:4px;font-size:0.85em;font-weight:600;">{band}</span></td>'
                    f'<td style="padding:4px 6px;text-align:right;font-weight:600;'
                    f'font-variant-numeric:tabular-nums;">{pop_m:.1f} M</td>'
                    f'<td style="padding:4px 10px;">'
                    f'<div style="background:{color};height:12px;width:{bar_w}px;'
                    f'border-radius:2px;opacity:0.85;"></div></td>'
                    f'</tr>'
                )

            st.markdown(
                f'<table style="border-collapse:collapse;font-size:0.9em;">'
                f'<thead><tr>'
                f'<th style="text-align:left;padding:4px 10px 4px 0;color:#57606a;">Band</th>'
                f'<th style="text-align:right;padding:4px 6px;color:#57606a;">Est. population</th>'
                f'<th style="padding:4px 10px;color:#57606a;"></th>'
                f'</tr></thead>'
                f'<tbody>{rows_html}</tbody>'
                f'</table>',
                unsafe_allow_html=True,
            )

        # Unknown coverage line
        if no_data > 0.05:
            st.markdown(
                f'<div style="margin-top:8px;color:#57606a;font-size:0.85em;">'
                f'☁️ <strong>No estimate available</strong> for areas holding an estimated '
                f'<strong>{no_data:.1f} M people</strong> (AOD gap / cloud cover — '
                f'their air quality could not be assessed).</div>',
                unsafe_allow_html=True,
            )

        # Method note
        method_note = (
            "**Population source:** WorldPop 2020 country mosaics aggregated to 0.25° (~28 km) "
            "grid cells (CC BY 4.0). **Method:** population assigned to the nearest grid cell "
            "at each estimated air-quality band; estimates carry ~30 µg/m³ model uncertainty. "
            "Cloud-covered cells are excluded from band totals and reported separately."
        )
        if forecast_prefix:
            method_note += (
                f" **Forecast note:** AOD values are persisted from the most recent satellite "
                f"pass (up to {AOD_LOOKBACK_DAYS} days old); forecast weather from Open-Meteo. "
                f"Forecast uncertainty is substantially higher than same-day estimates."
            )
        st.caption(method_note)


def _render_advisory(advisory: dict, is_forecast: bool = False) -> None:
    """
    Render the Granite (or fallback) health advisory in a styled box.
    advisory = {"text": str, "source": str}  — silently no-ops if empty.
    """
    if not advisory or not advisory.get("text"):
        return

    text   = advisory["text"]
    source = advisory.get("source", "rule-based")
    is_granite = "Granite" in source

    border = "#3b82d4" if is_granite else "#e5e7eb"
    bg     = "#eff6ff" if is_granite else "#f7f8fa"
    label  = (
        "Advisory generated by IBM Granite (running locally via Ollama)"
        if is_granite else
        "Advisory (rule-based)"
    )

    st.markdown(
        f'<div style="background:{bg};border-left:4px solid {border};'
        f'padding:12px 16px;border-radius:6px;margin-top:10px;margin-bottom:4px;">'
        f'<div style="font-size:0.82em;color:#57606a;margin-bottom:6px;">'
        f'{"🤖 " if is_granite else "📋 "}{label}</div>'
        f'<div style="color:#1f2328;line-height:1.6;">{text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


_PLOTLY_CONFIG = {
    "scrollZoom":      True,   # mouse-wheel / trackpad zoom
    "displayModeBar":  True,   # always show the zoom/pan toolbar
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],  # remove irrelevant tools
    "displaylogo":     False,
}


# ---------------------------------------------------------------------------
# UI — page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SkyWatch-AQ  |  Air Quality Estimator",
    page_icon="🌫️",
    layout="wide",
)

st.title("🌫️ SkyWatch-AQ")
st.markdown(
    "**Ground-level PM2.5 estimator for cities without monitoring stations "
    "— validated in South Asia, with regionally varying performance elsewhere.**  \n"
    "Powered by NASA MODIS MAIAC satellite AOD + Open-Meteo weather + a "
    "machine-learning model trained on Delhi and Mumbai air quality data."
)
st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_city, tab_map = st.tabs(["📍 City estimate", "🗺️ Regional map"])

# ============================================================================
# TAB 2 — REGIONAL MAP
# ============================================================================
with tab_map:
    st.subheader("Regional air-quality map")
    st.caption(
        "The model is applied to every ~22 km satellite pixel across the selected "
        "region. Grey gaps are cloud-covered cells with no satellite view — never "
        "interpolated. Resolution ~22 km; uncertainty ~30 ug/m3 RMSE. "
        "Select tomorrow or day-after for a **forecast** map (persistence AOD + forecast weather)."
    )

    map_col1, map_col2 = st.columns([1, 3])
    with map_col1:
        region_options = {v["label"]: k for k, v in REGIONS.items()}
        map_region_label = st.selectbox(
            "Region", list(region_options.keys()), key="map_region"
        )
        map_region_key = region_options[map_region_label]

        today_map     = datetime.date.today()
        yesterday_map = today_map - datetime.timedelta(days=1)
        max_map_date  = today_map + datetime.timedelta(days=2)
        map_date = st.date_input(
            "Date",
            value=yesterday_map,
            min_value=datetime.date(2023, 1, 1),
            max_value=max_map_date,
            key="map_date",
            help=(
                "Past dates: archive satellite + weather.  "
                "Today or future: FORECAST mode — persisted AOD + forecast weather. "
                "First build downloads ~15 MB of satellite data."
            ),
        )
        map_date_str  = map_date.isoformat()
        map_is_forecast = is_forecast_date(map_date_str)

        if map_is_forecast:
            fc_lbl = forecast_label(map_date_str)
            st.info(f"🔮 **FORECAST mode** — {fc_lbl}  \nPersisted AOD + Open-Meteo forecast weather. Higher uncertainty.", icon="🔮")

        run_map = st.button("Build map", type="primary",
                            use_container_width=True, key="run_map")

        st.divider()
        with st.expander("Map limitations"):
            st.markdown("""
**Resolution:** ~22 km per cell (1 MODIS pixel sampled every 24 pixels).

**Land mask:** Only land cells are shown. Ocean, sea, and large inland
water bodies are excluded because the model was trained exclusively on
land-based ground monitors; over-water estimates would be meaningless
extrapolation. Water cells render as map background, same as cloud gaps.

**Cloud gaps:** Land cells where no Terra or Aqua overpass returned
QA-valid AOD are shown as map background (no coloured point). On heavy
cloud days (monsoon) the map may be mostly empty — this is correct.

**Uncertainty:** Model RMSE ~30-36 ug/m3. The map conveys relative
spatial patterns (urban vs. rural gradient) more reliably than absolute
values.

**Validated region:** The model was trained on Delhi and Mumbai. Estimates
are most reliable within ~500 km of those cities. Areas outside may show
systematic bias.

**Temporal:** Each map is a snapshot for one calendar day. The granule
represents a single Terra+Aqua overpass (~10:30 AM local time).

**Population exposure:** WorldPop 2020 country mosaics (CC BY 4.0)
aggregated to 0.25° (~28 km) grid cells. Exposure = population in cells
at each estimated band. Estimates carry ~30 µg/m³ model uncertainty;
cloud-covered areas are reported separately as "no estimate available."

**Forecast mode (future dates):** AOD is persisted from the most recent
available satellite pass (up to 5 days). Forecast weather is from
Open-Meteo. Forecast uncertainty is substantially higher than same-day
estimates — treat as directional guidance only. Forecast cache is valid
for today only and is rebuilt if stale.
            """)

    with map_col2:
        # ---- helper: render forecast badge ----
        def _show_forecast_badge(aod_date: str | None) -> None:
            aod_note = f" (AOD from {aod_date})" if aod_date else ""
            st.markdown(
                f'<div style="display:inline-block;background:#6366f1;color:white;'
                f'padding:3px 12px;border-radius:12px;font-size:0.82em;font-weight:600;'
                f'margin-bottom:8px;">🔮 FORECAST{aod_note}</div>'
                f'<div style="color:#57606a;font-size:0.82em;margin-bottom:8px;">'
                f'Forecast uses latest available satellite AOD (persistence, up to '
                f'{AOD_LOOKBACK_DAYS} days old) + Open-Meteo forecast weather — '
                f'uncertainty is substantially higher than same-day estimates.</div>',
                unsafe_allow_html=True,
            )

        if not run_map:
            if map_is_forecast:
                # For forecast: check today's forecast cache
                fcache = load_forecast_cache(map_region_key, map_date_str)
                if fcache is not None:
                    cached_cells, cached_exposure, cached_advisory = fcache
                    fig = _cells_to_figure(cached_cells, map_region_key, map_date_str)
                    if fig:
                        fc_lbl = forecast_label(map_date_str)
                        st.success(
                            f"Showing cached forecast: {len(cached_cells):,} cells for {map_date_str}.",
                            icon="🔮",
                        )
                        _show_forecast_badge(None)
                        _render_exposure(cached_exposure, forecast_prefix=fc_lbl)
                        _render_advisory(cached_advisory, is_forecast=True)
                        st.plotly_chart(fig, use_container_width=True,
                                        config=_PLOTLY_CONFIG)
                    else:
                        st.info("Forecast data cached but empty — click **Build map**.", icon="🔮")
                else:
                    fc_lbl = forecast_label(map_date_str)
                    st.info(
                        f"🔮 **{fc_lbl} forecast** — click **Build map** to generate.  \n"
                        f"Uses persisted satellite AOD (up to {AOD_LOOKBACK_DAYS} days old) "
                        f"+ Open-Meteo forecast weather.",
                        icon="🔮",
                    )
            else:
                # Past date — check archive cache
                cached_result = load_cached_map(map_region_key, map_date_str)
                if cached_result is not None:
                    cached_cells, cached_exposure, cached_advisory = cached_result
                    fig = _cells_to_figure(cached_cells, map_region_key, map_date_str)
                    if fig:
                        n_cells = len(cached_cells)
                        st.success(
                            f"Showing cached map: {n_cells:,} valid cells for {map_date_str}.",
                            icon="🗺️",
                        )
                        # Re-compute exposure if legacy cache lacks it
                        if not cached_exposure.get("available"):
                            pop_grid = load_population_grid()
                            if pop_grid is not None:
                                bbox = REGIONS[map_region_key].get("bbox")
                                cached_exposure = compute_exposure(
                                    cached_cells, pop_grid, region_bbox=bbox
                                )
                        _render_exposure(cached_exposure)
                        # Generate advisory on-the-fly if not cached yet
                        if not cached_advisory.get("text"):
                            adv_ctx = build_advisory_context(
                                cached_cells, cached_exposure, map_region_key, map_date_str
                            )
                            cached_advisory = generate_advisory(adv_ctx)
                        _render_advisory(cached_advisory)
                        st.plotly_chart(fig, use_container_width=True,
                                        config=_PLOTLY_CONFIG)
                    else:
                        st.info(
                            "No satellite data available for this date "
                            "(complete cloud cover or data gap).",
                            icon="☁️",
                        )
                else:
                    st.info(
                        "Select a region and date, then click **Build map**.  \n"
                        "First render downloads ~15 MB of satellite data and takes ~60 s. "
                        "Subsequent renders for the same date are instant.",
                        icon="🗺️",
                    )
        else:
            # Build map with live progress
            progress_box = st.empty()
            progress_msgs: list[str] = []

            def _progress(msg: str) -> None:
                progress_msgs.append(msg.replace("[map] ", "").replace("[forecast] ", ""))
                progress_box.info(
                    "\n\n".join(f"- {m}" for m in progress_msgs[-5:]),
                    icon="🛰️",
                )

            bundle = load_model()

            if map_is_forecast:
                # --- FORECAST PATH ---
                with st.spinner(f"Building forecast map for {map_date_str}..."):
                    # Check today's cache first
                    fcache = load_forecast_cache(map_region_key, map_date_str)
                    if fcache is not None:
                        cells, exposure, advisory = fcache
                        aod_date_used = None
                    else:
                        result_fc = build_forecast_map(
                            map_region_key, map_date_str, bundle, progress_cb=_progress
                        )
                        cells, exposure, aod_date_used, advisory = result_fc

                progress_box.empty()

                if not cells:
                    st.warning(
                        "No persisted AOD found within the last 5 days — cannot build forecast. "
                        "Try a different date or check satellite availability.",
                        icon="🔮",
                    )
                else:
                    fc_lbl = forecast_label(map_date_str)
                    fig = _cells_to_figure(cells, map_region_key, map_date_str)
                    st.success(
                        f"Forecast map: {len(cells):,} cells with persisted AOD for {map_date_str}.",
                        icon="🔮",
                    )
                    _show_forecast_badge(aod_date_used)
                    _render_exposure(exposure, forecast_prefix=fc_lbl)
                    _render_advisory(advisory, is_forecast=True)
                    st.plotly_chart(fig, use_container_width=True,
                                    config=_PLOTLY_CONFIG)
            else:
                # --- ARCHIVE PATH ---
                with st.spinner("Building regional map — downloading satellite granule..."):
                    cached_result = load_cached_map(map_region_key, map_date_str)
                    if cached_result is not None:
                        cells, exposure, advisory = cached_result
                        # Re-compute exposure if legacy cache lacks it
                        if not exposure.get("available"):
                            pop_grid = load_population_grid()
                            if pop_grid is not None:
                                bbox = REGIONS[map_region_key].get("bbox")
                                exposure = compute_exposure(cells, pop_grid, region_bbox=bbox)
                    else:
                        result = build_region_map(
                            map_region_key, map_date_str, bundle, progress_cb=_progress
                        )
                        if result:
                            cells, exposure, advisory = result
                        else:
                            cells, exposure, advisory = [], {"available": False}, {}

                progress_box.empty()

                if not cells:
                    st.warning(
                        "No valid satellite data for this region and date. "
                        "The entire tile may be under cloud cover. Try a different date.",
                        icon="☁️",
                    )
                else:
                    fig = _cells_to_figure(cells, map_region_key, map_date_str)
                    st.success(
                        f"Map built: {len(cells):,} valid cells (of 2,500 possible).",
                        icon="🗺️",
                    )
                    _render_exposure(exposure)
                    _render_advisory(advisory)
                    st.plotly_chart(fig, use_container_width=True,
                                    config=_PLOTLY_CONFIG)

# ============================================================================
# TAB 1 — SINGLE-CITY ESTIMATE
# ============================================================================
with tab_city:

# ---------------------------------------------------------------------------
# Sidebar — city estimate inputs
# ---------------------------------------------------------------------------
    with st.sidebar:
        st.header("City estimate inputs")

        city_name = st.selectbox("City", list(CITIES.keys()), index=0)
        lat_default, lon_default, aod_default, openaq_coords = CITIES[city_name]

        if city_name == "Custom location":
            lat = st.number_input("Latitude",  value=28.6, min_value=-90.0,  max_value=90.0,  step=0.01)
            lon = st.number_input("Longitude", value=77.2, min_value=-180.0, max_value=180.0, step=0.01)
        else:
            lat, lon = lat_default, lon_default
            st.caption(f"Lat {lat:.4f}  ·  Lon {lon:.4f}")

        yesterday = datetime.date.today() - datetime.timedelta(days=1)
        max_city_date = datetime.date.today() + datetime.timedelta(days=2)
        selected_date = st.date_input(
            "Date",
            value=yesterday,
            min_value=datetime.date(2023, 1, 1),
            max_value=max_city_date,
            help=(
                "Archive dates (up to yesterday): satellite AOD + historical weather.  "
                "Today / tomorrow / day-after: FORECAST mode — persisted AOD + forecast weather."
            ),
        )
        city_is_forecast = is_forecast_date(selected_date.isoformat())
        if city_is_forecast:
            st.info(
                f"🔮 **FORECAST** — {forecast_label(selected_date.isoformat())}  \n"
                "Uses most recent satellite AOD + Open-Meteo forecast weather.",
                icon="🔮",
            )

        st.divider()
        run = st.button("Estimate PM2.5", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Landing state (before first run)
# ---------------------------------------------------------------------------
    if not run:
        st.info(
            "Select a city and date in the sidebar, then click **Estimate PM2.5**.  \n"
            "Satellite AOD and weather are fetched automatically.",
            icon="ℹ️",
        )
        with st.expander("How it works"):
            st.markdown("""
**Data pipeline:**
1. **Satellite AOD** — NASA MODIS MAIAC (MCD19A2 v061) aerosol optical depth.
   The app searches the most recent granule within a 7-day lookback of the
   selected date, downloads it, extracts the QA-filtered AOD at the city
   centre pixel (Terra + Aqua overpasses averaged), then deletes the file.
2. **Weather** — Open-Meteo 6-variable daily archive (temperature, humidity,
   wind, precipitation, pressure, shortwave radiation).
3. **ML model** — RandomForest regressor (log1p target) trained on 6,902
   matched AOD + PM2.5 + weather records from Delhi and Mumbai (2023–2025),
   validated with 5-fold date-grouped cross-validation.
4. **Reality check** — for cities with known OpenAQ monitors (Delhi, Mumbai,
   Karachi), the actual monitor average for that date is fetched and shown
   alongside the estimate — for reference only, not used in the prediction.

**Regional map mode** (tab 2): The model is applied across a full MODIS
tile (~10 × 10 degrees) at ~22 km resolution — every grid cell where the
satellite obtained a cloud-free AOD reading gets a PM2.5 estimate. This
paints air quality across thousands of locations that have no ground monitor.
Limitations: resolution ~22 km (downsampled from 926 m native pixels),
cloud gaps are shown honestly as background, and model uncertainty (~30
ug/m3 RMSE) is the same as the single-city mode. Use the map to understand
spatial patterns, not precise values.

**Health advisory layer:** For each regional map, a plain-language public
health advisory is generated by **IBM Granite** (`ibm/granite-3-8b-instruct`)
via watsonx.ai — summarising conditions, at-risk groups, and 2–3 concrete
recommendations in 3–5 sentences. When watsonx credentials are not
configured, a rule-based template produces an equivalent advisory. All
advisories contain the phrase "estimates, not measurements" and are clearly
labelled with their source.

**Model performance:**
| Metric | Delhi+Mumbai CV | Karachi zero-shot |
|--------|----------------|-------------------|
| R2 | 0.45 +/- 0.07 | 0.18 |
| RMSE | 35.6 ug/m3 | 31.2 ug/m3 |
| MAE | 21.3 ug/m3 | 22.8 ug/m3 |

**Note:** Estimates carry ~30 ug/m3 uncertainty. Not for regulatory use.
            """)
        st.stop()

    # ---------------------------------------------------------------------------
    # Run — fetch all data in parallel display order
    # ---------------------------------------------------------------------------
    date_str = selected_date.isoformat()

    # --- 1. Satellite AOD (persistence for forecast; archive search for past) ---
    aod_val     = aod_default     # fallback value
    aod_source  = "default"       # "satellite" | "manual" | "default"
    aod_date    = None

    aod_status = st.empty()

    if city_is_forecast:
        # Forecast: search for most recent past granule (lookback from yesterday)
        lookback_anchor = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
        aod_status.info("🛰️ Searching for most recent satellite AOD (persistence)...", icon="🛰️")
        auto_aod, auto_date = fetch_aod_auto(lat, lon, lookback_anchor)
    else:
        aod_status.info("🛰️ Searching NASA satellite archive for AOD...", icon="🛰️")
        auto_aod, auto_date = fetch_aod_auto(lat, lon, date_str)

    if auto_aod is not None:
        aod_val    = auto_aod
        aod_date   = auto_date
        aod_source = "satellite"
        persistence_note = f" · persistence from {aod_date}" if city_is_forecast else ""
        aod_status.success(
            f"**Satellite AOD: {aod_val:.3f}**  (from {aod_date} pass, "
            f"MODIS MAIAC MCD19A2, QA-filtered{persistence_note})",
            icon="🛰️",
        )
    else:
        aod_status.warning(
            "No valid satellite AOD found within 7 days "
            "(cloud cover or data gap). Using city historical average as fallback.",
            icon="☁️",
        )

    # Allow manual override in an expander
    with st.expander("Advanced: enter AOD manually", expanded=(aod_source != "satellite")):
        manual_aod = st.number_input(
            "MODIS MAIAC AOD at 0.55 µm",
            min_value=0.0, max_value=6.0,
            value=float(aod_val),
            step=0.01,
            key="manual_aod",
            help=(
                "Aerosol Optical Depth from MODIS MAIAC. "
                "Clear: 0.1–0.4 | Polluted: 0.5–1.5 | Severe: >1.5. "
                "If set here, this overrides the satellite auto-fetch."
            ),
        )
        use_manual = st.checkbox("Use this value instead of satellite AOD", value=(aod_source != "satellite"))
        if use_manual:
            aod_val    = manual_aod
            aod_source = "manual"

    # --- 2. Weather (forecast API for future dates, archive for past) ---
    with st.spinner(f"Fetching {'forecast' if city_is_forecast else 'archive'} weather for {city_name} on {date_str}..."):
        if city_is_forecast:
            weather = fetch_forecast_weather_point(lat, lon, date_str)
        else:
            weather = fetch_weather(lat, lon, date_str)
    weather_ok = weather is not None and any(v is not None for v in weather.values())

    # --- 3. Predict ---
    try:
        bundle = load_model()
        pm25   = predict(bundle, aod_val, selected_date, weather or {})
        label, color, advice = categorise(pm25)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # ---------------------------------------------------------------------------
    # Results header
    # ---------------------------------------------------------------------------
    fc_result_lbl = forecast_label(date_str) if city_is_forecast else ""
    title_suffix  = f" — {fc_result_lbl} Forecast" if city_is_forecast else ""
    st.subheader(f"Results — {city_name}, {date_str}{title_suffix}")

    # Forecast badge
    if city_is_forecast:
        st.markdown(
            f'<div style="display:inline-block;background:#6366f1;color:white;'
            f'padding:3px 12px;border-radius:12px;font-size:0.82em;font-weight:600;'
            f'margin-bottom:6px;">🔮 FORECAST — {fc_result_lbl}</div>'
            f'<div style="color:#57606a;font-size:0.82em;margin-bottom:8px;">'
            f'AOD persisted from most recent satellite pass · forecast weather · '
            f'higher uncertainty than archive estimates</div>',
            unsafe_allow_html=True,
        )

    # Region confidence badge
    is_validated, conf_text = region_confidence(lat, lon)
    conf_color = "#3b82d4" if is_validated else "#f97316"
    conf_bg    = "#eff6ff" if is_validated else "#fff7ed"
    st.markdown(
        f'<span style="background:{conf_bg};color:{conf_color};border:1px solid {conf_color};'
        f'padding:3px 10px;border-radius:12px;font-size:0.82em;font-weight:500;">'
        f"{'✓' if is_validated else '⚠'} {conf_text}</span>",
        unsafe_allow_html=True,
    )
    st.write("")   # spacer

    col1, col2 = st.columns([2, 3])
    with col1:
        st.metric("SkyWatch PM2.5 estimate", f"{pm25:.1f} µg/m3")
    with col2:
        txt_color = "#1f2328" if color in ("#22c55e", "#84cc16", "#eab308") else "white"
        st.markdown(
            f'<span style="background:{color};color:{txt_color};padding:8px 18px;'
            f'border-radius:8px;font-weight:600;font-size:1.1em;display:inline-block;">'
            f"{label}</span>",
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="border-left:4px solid {color};padding:10px 16px;'
        f'background:#f7f8fa;border-radius:4px;margin-top:12px;color:#1f2328;">'
        f"<strong>Health advisory:</strong> {advice}</div>",
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------------------
    # REALITY CHECK PANEL  (Part 2) — skipped for forecast dates
    # ---------------------------------------------------------------------------
    st.divider()
    st.subheader("Reality check")

    if city_is_forecast:
        st.info(
            "Ground-monitor data is not available for future dates — "
            "this is a forecast estimate only.  Check back after the date passes.",
            icon="🔮",
        )
    elif openaq_coords is not None:
        with st.spinner("Fetching ground monitor data from OpenAQ..."):
            ground_pm25 = fetch_ground_truth(openaq_coords, date_str)

        rc_col1, rc_col2 = st.columns(2)
        with rc_col1:
            st.metric(
                label="SkyWatch estimate",
                value=f"{pm25:.1f} µg/m³",
                help="ML model prediction using satellite AOD + weather.",
            )
        with rc_col2:
            if ground_pm25 is not None:
                delta_val = round(pm25 - ground_pm25, 1)
                st.metric(
                    label="Ground monitors measured",
                    value=f"{ground_pm25:.1f} µg/m³",
                    delta=f"{delta_val:+.1f} µg/m³ model bias",
                    delta_color="inverse",
                    help="Average across all OpenAQ PM2.5 sensors near this city on the selected date.",
                )
            else:
                st.metric(label="Ground monitors measured", value="No data for this date")
                st.caption(
                    "OpenAQ has no readings for this city on the selected date. "
                    "Try an earlier date."
                )
    else:
        st.info(
            "**No ground monitors available here** — this is exactly the situation "
            "SkyWatch is built for.  \n"
            "There are no OpenAQ PM2.5 stations near this location in our dataset, "
            "so the satellite + ML estimate is the best available indicator.",
            icon="📡",
        )

    # ---------------------------------------------------------------------------
    # Weather panel
    # ---------------------------------------------------------------------------
    st.divider()
    if weather_ok:
        wx_lbl = "Forecast weather (Open-Meteo)" if city_is_forecast else "Weather inputs (Open-Meteo)"
        st.subheader(wx_lbl)
        W_LABELS = {
            "temperature_2m_mean":        ("Temperature",      "°C"),
            "relative_humidity_2m_mean":  ("Relative Humidity","%"),
            "wind_speed_10m_mean":        ("Wind Speed",       "km/h"),
            "precipitation_sum":          ("Precipitation",    "mm"),
            "surface_pressure_mean":      ("Surface Pressure", "hPa"),
            "shortwave_radiation_sum":    ("Solar Radiation",  "MJ/m²"),
        }
        cols = st.columns(3)
        for idx, (key, (lbl, unit)) in enumerate(W_LABELS.items()):
            v = weather.get(key)
            cols[idx % 3].metric(lbl, f"{v:.1f} {unit}" if v is not None else "N/A")
    else:
        st.warning("Could not fetch weather from Open-Meteo. Defaults used — estimate may be less accurate.", icon="⚠️")

    # ---------------------------------------------------------------------------
    # Inputs summary (collapsible)
    # ---------------------------------------------------------------------------
    with st.expander("Inputs summary"):
        aod_src_label = {
            "satellite": (
                f"Satellite persistence (AOD date: {aod_date}, forecast mode)"
                if city_is_forecast else
                f"Satellite auto-fetch (pass date: {aod_date})"
            ),
            "manual":    "Manual override",
            "default":   "City historical average (fallback)",
        }[aod_source]
        st.markdown(f"""
    | Input | Value |
    |-------|-------|
    | City | {city_name} |
    | Latitude | {lat:.4f} |
    | Longitude | {lon:.4f} |
    | Date | {date_str} |
    | AOD value | {aod_val:.3f} |
    | AOD source | {aod_src_label} |
    | Season | {season(selected_date.month).capitalize()} |
    | Day of year | {selected_date.timetuple().tm_yday} |
        """)

    # ---------------------------------------------------------------------------
    # Disclaimer
    # ---------------------------------------------------------------------------
    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** These are ML-model estimates, not measurements. "
        "Model RMSE ≈ 30–36 µg/m³. Do not use for regulatory or medical decisions.  \n"
        "Data sources: NASA MODIS MAIAC (MCD19A2 v061), OpenAQ v3, Open-Meteo archive API."
    )
