"""
app.py — skywatch-aq Streamlit health-alert app
------------------------------------------------
v2 — three upgrades over v1:
  1. AUTO-FETCH SATELLITE AOD  — earthaccess MCD19A2 7-day lookback,
     QA-filtered, granule deleted after extraction, Streamlit-cached.
     Falls back gracefully to manual input on failure.
  2. REALITY CHECK PANEL — for cities with OpenAQ monitors, fetches
     that date's ground-truth PM2.5 and shows it alongside our estimate.
     Display-only; never feeds the prediction.
  3. DEPLOYABILITY — model committed; requirements pinned for Cloud.

Prediction uses models/pm25_model.pkl (RandomForest, log1p target,
trained on Delhi + Mumbai + Karachi-calibration slice).
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

# Add src/ to path for aod_utils
import sys
sys.path.insert(0, str(_REPO_ROOT / "src"))
from aod_utils import (
    latlon_to_tile_pixel, tile_id,
    earthaccess_login, find_granule_for_date,
    download_granule, extract_aod_at_point,
)

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
# UI — page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SkyWatch-AQ  |  Air Quality Estimator",
    page_icon="🌫️",
    layout="centered",
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
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Inputs")

    city_name = st.selectbox("City", list(CITIES.keys()), index=0)
    lat_default, lon_default, aod_default, openaq_coords = CITIES[city_name]

    if city_name == "Custom location":
        lat = st.number_input("Latitude",  value=28.6, min_value=-90.0,  max_value=90.0,  step=0.01)
        lon = st.number_input("Longitude", value=77.2, min_value=-180.0, max_value=180.0, step=0.01)
    else:
        lat, lon = lat_default, lon_default
        st.caption(f"Lat {lat:.4f}  ·  Lon {lon:.4f}")

    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    selected_date = st.date_input(
        "Date",
        value=yesterday,
        min_value=datetime.date(2023, 1, 1),
        max_value=yesterday,
        help="Open-Meteo archive is available up to yesterday.",
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
3. **ML model** — RandomForest regressor (log₁p target) trained on 6,902
   matched AOD + PM2.5 + weather records from Delhi and Mumbai (2023–2025),
   validated with 5-fold date-grouped cross-validation.
4. **Reality check** — for cities with known OpenAQ monitors (Delhi, Mumbai,
   Karachi), the actual monitor average for that date is fetched and shown
   alongside the estimate — for reference only, not used in the prediction.

**Model performance:**
| Metric | Delhi+Mumbai CV | Karachi zero-shot |
|--------|----------------|-------------------|
| R² | 0.45 ± 0.07 | 0.18 |
| RMSE | 35.6 µg/m³ | 31.2 µg/m³ |
| MAE | 21.3 µg/m³ | 22.8 µg/m³ |

**Note:** Estimates carry ~±30 µg/m³ uncertainty. Not for regulatory use.
        """)
    st.stop()

# ---------------------------------------------------------------------------
# Run — fetch all data in parallel display order
# ---------------------------------------------------------------------------
date_str = selected_date.isoformat()

# --- 1. Auto-fetch satellite AOD ---
aod_val     = aod_default     # fallback value
aod_source  = "default"       # "satellite" | "manual" | "default"
aod_date    = None

aod_status = st.empty()
aod_status.info("🛰️ Searching NASA satellite archive for AOD...", icon="🛰️")

auto_aod, auto_date = fetch_aod_auto(lat, lon, date_str)

if auto_aod is not None:
    aod_val    = auto_aod
    aod_date   = auto_date
    aod_source = "satellite"
    aod_status.success(
        f"**Satellite AOD: {aod_val:.3f}**  (from {aod_date} pass, "
        f"MODIS MAIAC MCD19A2, QA-filtered)",
        icon="🛰️",
    )
else:
    aod_status.warning(
        "No valid satellite AOD found within 7 days of the selected date "
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

# --- 2. Weather ---
with st.spinner(f"Fetching weather for {city_name} on {date_str}..."):
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
st.subheader(f"Results — {city_name}, {date_str}")

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
# REALITY CHECK PANEL  (Part 2)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Reality check")

if openaq_coords is not None:
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
    st.subheader("Weather inputs (Open-Meteo)")
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
        "satellite": f"Satellite auto-fetch (pass date: {aod_date})",
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
