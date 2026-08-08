"""
app.py — skywatch-aq Streamlit health-alert app
------------------------------------------------
Predicts ground-level PM2.5 from:
  • Satellite AOD (user-entered or default for the city)
  • Weather (auto-fetched from Open-Meteo for the selected date)
  • Date features derived from the chosen date

Uses models/pm25_model.pkl (RandomForest, log1p target, trained on
Delhi + Mumbai + Karachi-calibration slice).
"""

import math
import datetime
import pathlib
import sys

import requests
import numpy as np
import joblib
import streamlit as st

# ---------------------------------------------------------------------------
# Paths (resolve relative to repo root regardless of cwd)
# ---------------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "pm25_model.pkl"

# ---------------------------------------------------------------------------
# Known cities  {name: (lat, lon, typical_aod)}
# ---------------------------------------------------------------------------
CITIES = {
    "Delhi, India":       (28.6139, 77.2090, 0.65),
    "Mumbai, India":      (19.0760, 72.8777, 0.52),
    "Karachi, Pakistan":  (24.8607, 67.0011, 0.45),
    "Lahore, Pakistan":   (31.5497, 74.3436, 0.70),
    "Dhaka, Bangladesh":  (23.8103, 90.4125, 0.60),
    "Kolkata, India":     (22.5726, 88.3639, 0.58),
    "Dhanbad, India":     (23.7957, 86.4304, 0.55),
    "Custom location":    (None, None, 0.50),
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
# AQI / health categories  (PM2.5 µg/m³, WHO & India NAAQS informed)
# ---------------------------------------------------------------------------
CATEGORIES = [
    (  0,  12,  "Good",               "#22c55e", "Air quality is satisfactory. "
                                                  "No health risk for the general population."),
    ( 12,  35,  "Moderate",           "#84cc16", "Air quality is acceptable. "
                                                  "Sensitive individuals may experience minor symptoms."),
    ( 35,  55,  "Unhealthy for\nSensitive Groups", "#eab308",
                                                 "Children, elderly and those with respiratory or heart "
                                                  "conditions should limit prolonged outdoor exertion."),
    ( 55, 150,  "Unhealthy",          "#f97316", "Everyone may begin to experience health effects. "
                                                  "Reduce outdoor activities, especially strenuous exercise."),
    (150, 250,  "Very Unhealthy",     "#ef4444", "Health alert: everyone should avoid prolonged outdoor "
                                                  "exertion. Sensitive groups must stay indoors."),
    (250, 9999, "Hazardous",          "#7c3aed", "Health emergency. Entire population is likely to be "
                                                  "affected. Stay indoors with windows closed."),
]

def categorise(pm25: float) -> tuple[str, str, str]:
    """Return (label, hex_color, advice)."""
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

OM_VAR_MAP = {
    "temperature_2m_mean":        "temperature_2m_mean",
    "relative_humidity_2m_mean":  "relative_humidity_2m_mean",
    "wind_speed_10m_mean":        "wind_speed_10m_mean",
    "precipitation_sum":          "precipitation_sum",
    "surface_pressure_mean":      "surface_pressure_mean",
    "shortwave_radiation_sum":    "shortwave_radiation_sum",
}

@st.cache_data(ttl=3600)
def fetch_weather(lat: float, lon: float, date_str: str) -> dict | None:
    """
    Fetch daily weather from Open-Meteo archive API for a single day.
    Returns dict {var: value} or None on failure.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date":   date_str,
        "daily": ",".join(OM_VAR_MAP.values()),
        "timezone": "auto",
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        daily = data.get("daily", {})
        result: dict = {}
        for feat, om_key in OM_VAR_MAP.items():
            vals = daily.get(om_key, [None])
            result[feat] = vals[0] if vals else None
        return result
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict(bundle: dict, aod: float, date: datetime.date, weather: dict) -> float:
    """Return predicted PM2.5 (µg/m³)."""
    m = date.month
    row = [
        aod,
        m,
        date.timetuple().tm_yday,
        date.weekday(),
        SEASON_ENCODE[season(m)],
        weather.get("temperature_2m_mean") or 25.0,
        weather.get("relative_humidity_2m_mean") or 50.0,
        weather.get("wind_speed_10m_mean") or 3.0,
        weather.get("precipitation_sum") or 0.0,
        weather.get("surface_pressure_mean") or 1010.0,
        weather.get("shortwave_radiation_sum") or 15.0,
    ]
    X = np.array([row], dtype=np.float64)
    y_log = bundle["model"].predict(X)[0]
    return float(max(0.0, math.expm1(y_log)))

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SkyWatch-AQ  |  Air Quality Estimator",
    page_icon="🌫️",
    layout="centered",
)

st.title("🌫️ SkyWatch-AQ")
st.markdown(
    "**Ground-level PM2.5 estimator for cities without monitoring stations.**  \n"
    "Powered by NASA MODIS MAIAC satellite AOD + Open-Meteo weather + a "
    "machine-learning model trained on Delhi and Mumbai air quality data."
)
st.divider()

# ---- Sidebar inputs ----
with st.sidebar:
    st.header("Inputs")

    city_name = st.selectbox("City", list(CITIES.keys()), index=0)
    lat_default, lon_default, aod_default = CITIES[city_name]

    if city_name == "Custom location":
        lat = st.number_input("Latitude",  value=28.6, min_value=-90.0,  max_value=90.0,  step=0.01)
        lon = st.number_input("Longitude", value=77.2, min_value=-180.0, max_value=180.0, step=0.01)
    else:
        lat, lon = lat_default, lon_default
        st.caption(f"Lat {lat:.4f}, Lon {lon:.4f}")

    # Date: default yesterday so weather archive is available
    yesterday = datetime.date.today() - datetime.timedelta(days=1)
    selected_date = st.date_input(
        "Date",
        value=yesterday,
        min_value=datetime.date(2023, 1, 1),
        max_value=yesterday,
        help="Open-Meteo archive is available up to yesterday.",
    )

    st.subheader("Satellite AOD")
    aod_val = st.number_input(
        "MODIS MAIAC AOD at 0.55 µm",
        min_value=0.0, max_value=6.0,
        value=aod_default,
        step=0.01,
        help=(
            "Aerosol Optical Depth from the NASA MODIS MAIAC product. "
            "Typical clear-day values: 0.1–0.4 | polluted: 0.5–1.5 | severe: >1.5. "
            "Default is the historical city average from our dataset."
        ),
    )

    st.divider()
    run = st.button("Estimate PM2.5", type="primary", use_container_width=True)

# ---- Main panel ----
if not run:
    st.info(
        "Select a city and date in the sidebar, then click **Estimate PM2.5**.  \n"
        "Weather data is fetched automatically from Open-Meteo.",
        icon="ℹ️"
    )

    with st.expander("How it works"):
        st.markdown("""
**Data pipeline:**
1. **Satellite AOD** — NASA MODIS MAIAC (MCD19A2) aerosol optical depth, a measure
   of how much light is scattered/absorbed by atmospheric particles.
2. **Ground PM2.5** — OpenAQ v3 daily averages from government monitoring stations
   in Delhi (51 stations), Mumbai (36 stations), and Karachi (27 stations).
3. **Weather** — Open-Meteo 6-variable daily climate data (temperature, humidity,
   wind, precipitation, pressure, shortwave radiation).
4. **ML model** — RandomForest regressor trained on matched AOD + PM2.5 + weather
   records (6,902 training rows, 2023–2025). Target: log₁p(PM2.5), evaluated with
   5-fold date-grouped cross-validation to prevent data leakage.

**Model performance (held-out cross-validation):**
| Metric | Delhi + Mumbai CV | Karachi zero-shot |
|--------|-------------------|-------------------|
| R² | 0.45 ± 0.07 | 0.18 |
| RMSE | 35.6 µg/m³ | 31.2 µg/m³ |
| MAE | 21.3 µg/m³ | 22.8 µg/m³ |

**Important:** PM2.5 estimates carry uncertainty (~±30 µg/m³ RMSE).
Always consult official monitoring data for health decisions.
        """)
    st.stop()

# ---- Fetch weather ----
date_str = selected_date.isoformat()

with st.spinner(f"Fetching weather for {city_name} on {date_str}..."):
    weather = fetch_weather(lat, lon, date_str)

weather_ok = weather is not None and any(v is not None for v in weather.values())

# ---- Run prediction ----
try:
    bundle = load_model()
    pm25 = predict(bundle, aod_val, selected_date, weather or {})
    label, color, advice = categorise(pm25)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# ---- Results ----
st.subheader(f"PM2.5 estimate — {city_name}, {date_str}")

# Big metric + colour badge
col1, col2 = st.columns([2, 3])
with col1:
    st.metric("Estimated PM2.5", f"{pm25:.1f} µg/m³")
with col2:
    badge_style = (
        f"background:{color};color:{'#1f2328' if color in ('#22c55e','#84cc16','#eab308') else 'white'};"
        "padding:8px 18px;border-radius:8px;font-weight:600;font-size:1.1em;display:inline-block;"
    )
    label_clean = label.replace("\n", " ")
    st.markdown(f'<span style="{badge_style}">{label_clean}</span>', unsafe_allow_html=True)

# Health advice box
st.markdown(
    f"""<div style="border-left:4px solid {color};padding:10px 16px;
    background:#f7f8fa;border-radius:4px;margin-top:12px;color:#1f2328;">
    <strong>Health advisory:</strong> {advice}
    </div>""",
    unsafe_allow_html=True,
)

st.divider()

# ---- Weather used ----
if weather_ok:
    st.subheader("Weather data used (Open-Meteo)")
    wlabels = {
        "temperature_2m_mean":       ("Temperature",      "°C"),
        "relative_humidity_2m_mean": ("Relative Humidity","%"),
        "wind_speed_10m_mean":       ("Wind Speed",       "km/h"),
        "precipitation_sum":         ("Precipitation",    "mm"),
        "surface_pressure_mean":     ("Surface Pressure", "hPa"),
        "shortwave_radiation_sum":   ("Solar Radiation",  "MJ/m²"),
    }
    cols = st.columns(3)
    for idx, (key, (label_w, unit)) in enumerate(wlabels.items()):
        val = weather.get(key)
        display = f"{val:.1f} {unit}" if val is not None else "N/A"
        cols[idx % 3].metric(label_w, display)
else:
    st.warning(
        "Could not fetch weather from Open-Meteo. "
        "Default values were used — estimate may be less accurate.",
        icon="⚠️"
    )

# ---- Inputs summary ----
with st.expander("Inputs summary"):
    st.markdown(f"""
| Input | Value |
|-------|-------|
| City | {city_name} |
| Latitude | {lat:.4f} |
| Longitude | {lon:.4f} |
| Date | {date_str} |
| MODIS MAIAC AOD | {aod_val:.3f} |
| Season | {season(selected_date.month).capitalize()} |
| Day of year | {selected_date.timetuple().tm_yday} |
    """)

# ---- Disclaimer ----
st.divider()
st.caption(
    "⚠️ **Disclaimer:** These are ML-model estimates, not measurements. "
    "Model RMSE ≈ 30–36 µg/m³. Do not use for regulatory or medical decisions.  \n"
    "Data sources: NASA MODIS MAIAC (MCD19A2), OpenAQ v3, Open-Meteo archive API."
)
