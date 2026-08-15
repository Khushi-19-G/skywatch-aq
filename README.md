# SkyWatch-AQ 🌫️

**Ground-level air quality estimates for cities without monitoring stations — from NASA satellite data, machine learning, and open weather.**

**Live app:** https://skywatch-aq.streamlit.app
**Built for the IBM AI Builders Challenge (August 2026) with IBM Bob as the primary development tool.**

---

## Problem statement

Air pollution (PM2.5) is one of the largest environmental health risks in the world, associated with roughly seven million deaths per year. The most basic protection is information: knowing when the air is dangerous so that people can limit exposure. That information requires ground monitoring stations, and most of the world does not have them. Delhi operates dozens of official monitors; most cities in South Asia and Africa operate a handful or none. The situation worsened in 2025 when the US State Department paused public data sharing from its embassy air monitors, which had been the only reliable reference in many cities. Billions of people have no way to know what they are breathing.

## Solution description

NASA satellites measure aerosol optical depth (AOD) — the haziness of the atmospheric column — over every city on Earth, every day, for free. AOD is not the same as street-level PM2.5, but the relationship between them can be learned from cities that have both satellite coverage and ground monitors.

SkyWatch-AQ trains a machine-learning model on cities with dense monitoring (Delhi and Mumbai) and applies it where monitoring is sparse or absent. The public web app provides:

- **City estimate:** select a city and date; the app fetches the most recent MODIS satellite pass and the day's weather automatically and returns an estimated PM2.5 with a color-coded health band and plain-language advice. A "Reality check" panel displays the estimate next to actual monitor readings where monitors exist.
- **Regional map:** the model applied to every ~22 km satellite pixel across a region — including the many locations that have never had a ground monitor. Cloud-covered cells are shown as gaps, never interpolated. Ocean cells are excluded via a land mask.
- **Population exposure:** open gridded population data (WorldPop) is overlaid on the map to compute headline statistics such as "~37.5 M people estimated in Very Unhealthy+ air today," with unassessed (cloud-covered) populations reported separately.
- **Forecast mode:** tomorrow's forecast weather combined with the latest available satellite pass (persistence assumption, labeled) produces a next-day estimate — a warning rather than a report.
- **AI health advisories:** an IBM Granite language model turns the day's numbers into a short plain-language public-health advisory (who is at risk, what to do), with a rule-based fallback when no model backend is available. Every advisory includes the phrase "estimates, not measurements."

## AI approach and architecture

**Data (2023-01 to 2025-07):**
- Ground truth: ~15,000 daily PM2.5 readings from monitors in Delhi, Mumbai, and Karachi via the OpenAQ v3 API.
- Satellite: NASA MODIS MAIAC daily AOD (MCD19A2 v061) for tiles h24v06 and h24v07, QA-filtered, Terra and Aqua overpasses averaged, accessed through NASA Earthdata (earthaccess).
- Weather: six daily variables (temperature, humidity, wind, precipitation, pressure, shortwave radiation) from the Open-Meteo archive API.
- Joined on (date, location): 7,267 matched training rows across three cities.

**Model:** RandomForest regressor on a log1p-transformed PM2.5 target, with AOD, weather, and calendar features. Evaluated with 5-fold date-grouped cross-validation (all rows sharing a date stay in one fold) to prevent leakage, always against a naive-mean baseline.

**Transfer design:** the model trains on Delhi and Mumbai only; Karachi is held out entirely as an unseen-city test. The first single-city model (Delhi only) failed on coastal Karachi (negative R²). Adding coastal Mumbai to training raised Karachi zero-shot performance from R² −1.33 to +0.18 — evidence that cross-city transfer works when the training portfolio covers the target's climate regime.

**Performance:**

| Metric | Delhi+Mumbai (CV) | Karachi (zero-shot) |
|---|---|---|
| R² | 0.45 ± 0.07 | 0.18 |
| RMSE | 35.6 µg/m³ | 31.2 µg/m³ |
| MAE | 21.3 µg/m³ | 22.8 µg/m³ |
| Exact health band | 58.6% (naive 48.5%) | 50.7% (naive 34.8%) |
| Within one band | 93.8% (naive 78.3%) | 88.8% (naive 59.7%) |

The model is most accurate in the "Unhealthy" band (80–87% exact) — the band where warnings matter most.

**Verification:** an independent script (`src/verify_data.py`-style, in `scripts/`) re-fetched randomly sampled training rows from the original sources; all PM2.5 values matched exactly and AOD matched at pixel level, with residual differences explained by spatial averaging.

## Selected theme

August Challenge — **satellite data analysis platforms** and **tools that translate complex space data into clear insights**. SkyWatch-AQ makes NASA Earth-observation data usable and accessible for public health: raw HDF granules in, plain-language health guidance out.

## How IBM Bob was used

IBM Bob was the primary development tool for the entire codebase across the data pipeline, model training, and application:

- Wrote the OpenAQ, earthaccess/MODIS, and Open-Meteo fetchers, including API probing before implementation.
- Diagnosed and fixed real bugs: a capped CMR search that silently discarded ~80% of satellite granules (fixing it grew the dataset 5×), an invisible UTF-8 BOM breaking credential loading, Windows/Linux path differences, and a cloud-only HDF4 reader failure solved with a dual netCDF4/pyhdf reader verified pixel-identical across both backends.
- Built the Streamlit app, the regional map pipeline, the population-exposure layer, forecast mode, and the four-backend advisory chain (watsonx → local Ollama Granite → Hugging Face → rule-based).
- IBM Granite (granite3.3, running locally via Ollama) generates the health advisories in development; the deployed app serves cached Granite-written advisories for demo dates and correctly labeled rule-based advisories otherwise.

## Limitations (stated honestly)

- Estimates carry ~±30 µg/m³ RMSE and are not measurements; the app says so on every screen. Not for regulatory or medical decisions.
- Clouds blind the satellite: coverage averages ~50% of location-days and drops sharply in monsoon; unassessed areas are reported, never interpolated.
- The model has almost no clean-air training days (Delhi/Mumbai rarely provide them), so the "Good" band is underrepresented.
- Validation exists for South Asia only; the app labels locations outside a 500 km radius of the training/validation cities as lower confidence.
- Karachi's zero-shot R² (0.18) is meaningfully better than baseline but far from monitor-grade — the roadmap is a larger, climate-diverse training portfolio.

## Data sources and licenses

NASA MODIS MAIAC MCD19A2 v061 (NASA Earthdata, free and open) · OpenAQ v3 (open air-quality data) · Open-Meteo (free non-commercial, attribution) · WorldPop 2020 (CC BY 4.0) · IBM Granite (Apache 2.0).

## Run it yourself

```
git clone https://github.com/Khushi-19-G/skywatch-aq.git
cd skywatch-aq
pip install -r requirements.txt
# .env: EARTHDATA_USERNAME, EARTHDATA_PASSWORD, OPENAQ_API_KEY
streamlit run app/app.py
```

Optional: `ollama pull granite3.3:2b` for locally generated Granite advisories.