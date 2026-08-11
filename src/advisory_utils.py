"""
advisory_utils.py
-----------------
Plain-language public health advisories for SkyWatch-AQ regional maps.

Generation backends, tried in order:
  1. IBM Granite via watsonx.ai          — if WATSONX_API_KEY + WATSONX_PROJECT_ID set
  2. IBM Granite via Ollama (local)      — if http://localhost:11434 is reachable
     Model: granite3.3:2b  (2-second health-check so deployed app skips fast)
  3. IBM Granite via Hugging Face router — if HF_TOKEN set
     Endpoint: https://router.huggingface.co/v1/chat/completions  (OpenAI-compatible)
     Models tried (one attempt each, no retry loop):
       ibm-granite/granite-3.3-8b-instruct  /  ibm-granite/granite-3.1-8b-instruct
     Note: as of 2025, ibm-granite models carry inference=N/A on HF.
  4. Rule-based template                 — always available

Credentials (from .env or Streamlit secrets):
  WATSONX_API_KEY      — IBM Cloud IAM API key
  WATSONX_PROJECT_ID   — watsonx.ai project ID
  WATSONX_URL          — (optional) watsonx.ai endpoint, default us-south.ml.cloud.ibm.com
  HF_TOKEN             — Hugging Face user access token
  OLLAMA_URL           — (optional) Ollama base URL, default http://localhost:11434
  OLLAMA_MODEL         — (optional) Ollama model tag, default granite3.3:2b

Public API
----------
  build_advisory_context(cells, exposure, region_key, date_str, is_forecast)
      -> dict   (structured context for advisory generation)
  generate_advisory(context) -> {"text": str, "source": str}
      source: "IBM Granite (watsonx.ai)"
            | "IBM Granite (running locally via Ollama)"
            | "IBM Granite via Hugging Face"
            | "rule-based"
"""

import math
import os
import pathlib
import sys
from collections import Counter
from typing import Optional

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
_SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Health band severity for ordering
# ---------------------------------------------------------------------------
_BAND_SEVERITY_ORDER = [
    "Hazardous", "Very Unhealthy", "Unhealthy", "USG", "Moderate", "Good",
]
_BAND_ADVICE = {
    "Hazardous":       "a health emergency",
    "Very Unhealthy":  "very unhealthy",
    "Unhealthy":       "unhealthy",
    "USG":             "unhealthy for sensitive groups",
    "Moderate":        "moderate",
    "Good":            "good",
}
_BAND_RISK_GROUPS = {
    "Hazardous":      "the entire population",
    "Very Unhealthy": "everyone, especially people with heart or lung conditions, children, and the elderly",
    "Unhealthy":      "people with respiratory or cardiovascular conditions, children, pregnant women, and the elderly",
    "USG":            "children, the elderly, pregnant women, and those with asthma or heart disease",
    "Moderate":       "unusually sensitive individuals",
    "Good":           "almost nobody",
}

# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _top_affected_areas(cells: list[dict], n: int = 3) -> list[str]:
    """
    Return up to n plain-language descriptions of the worst-affected areas
    based on mean PM2.5 in 2°×2° buckets.  Descriptions use rough compass
    quadrants relative to the tile mid-point, with each distinct direction
    appearing at most once.
    """
    if not cells:
        return []

    from collections import defaultdict
    # Bucket into 2° cells to avoid redundant adjacent descriptions
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for c in cells:
        key = (round(c["lat"] / 2) * 2, round(c["lon"] / 2) * 2)
        buckets[key].append(c["pm25"])

    ranked = sorted(buckets.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))

    all_lats = [c["lat"] for c in cells]
    all_lons = [c["lon"] for c in cells]
    mid_lat = (min(all_lats) + max(all_lats)) / 2
    mid_lon = (min(all_lons) + max(all_lons)) / 2

    seen_dirs: set[str] = set()
    descriptions: list[str] = []
    for (blat, blon), pms in ranked:
        v = "northern" if blat > mid_lat else "southern"
        h = "eastern"  if blon > mid_lon else "western"
        direction = f"{v} {h}"
        if direction in seen_dirs:
            continue
        seen_dirs.add(direction)
        mean_pm = sum(pms) / len(pms)
        descriptions.append(f"the {direction} part (~{mean_pm:.0f} µg/m³)")
        if len(descriptions) >= n:
            break

    return descriptions


def build_advisory_context(
    cells: list[dict],
    exposure: dict,
    region_key: str,
    date_str: str,
    is_forecast: bool = False,
) -> dict:
    """
    Build structured context dict for advisory generation.

    Keys returned:
      region_label, date_str, is_forecast,
      headline,           — e.g. "~37.5 M people estimated in Very Unhealthy+ air"
      worst_band,         — most severe band with nonzero pop
      by_band,            — {band: millions}
      coverage_pct,       — % of land cells with satellite data
      no_data_pop,        — millions without coverage
      top_areas,          — list of plain-English worst-area descriptions
      n_cells,
    """
    from population_utils import exposure_headline, _BAND_SEVERITY
    from map_utils import REGIONS

    region_label = REGIONS.get(region_key, {}).get("label", region_key)
    n_land = REGIONS.get(region_key, {}).get("n_land", 1)
    n_cells = len(cells)
    coverage_pct = round(100 * n_cells / max(n_land, 1))

    by_band = exposure.get("by_band", {}) if exposure.get("available") else {}
    no_data_pop = exposure.get("no_data_pop", 0.0)

    # Headline without "today" suffix — caller adds temporal context
    headline = exposure_headline(exposure, suffix="")
    if is_forecast:
        headline = headline  # caller will prepend "Tomorrow:" in UI
    else:
        headline = headline + " today" if headline else ""

    # Worst band
    worst_band = None
    for band in _BAND_SEVERITY:
        if by_band.get(band, 0.0) >= 0.05:
            worst_band = band
            break

    top_areas = _top_affected_areas(cells)

    return {
        "region_label":  region_label,
        "date_str":      date_str,
        "is_forecast":   is_forecast,
        "headline":      headline,
        "worst_band":    worst_band,
        "by_band":       by_band,
        "coverage_pct":  coverage_pct,
        "no_data_pop":   no_data_pop,
        "top_areas":     top_areas,
        "n_cells":       n_cells,
    }


# ---------------------------------------------------------------------------
# Watsonx IAM token helper
# ---------------------------------------------------------------------------

def _get_iam_token(api_key: str) -> Optional[str]:
    """Exchange an IBM Cloud IAM API key for a bearer token."""
    try:
        import requests as _req
        r = _req.post(
            "https://iam.cloud.ibm.com/identity/token",
            data={
                "grant_type":    "urn:ibm:params:oauth:grant-type:apikey",
                "apikey":        api_key,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("access_token")
    except Exception as e:
        print(f"  [advisory] IAM token error: {e}")
        return None


# ---------------------------------------------------------------------------
# Granite generation
# ---------------------------------------------------------------------------

_GRANITE_MODEL = "ibm/granite-3-8b-instruct"

def _build_prompt(ctx: dict) -> str:
    by_band = ctx["by_band"]
    band_lines = "\n".join(
        f"  • {band}: {pop:.1f} M people"
        for band in _BAND_SEVERITY_ORDER
        if by_band.get(band, 0.0) >= 0.05
    )
    areas = ", ".join(ctx["top_areas"]) if ctx["top_areas"] else "various parts of the region"
    temporal = "tomorrow" if ctx["is_forecast"] else "today"
    coverage = ctx["coverage_pct"]
    no_data = ctx["no_data_pop"]

    return (
        f"<|system|>\n"
        f"You are a public health communicator writing air-quality advisories. "
        f"Be calm, factual, and concise. Never use panic language. "
        f"Always include the phrase 'estimates, not measurements'. "
        f"Write for a general public audience.\n"
        f"<|user|>\n"
        f"Write a 3–5 sentence public health advisory for the following air-quality situation. "
        f"Region: {ctx['region_label']}. Date: {ctx['date_str']} ({temporal}). "
        f"{'This is a FORECAST — higher uncertainty than observed data.' if ctx['is_forecast'] else ''}\n\n"
        f"Estimated population exposure ({coverage}% of the region has satellite coverage):\n"
        f"{band_lines if band_lines else '  • No significant pollution detected'}\n"
        f"Areas most affected: {areas}.\n"
        f"Approximately {no_data:.1f} million people are in cloud-covered areas with no available estimate.\n\n"
        f"The advisory must:\n"
        f"- State overall conditions in plain language\n"
        f"- Name who is most at risk\n"
        f"- Give 2–3 concrete, actionable recommendations\n"
        f"- Include the phrase 'estimates, not measurements'\n"
        f"- Be 3–5 sentences total\n"
        f"<|assistant|>\n"
    )


def _call_granite(api_key: str, project_id: str, url: str, prompt: str) -> Optional[str]:
    """Call the watsonx.ai text generation endpoint. Returns advisory text or None."""
    token = _get_iam_token(api_key)
    if not token:
        return None

    try:
        import requests as _req
        endpoint = f"{url.rstrip('/')}/ml/v1/text/generation?version=2023-05-29"
        payload = {
            "model_id":   _GRANITE_MODEL,
            "project_id": project_id,
            "input":      prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens":  300,
                "min_new_tokens":  60,
                "stop_sequences":  ["<|user|>", "<|system|>"],
                "repetition_penalty": 1.05,
            },
        }
        r = _req.post(
            endpoint,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json",
                "Accept":        "application/json",
            },
            timeout=30,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if results:
            return results[0].get("generated_text", "").strip()
        return None
    except Exception as e:
        print(f"  [advisory] Granite API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def _rule_based_advisory(ctx: dict) -> str:
    """Generate a template advisory from context. Always succeeds."""
    worst = ctx["worst_band"] or "Moderate"
    region = ctx["region_label"]
    date   = ctx["date_str"]
    temporal = "tomorrow" if ctx["is_forecast"] else "today"
    condition = _BAND_ADVICE.get(worst, "elevated")
    at_risk   = _BAND_RISK_GROUPS.get(worst, "sensitive individuals")
    by_band   = ctx["by_band"]
    coverage  = ctx["coverage_pct"]
    areas     = ", ".join(ctx["top_areas"]) if ctx["top_areas"] else "parts of the region"

    # Count people in bands at or above the worst
    idx = _BAND_SEVERITY_ORDER.index(worst)
    total_exposed = sum(
        by_band.get(b, 0.0) for b in _BAND_SEVERITY_ORDER[:idx + 1]
    )

    forecast_caveat = (
        " This is a forecast based on persisted satellite AOD and weather model data — "
        "actual conditions may differ."
    ) if ctx["is_forecast"] else ""

    # Pick recommendations based on severity
    if worst in ("Hazardous", "Very Unhealthy"):
        recs = (
            "Everyone should avoid all outdoor activities and stay indoors with windows closed. "
            "Run air purifiers if available and wear N95/FFP2 masks if outdoor travel is unavoidable."
        )
    elif worst == "Unhealthy":
        recs = (
            "Reduce prolonged outdoor exertion, especially strenuous exercise. "
            "Keep windows closed during peak afternoon hours and consider wearing a mask outdoors."
        )
    elif worst == "USG":
        recs = (
            "Sensitive groups should limit outdoor activities and stay indoors during peak hours. "
            "Others may exercise outdoors but should take breaks and watch for symptoms."
        )
    else:
        recs = (
            "Air quality is acceptable for most people. "
            "Unusually sensitive individuals may wish to limit prolonged outdoor exertion."
        )

    lines = [
        f"Air quality conditions are estimated to be {condition} across much of "
        f"{region} {temporal}, with the highest concentrations in {areas}.",

        f"These are estimates, not measurements — approximately {total_exposed:.1f} million people "
        f"are in areas where air quality is estimated at {worst} levels or worse "
        f"({coverage}% of the region has satellite coverage{'; ' + str(round(ctx['no_data_pop'],1)) + ' M people are in cloud-covered areas with no estimate' if ctx['no_data_pop'] > 0.5 else ''}).",

        f"People most at risk include {at_risk}, who should take extra precautions. " + recs,
    ]

    if ctx["is_forecast"]:
        lines.append(forecast_caveat.strip())

    return " ".join(lines)


# ---------------------------------------------------------------------------
# Ollama local backend
# ---------------------------------------------------------------------------

_OLLAMA_DEFAULT_URL   = "http://localhost:11434"
_OLLAMA_DEFAULT_MODEL = "granite3.3:2b"
_OLLAMA_HEALTH_TIMEOUT = 2   # seconds — short so deployed app skips instantly


def _ollama_available(base_url: str) -> bool:
    """Return True if Ollama's root endpoint responds within the health timeout."""
    try:
        import requests as _req
        r = _req.get(base_url + "/", timeout=_OLLAMA_HEALTH_TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False


def _call_granite_ollama(base_url: str, model: str, prompt_str: str) -> Optional[str]:
    """
    Call Ollama /api/chat with the given model and user prompt.
    Uses a non-streaming request; returns the assistant message or None.
    """
    try:
        import requests as _req
        r = _req.post(
            base_url.rstrip("/") + "/api/chat",
            json={
                "model":  model,
                "stream": False,
                "messages": [
                    {
                        "role":    "system",
                        "content": (
                            "You are a public health communicator writing air-quality advisories. "
                            "Be calm, factual, and concise. Never use panic language. "
                            "You MUST use the exact phrase 'estimates, not measurements' verbatim. "
                            "Write for a general public audience."
                        ),
                    },
                    {"role": "user", "content": prompt_str},
                ],
                "options": {
                    "temperature":    0.3,
                    "num_predict":    350,
                    "repeat_penalty": 1.05,
                },
            },
            timeout=120,   # generation can take 30-60 s for a 2B model on CPU
        )
        r.raise_for_status()
        text = r.json().get("message", {}).get("content", "").strip()
        return text or None
    except Exception as e:
        print(f"  [advisory/ollama] error: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Hugging Face Inference backend
# ---------------------------------------------------------------------------

# Ordered list of Granite model IDs to try via HF router (one attempt each)
_HF_GRANITE_MODELS = [
    "ibm-granite/granite-3.3-8b-instruct",
    "ibm-granite/granite-3.1-8b-instruct",
]
_HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"


def _call_granite_hf(hf_token: str, prompt_str: str) -> Optional[str]:
    """
    Try each Granite model via the HF router (OpenAI-compatible endpoint).
    One clean attempt per model — no retry loop.
    Returns generated text on success, None if all attempts fail.
    Prints the exact HTTP status + error body for each rejection.
    """
    try:
        import requests as _req
    except ImportError:
        return None

    # Convert the chat-template prompt into messages
    messages = [
        {
            "role":    "system",
            "content": (
                "You are a public health communicator writing air-quality advisories. "
                "Be calm, factual, and concise. Never use panic language. "
                "Always include the phrase 'estimates, not measurements'. "
                "Write for a general public audience."
            ),
        },
        {"role": "user", "content": prompt_str},
    ]

    for model_id in _HF_GRANITE_MODELS:
        try:
            r = _req.post(
                _HF_ROUTER_URL,
                headers={
                    "Authorization": f"Bearer {hf_token}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       model_id,
                    "messages":    messages,
                    "max_tokens":  350,
                    "temperature": 0.3,
                    "stream":      False,
                },
                timeout=45,
            )
            print(f"  [advisory/HF] {model_id}: HTTP {r.status_code}")
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"].strip()
                if text:
                    return text
            else:
                # Print exact rejection so caller can see it
                print(f"  [advisory/HF] rejection body: {r.text[:400]}")
        except Exception as e:
            print(f"  [advisory/HF] {model_id}: {type(e).__name__}: {e}")

    return None


def _build_user_message(ctx: dict) -> str:
    """Plain-text user message for HF chat API (no special tokens needed)."""
    by_band = ctx["by_band"]
    band_lines = "\n".join(
        f"  \u2022 {band}: {by_band[band]:.1f} M people"
        for band in _BAND_SEVERITY_ORDER
        if by_band.get(band, 0.0) >= 0.05
    )
    areas = ", ".join(ctx["top_areas"]) if ctx["top_areas"] else "various parts of the region"
    temporal = "tomorrow" if ctx["is_forecast"] else "today"
    coverage = ctx["coverage_pct"]
    no_data  = ctx["no_data_pop"]

    return (
        f"Write a 3–5 sentence public health advisory for the following air-quality situation.\n\n"
        f"Region: {ctx['region_label']}. Date: {ctx['date_str']} ({temporal}).\n"
        + (f"This is a FORECAST — higher uncertainty than observed data.\n\n" if ctx["is_forecast"] else "\n")
        + f"Estimated population exposure ({coverage}% of the region has satellite coverage):\n"
        f"{band_lines if band_lines else '  • No significant pollution detected'}\n"
        f"Areas most affected: {areas}.\n"
        f"Approximately {no_data:.1f} million people are in cloud-covered areas with no available estimate.\n\n"
        f"Requirements:\n"
        f"- State overall conditions in plain language\n"
        f"- Name who is most at risk\n"
        f"- Give 2–3 concrete, actionable recommendations\n"
        f"- Include the exact phrase 'estimates, not measurements'\n"
        f"- 3–5 sentences total, calm and factual"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_advisory(context: dict) -> dict:
    """
    Generate a plain-language health advisory for the given context.

    Tries backends in order:
      1. watsonx.ai  (WATSONX_API_KEY + WATSONX_PROJECT_ID)
      2. Ollama      (local, OLLAMA_URL / OLLAMA_MODEL; fast health-check)
      3. HF router   (HF_TOKEN)
      4. Rule-based  (always succeeds)

    Returns {"text": str, "source": str}.
    """
    user_msg = _build_user_message(context)   # shared by Ollama + HF

    # --- Backend 1: watsonx ---
    api_key    = os.getenv("WATSONX_API_KEY", "").strip()
    project_id = os.getenv("WATSONX_PROJECT_ID", "").strip()
    wx_url     = os.getenv("WATSONX_URL",
                           "https://us-south.ml.cloud.ibm.com").strip()

    if api_key and project_id:
        prompt = _build_prompt(context)
        text   = _call_granite(api_key, project_id, wx_url, prompt)
        if text:
            return {"text": text, "source": "IBM Granite (watsonx.ai)"}
        print("  [advisory] watsonx failed, trying Ollama")

    # --- Backend 2: Ollama (local) ---
    ollama_url   = os.getenv("OLLAMA_URL",   _OLLAMA_DEFAULT_URL).strip()
    ollama_model = os.getenv("OLLAMA_MODEL", _OLLAMA_DEFAULT_MODEL).strip()
    if _ollama_available(ollama_url):
        print(f"  [advisory/ollama] Ollama reachable at {ollama_url}, model={ollama_model}")
        text = _call_granite_ollama(ollama_url, ollama_model, user_msg)
        if text:
            return {"text": text,
                    "source": "IBM Granite (running locally via Ollama)"}
        print("  [advisory/ollama] generation failed, trying HF")
    else:
        print(f"  [advisory/ollama] not reachable at {ollama_url} — skipping")

    # --- Backend 3: Hugging Face ---
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token:
        text = _call_granite_hf(hf_token, user_msg)
        if text:
            return {"text": text, "source": "IBM Granite via Hugging Face"}
        print("  [advisory] HF backend failed, falling back to rule-based")

    # --- Backend 4: rule-based ---
    return {"text": _rule_based_advisory(context), "source": "rule-based"}
