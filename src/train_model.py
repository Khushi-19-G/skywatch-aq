"""
train_model.py  (iteration 2)
------------------------------
Changes from v1
  - Target: log1p(pm25_daily_avg); metrics back-transformed to µg/m³
  - Features: lat/lon removed; geo context replaced by weather + season
  - Delhi eval: 5-fold GroupKFold, grouped by date (no leakage)
  - Karachi eval A: zero-shot transfer (Delhi-only train)
  - Karachi eval B: one-monitor calibration (Delhi + earliest 20% Karachi dates)
  - Better model (by Delhi CV R²) saved to models/pm25_model.pkl
    trained on Delhi + Karachi-calibration slice

Outputs
-------
  models/pm25_model.pkl          joblib bundle: model + features + metadata
  data/processed/eval_delhi.png  predicted-vs-actual (Delhi OOF predictions)
  data/processed/eval_karachi.png  same (Karachi scenario b)
"""

import csv
import math
import pathlib

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAINING_CSV = "data/processed/training_data.csv"
MODEL_PKL    = "models/pm25_model.pkl"
EVAL_DELHI   = "data/processed/eval_delhi.png"
EVAL_KARACHI = "data/processed/eval_karachi.png"

SEASON_ENCODE = {"winter": 0, "spring": 1, "monsoon": 2, "autumn": 3}

FEATURE_COLS = [
    "aod_055",
    "month",
    "day_of_year",
    "day_of_week",
    "season_enc",
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "precipitation_sum",
    "surface_pressure_mean",
    "shortwave_radiation_sum",
]

TARGET_COL = "pm25_daily_avg"

N_FOLDS        = 5
KARACHI_CAL_PCT = 0.20   # fraction of earliest Karachi dates used for calibration

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    float_cols = [
        "aod_055", "pm25_daily_avg", "latitude", "longitude",
        "temperature_2m_mean", "relative_humidity_2m_mean",
        "wind_speed_10m_mean", "precipitation_sum",
        "surface_pressure_mean", "shortwave_radiation_sum",
    ]
    for r in rows:
        for c in float_cols: r[c] = float(r[c])
        r["month"]      = int(r["month"])
        r["day_of_year"] = int(r["day_of_year"])
        r["day_of_week"] = int(r["day_of_week"])
        r["season_enc"]  = SEASON_ENCODE[r["season"]]
    return rows


def to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y_log1p)."""
    X     = np.array([[r[c] for c in FEATURE_COLS] for r in rows], dtype=np.float64)
    y_raw = np.array([r[TARGET_COL]                for r in rows], dtype=np.float64)
    y     = np.log1p(y_raw)
    return X, y

# ---------------------------------------------------------------------------
# Metrics — always in original µg/m³ units via expm1
# ---------------------------------------------------------------------------

def eval_metrics(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> dict:
    """Back-transform both arrays, then compute metrics in original units."""
    yt = np.expm1(y_log_true)
    yp = np.expm1(y_log_pred)
    # Clip predictions to [0, ∞) — log model can produce tiny negatives near 0
    yp = np.clip(yp, 0, None)
    return {
        "R2":   r2_score(yt, yp),
        "RMSE": math.sqrt(mean_squared_error(yt, yp)),
        "MAE":  mean_absolute_error(yt, yp),
        "n":    len(yt),
    }


def naive_metrics(y_log_train: np.ndarray, y_log_test: np.ndarray) -> dict:
    """Naive baseline: predict training-set mean (in log space) for every test row."""
    pred = np.full(len(y_log_test), float(np.mean(y_log_train)))
    return eval_metrics(y_log_test, pred)

# ---------------------------------------------------------------------------
# GroupKFold Delhi CV
# ---------------------------------------------------------------------------

def delhi_cv(
    model_factory,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = N_FOLDS,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    5-fold GroupKFold on Delhi data (groups = integer date codes so all rows
    with the same date stay together).
    Returns fold_stats dict, out-of-fold true values (original units),
    and out-of-fold predicted values (original units).
    """
    gkf   = GroupKFold(n_splits=n_splits)
    r2s, rmses, maes = [], [], []
    oof_true = np.zeros(len(y))
    oof_pred = np.zeros(len(y))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
        m = model_factory()
        m.fit(X[tr_idx], y[tr_idx])
        pv = m.predict(X[va_idx])
        met = eval_metrics(y[va_idx], pv)
        r2s.append(met["R2"]); rmses.append(met["RMSE"]); maes.append(met["MAE"])
        oof_true[va_idx] = np.expm1(y[va_idx])
        oof_pred[va_idx] = np.clip(np.expm1(pv), 0, None)

    return {
        "R2_mean":   float(np.mean(r2s)),   "R2_std":   float(np.std(r2s)),
        "RMSE_mean": float(np.mean(rmses)), "RMSE_std": float(np.std(rmses)),
        "MAE_mean":  float(np.mean(maes)),  "MAE_std":  float(np.std(maes)),
        "folds":     list(zip(r2s, rmses, maes)),
    }, oof_true, oof_pred

# ---------------------------------------------------------------------------
# Karachi split — earliest 20% of dates for calibration
# ---------------------------------------------------------------------------

def karachi_split(
    rows_k: list[dict],
    cal_pct: float = KARACHI_CAL_PCT,
) -> tuple[list[dict], list[dict]]:
    """Return (calibration_rows, test_rows) by earliest date fraction."""
    sorted_dates = sorted({r["date"] for r in rows_k})
    n_cal        = max(1, int(len(sorted_dates) * cal_pct))
    cal_dates    = set(sorted_dates[:n_cal])
    cal  = [r for r in rows_k if r["date"] in cal_dates]
    test = [r for r in rows_k if r["date"] not in cal_dates]
    return cal, test

# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

SEP = "  " + "-" * 66

def print_cv_results(model_name: str, cv: dict, naive: dict) -> None:
    print(f"\n  *** {model_name}  —  Delhi 5-fold GroupKFold CV ***")
    print(f"  {'Metric':<8}  {'Mean':>8}  {'±Std':>7}  |  Naive baseline")
    print(SEP)
    for metric, naive_key in [("R2","R2"), ("RMSE","RMSE"), ("MAE","MAE")]:
        mu  = cv[f"{metric}_mean"]
        sd  = cv[f"{metric}_std"]
        nv  = naive[metric]
        print(f"  {metric:<8}  {mu:>8.4f}  {sd:>7.4f}  |  {nv:.4f}")
    print(f"  Per-fold R²: "
          + "  ".join(f"F{i+1}:{r2:.3f}" for i,(r2,_,_) in enumerate(cv["folds"])))


def print_transfer_block(model_name: str, scenario: str, met: dict, naive: dict) -> None:
    print(f"\n  *** {model_name}  —  Karachi {scenario} ***")
    print(f"  {'Metric':<8}  {'Model':>8}  {'Naive':>8}")
    print(SEP)
    for key in ("R2", "RMSE", "MAE"):
        print(f"  {key:<8}  {met[key]:>8.4f}  {naive[key]:>8.4f}")
    print(f"  n = {met['n']}")


def print_importances(importances: np.ndarray, label: str) -> None:
    print(f"\n  [{label}] Permutation feature importances (mean ΔR²)")
    print(f"  {'Feature':<32}  Importance")
    print(f"  {'-'*32}  ----------")
    for i in np.argsort(importances)[::-1]:
        print(f"  {FEATURE_COLS[i]:<32}  {importances[i]:+.4f}")

# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

def scatter_eval(
    y_true: np.ndarray, y_pred: np.ndarray,
    title: str, out_path: str, r2: float, rmse_val: float, color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7f8fa")
    ax.grid(True, color="#e5e7eb", linewidth=0.7, zorder=0)

    lo = float(min(y_true.min(), y_pred.min())) * 0.95
    hi = float(max(y_true.max(), y_pred.max())) * 1.05
    ax.plot([lo, hi], [lo, hi], color="#e05c2a", linewidth=1.2,
            linestyle="--", zorder=2, label="1:1 line")
    ax.scatter(y_true, y_pred, color=color, alpha=0.30, s=14,
               linewidths=0, zorder=3, label="predictions")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual PM2.5 (ug/m3)", fontsize=11)
    ax.set_ylabel("Predicted PM2.5 (ug/m3)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.text(0.04, 0.95, f"R\u00b2 = {r2:.3f}\nRMSE = {rmse_val:.1f} ug/m3",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    plt.tight_layout()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Scatter saved -> {out_path}")

# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_rf():
    return RandomForestRegressor(
        n_estimators=400, max_features="sqrt",
        min_samples_leaf=4, random_state=42, n_jobs=-1,
    )


def make_hgbt():
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.04, max_depth=5,
        min_samples_leaf=30, l2_regularization=0.5, random_state=42,
    )


MODELS = {
    "RandomForest":        make_rf,
    "HistGradientBoosting": make_hgbt,
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 68)
    print("skywatch-aq  |  PM2.5 model training  (iteration 2)")
    print("=" * 68)
    print(f"Target      : log1p(pm25_daily_avg)  → metrics back-transformed to ug/m3")
    print(f"Features    : {FEATURE_COLS}")

    # ---- Load ----
    rows    = load_rows(TRAINING_CSV)
    delhi   = [r for r in rows if r["city"] == "Delhi"]
    karachi = [r for r in rows if r["city"] == "Karachi"]
    print(f"\nDelhi rows  : {len(delhi):,}  |  Karachi rows: {len(karachi):,}")

    X_d, y_d = to_arrays(delhi)
    X_k, y_k = to_arrays(karachi)

    # Date → integer group code for Delhi GroupKFold
    unique_dates_d   = sorted({r["date"] for r in delhi})
    date_to_group    = {d: i for i, d in enumerate(unique_dates_d)}
    groups_d         = np.array([date_to_group[r["date"]] for r in delhi])

    # Karachi split: cal 20% / test 80% by date
    k_cal_rows, k_test_rows = karachi_split(karachi)
    X_kc, y_kc = to_arrays(k_cal_rows)
    X_kt, y_kt = to_arrays(k_test_rows)
    print(f"Karachi cal : {len(k_cal_rows):,} rows ({len({r['date'] for r in k_cal_rows})} dates)")
    print(f"Karachi test: {len(k_test_rows):,} rows ({len({r['date'] for r in k_test_rows})} dates)")

    # ---- Delhi CV ----
    print("\n" + "=" * 68)
    print("DELHI  —  5-fold GroupKFold cross-validation")
    print("=" * 68)

    cv_results: dict[str, dict] = {}
    naive_delhi = naive_metrics(y_d, y_d)  # same distribution (upper bound of naive)
    # Proper naive: use per-fold held-out distribution — approximate with overall mean
    naive_delhi_cv = {
        "R2": r2_score(np.expm1(y_d),
                       np.full(len(y_d), np.expm1(y_d).mean())),
        "RMSE": math.sqrt(mean_squared_error(
                    np.expm1(y_d), np.full(len(y_d), np.expm1(y_d).mean()))),
        "MAE":  mean_absolute_error(
                    np.expm1(y_d), np.full(len(y_d), np.expm1(y_d).mean())),
    }

    oof_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name, factory in MODELS.items():
        cv, oof_true, oof_pred = delhi_cv(factory, X_d, y_d, groups_d)
        cv_results[name] = cv
        oof_preds[name]  = (oof_true, oof_pred)
        print_cv_results(name, cv, naive_delhi_cv)

    # Best model = highest mean Delhi CV R²
    best_name = max(cv_results, key=lambda n: cv_results[n]["R2_mean"])
    best_cv   = cv_results[best_name]
    print(f"\n  --> Best by Delhi CV R²: {best_name}  "
          f"(mean R² = {best_cv['R2_mean']:.4f})")

    # ---- Karachi transfer ----
    print("\n" + "=" * 68)
    print("KARACHI  —  transfer scenarios")
    print("=" * 68)

    best_factory = MODELS[best_name]

    # Scenario A: zero-shot (train on all Delhi, test on all Karachi)
    m_a = best_factory()
    m_a.fit(X_d, y_d)
    pred_ka  = m_a.predict(X_k)
    met_ka   = eval_metrics(y_k, pred_ka)
    naive_ka = naive_metrics(y_d, y_k)
    print_transfer_block(best_name, "A: zero-shot transfer (all Delhi -> all Karachi)", met_ka, naive_ka)

    # Scenario B: one-monitor calibration (all Delhi + Karachi cal 20%)
    X_train_b = np.vstack([X_d, X_kc])
    y_train_b = np.concatenate([y_d, y_kc])
    m_b = best_factory()
    m_b.fit(X_train_b, y_train_b)
    pred_kb  = m_b.predict(X_kt)
    met_kb   = eval_metrics(y_kt, pred_kb)
    naive_kb = naive_metrics(y_train_b, y_kt)
    print_transfer_block(best_name,
        f"B: one-monitor calibration (Delhi + {KARACHI_CAL_PCT*100:.0f}% Karachi cal -> remaining Karachi)",
        met_kb, naive_kb)

    # ---- Feature importances (permutation on full Delhi) ----
    print(f"\nComputing permutation importances on Delhi (n={len(X_d):,})...")
    perm = permutation_importance(
        m_a, X_d, y_d, n_repeats=10, random_state=42, n_jobs=-1, scoring="r2",
    )
    print_importances(perm.importances_mean, best_name)

    # ---- Scatter plots ----
    oof_t, oof_p = oof_preds[best_name]
    scatter_eval(
        oof_t, oof_p,
        title=f"{best_name}  —  Delhi 5-fold OOF\n(all Delhi, grouped by date)",
        out_path=EVAL_DELHI,
        r2=best_cv["R2_mean"], rmse_val=best_cv["RMSE_mean"],
        color="#e05c2a",
    )
    y_kt_orig   = np.expm1(y_kt)
    y_kb_pred   = np.clip(np.expm1(pred_kb), 0, None)
    scatter_eval(
        y_kt_orig, y_kb_pred,
        title=f"{best_name}  —  Karachi scenario B\n(Delhi + {KARACHI_CAL_PCT*100:.0f}% cal -> held-out Karachi)",
        out_path=EVAL_KARACHI,
        r2=met_kb["R2"], rmse_val=met_kb["RMSE"],
        color="#2a7ae0",
    )

    # ---- Save final model ----
    # Retrain best model on Delhi + Karachi calibration slice (scenario B training set)
    print(f"\nRetraining {best_name} on Delhi + Karachi-cal ({len(X_train_b):,} rows) for deployment...")
    m_final = best_factory()
    m_final.fit(X_train_b, y_train_b)

    pathlib.Path(MODEL_PKL).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model":         m_final,
        "features":      FEATURE_COLS,
        "target":        TARGET_COL,
        "target_transform": "log1p",
        "season_encode": SEASON_ENCODE,
        "model_name":    best_name,
    }, MODEL_PKL)
    print(f"  Model saved -> {MODEL_PKL}")
    print("\nDone.")


if __name__ == "__main__":
    main()
