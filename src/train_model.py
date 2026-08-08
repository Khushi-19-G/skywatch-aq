"""
train_model.py
--------------
Trains PM2.5 estimation models on the AOD + weather dataset and evaluates
on two held-out test sets:
  - TEST 1  Delhi held-out   : Delhi rows with date >= 2025-01-01
  - TEST 2  Karachi transfer : ALL Karachi rows (unseen city)

Outputs
-------
  models/pm25_model.pkl          best model + feature list (joblib)
  data/processed/eval_delhi.png  predicted-vs-actual scatter (test 1)
  data/processed/eval_karachi.png  same for test 2
"""

import csv
import math
import pathlib

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
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
    # Satellite
    "aod_055",
    # Geography — encodes location-level pollution baseline
    "latitude",
    "longitude",
    # Time
    "month",
    "day_of_year",
    "day_of_week",
    "season_enc",
    # Weather
    "temperature_2m_mean",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "precipitation_sum",
    "surface_pressure_mean",
    "shortwave_radiation_sum",
]

TARGET_COL = "pm25_daily_avg"

# ---------------------------------------------------------------------------
# Load & prepare data
# ---------------------------------------------------------------------------

def load_data(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    float_cols = [
        "aod_055", "pm25_daily_avg", "latitude", "longitude",
        "temperature_2m_mean", "relative_humidity_2m_mean",
        "wind_speed_10m_mean", "precipitation_sum",
        "surface_pressure_mean", "shortwave_radiation_sum",
    ]
    int_cols = ["month", "day_of_year", "day_of_week", "measurement_count"]
    for r in rows:
        for c in float_cols: r[c] = float(r[c])
        for c in int_cols:   r[c] = int(r[c])
        r["season_enc"] = SEASON_ENCODE[r["season"]]
    return rows


def to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[r[c] for c in FEATURE_COLS] for r in rows], dtype=np.float64)
    y = np.array([r[TARGET_COL] for r in rows],                dtype=np.float64)
    return X, y


def split_data(rows: list[dict]) -> dict[str, list[dict]]:
    train, test1, test2 = [], [], []
    for r in rows:
        if r["city"] == "Delhi":
            (train if r["date"] < "2025-01-01" else test1).append(r)
        else:
            test2.append(r)
    return {"train": train, "test_delhi": test1, "test_karachi": test2}

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict:
    return {
        "label": label,
        "n":     len(y_true),
        "R2":    r2_score(y_true, y_pred),
        "RMSE":  rmse(y_true, y_pred),
        "MAE":   mean_absolute_error(y_true, y_pred),
    }


def naive_metrics(y_train: np.ndarray, y_test: np.ndarray, label: str) -> dict:
    """Predict the training-set mean for every test sample."""
    pred = np.full(len(y_test), float(np.mean(y_train)))
    return compute_metrics(y_test, pred, label)

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

def print_results_table(
    model_name:  str,
    res_train:   dict,
    res_test1:   dict,
    res_test2:   dict,
    naive_test1: dict,
    naive_test2: dict,
) -> None:
    rows = [res_train, res_test1, naive_test1, res_test2, naive_test2]
    hdr  = f"  {'Set':<36}  {'n':>5}  {'R2':>8}  {'RMSE':>7}  {'MAE':>7}"
    sep  = "  " + "-" * (len(hdr) - 2)
    print(f"\n  *** {model_name} ***")
    print(hdr)
    print(sep)
    for r in rows:
        marker = "  " if "Naive" not in r["label"] else "  "
        print(f"  {r['label']:<36}  {r['n']:>5}  "
              f"{r['R2']:>8.4f}  {r['RMSE']:>7.2f}  {r['MAE']:>7.2f}")


def print_importances(importances: np.ndarray, model_name: str) -> None:
    print(f"\n  [{model_name}] Feature importances (mean decrease in R2)")
    print(f"  {'Feature':<32}  Importance")
    print(f"  {'-'*32}  ----------")
    order = np.argsort(importances)[::-1]
    for i in order:
        print(f"  {FEATURE_COLS[i]:<32}  {importances[i]:.4f}")

# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

def scatter_eval(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    title:    str,
    out_path: str,
    r2:       float,
    rmse_val: float,
    color:    str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7f8fa")
    ax.grid(True, color="#e5e7eb", linewidth=0.7, zorder=0)

    lim_lo = float(min(y_true.min(), y_pred.min())) * 0.95
    lim_hi = float(max(y_true.max(), y_pred.max())) * 1.05

    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
            color="#e05c2a", linewidth=1.2, linestyle="--", zorder=2, label="1:1 line")
    ax.scatter(y_true, y_pred, color=color, alpha=0.35, s=16,
               linewidths=0, zorder=3, label="predictions")

    ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Actual PM2.5 (ug/m3)", fontsize=11)
    ax.set_ylabel("Predicted PM2.5 (ug/m3)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.text(0.04, 0.95,
            f"R\u00b2 = {r2:.3f}\nRMSE = {rmse_val:.1f} ug/m3",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    plt.tight_layout()
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Scatter saved -> {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 62)
    print("skywatch-aq  |  PM2.5 model training")
    print("=" * 62)

    # ---- Load & split ----
    rows   = load_data(TRAINING_CSV)
    splits = split_data(rows)

    print(f"\nData splits:")
    for name, subset in splits.items():
        udates = len({r["date"] for r in subset})
        ulocs  = len({r["location_id"] for r in subset})
        print(f"  {name:<18}: {len(subset):>5,} rows  "
              f"({udates} unique dates, {ulocs} locations)")

    print(f"\nFeatures ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    X_tr, y_tr = to_arrays(splits["train"])
    X_t1, y_t1 = to_arrays(splits["test_delhi"])
    X_t2, y_t2 = to_arrays(splits["test_karachi"])

    print(f"\nTrain PM2.5 mean: {y_tr.mean():.1f}  sd: {y_tr.std():.1f}")
    print(f"Test1 PM2.5 mean: {y_t1.mean():.1f}  sd: {y_t1.std():.1f}")
    print(f"Test2 PM2.5 mean: {y_t2.mean():.1f}  sd: {y_t2.std():.1f}")

    # ---- Models ----
    candidates = {
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=4,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.04,
            max_depth=5,
            min_samples_leaf=30,
            l2_regularization=0.5,
            random_state=42,
        ),
    }

    print("\n" + "=" * 62)
    print("RESULTS")
    print("=" * 62)

    all_results: dict[str, dict] = {}

    for model_name, model in candidates.items():
        print(f"\nTraining {model_name}...")
        model.fit(X_tr, y_tr)

        p_tr = model.predict(X_tr)
        p_t1 = model.predict(X_t1)
        p_t2 = model.predict(X_t2)

        res_tr = compute_metrics(y_tr, p_tr, "Train  (Delhi 2023-2024)")
        res_t1 = compute_metrics(y_t1, p_t1, "TEST1  Delhi held-out 2025+")
        res_t2 = compute_metrics(y_t2, p_t2, "TEST2  Karachi transfer")
        nai_t1 = naive_metrics(y_tr, y_t1,   "  Naive mean -> TEST1")
        nai_t2 = naive_metrics(y_tr, y_t2,   "  Naive mean -> TEST2")

        print_results_table(model_name, res_tr, res_t1, res_t2, nai_t1, nai_t2)

        all_results[model_name] = {
            "model":  model,
            "res_t1": res_t1, "res_t2": res_t2,
            "p_t1":   p_t1,   "p_t2":   p_t2,
        }

    # ---- Pick winner: best mean test R2 ----
    def mean_r2(name: str) -> float:
        r = all_results[name]
        return (r["res_t1"]["R2"] + r["res_t2"]["R2"]) / 2.0

    best_name  = max(all_results, key=mean_r2)
    best       = all_results[best_name]
    best_model = best["model"]

    print(f"\n{'='*62}")
    print(f"WINNER: {best_name}  (mean test R2 = {mean_r2(best_name):.4f})")
    print(f"{'='*62}")

    # ---- Feature importance ----
    # Both RF and HistGBT: use permutation importance on TEST1 (larger set)
    print(f"\nComputing permutation importances on TEST1 ({len(y_t1):,} samples)...")
    perm = permutation_importance(
        best_model, X_t1, y_t1,
        n_repeats=10, random_state=42, n_jobs=-1,
        scoring="r2",
    )
    print_importances(perm.importances_mean, best_name)

    # Also print built-in importances if available (RF gives them "for free")
    if hasattr(best_model, "feature_importances_"):
        print(f"\n  [{best_name}] Built-in (impurity) feature importances")
        print(f"  {'Feature':<32}  Importance")
        print(f"  {'-'*32}  ----------")
        imp = best_model.feature_importances_
        for i in np.argsort(imp)[::-1]:
            print(f"  {FEATURE_COLS[i]:<32}  {imp[i]:.4f}")

    # ---- Scatter plots ----
    res_t1 = best["res_t1"]
    res_t2 = best["res_t2"]

    scatter_eval(
        y_t1, best["p_t1"],
        title=f"{best_name}\nDelhi held-out (2025+)",
        out_path=EVAL_DELHI,
        r2=res_t1["R2"], rmse_val=res_t1["RMSE"],
        color="#e05c2a",
    )
    scatter_eval(
        y_t2, best["p_t2"],
        title=f"{best_name}\nKarachi transfer (unseen city)",
        out_path=EVAL_KARACHI,
        r2=res_t2["R2"], rmse_val=res_t2["RMSE"],
        color="#2a7ae0",
    )

    # ---- Save model ----
    pathlib.Path(MODEL_PKL).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model":         best_model,
        "features":      FEATURE_COLS,
        "target":        TARGET_COL,
        "season_encode": SEASON_ENCODE,
        "model_name":    best_name,
    }, MODEL_PKL)
    print(f"\n  Model saved -> {MODEL_PKL}")
    print("\nDone.")


if __name__ == "__main__":
    main()
