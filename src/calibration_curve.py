"""
calibration_curve.py
--------------------
Sweeps Karachi calibration fractions (10%–60% of earliest Karachi dates),
always testing on the FIXED final 40% of Karachi dates.
Trains: all Delhi rows + calibration slice.
Reports R² and MAE at each fraction.
Saves: data/processed/calibration_curve.png
"""

import csv, math, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

TRAINING_CSV = "data/processed/training_data.csv"
CURVE_PNG    = "data/processed/calibration_curve.png"

SEASON_ENCODE = {"winter": 0, "spring": 1, "monsoon": 2, "autumn": 3}
FEATURE_COLS  = [
    "aod_055", "month", "day_of_year", "day_of_week", "season_enc",
    "temperature_2m_mean", "relative_humidity_2m_mean", "wind_speed_10m_mean",
    "precipitation_sum", "surface_pressure_mean", "shortwave_radiation_sum",
]

CAL_FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
TEST_FRACTION = 0.40   # always the LAST 40% of Karachi dates

def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    float_cols = ["aod_055","pm25_daily_avg","latitude","longitude",
                  "temperature_2m_mean","relative_humidity_2m_mean",
                  "wind_speed_10m_mean","precipitation_sum",
                  "surface_pressure_mean","shortwave_radiation_sum"]
    for r in rows:
        for c in float_cols: r[c] = float(r[c])
        r["month"] = int(r["month"]); r["day_of_year"] = int(r["day_of_year"])
        r["day_of_week"] = int(r["day_of_week"])
        r["season_enc"] = SEASON_ENCODE[r["season"]]
    return rows

def to_Xy(rows):
    X = np.array([[r[c] for c in FEATURE_COLS] for r in rows], dtype=np.float64)
    y = np.log1p(np.array([r["pm25_daily_avg"] for r in rows], dtype=np.float64))
    return X, y

def eval_orig(y_log_true, y_log_pred):
    yt = np.expm1(y_log_true)
    yp = np.clip(np.expm1(y_log_pred), 0, None)
    return {
        "R2":   r2_score(yt, yp),
        "RMSE": math.sqrt(mean_squared_error(yt, yp)),
        "MAE":  mean_absolute_error(yt, yp),
        "n":    len(yt),
    }

def make_rf():
    return RandomForestRegressor(n_estimators=400, max_features="sqrt",
                                 min_samples_leaf=4, random_state=42, n_jobs=-1)

def main():
    rows    = load_rows(TRAINING_CSV)
    delhi   = [r for r in rows if r["city"] == "Delhi"]
    karachi = [r for r in rows if r["city"] == "Karachi"]
    X_d, y_d = to_Xy(delhi)

    # Fixed test set: last TEST_FRACTION of Karachi dates
    sorted_dates = sorted({r["date"] for r in karachi})
    n_test_dates = max(1, int(len(sorted_dates) * TEST_FRACTION))
    test_dates   = set(sorted_dates[-n_test_dates:])
    # The pool of dates available for calibration = everything NOT in test
    pool_dates   = [d for d in sorted_dates if d not in test_dates]

    k_test = [r for r in karachi if r["date"] in test_dates]
    X_kt, y_kt = to_Xy(k_test)
    print(f"Karachi test set (fixed, last {TEST_FRACTION*100:.0f}%): "
          f"{len(k_test)} rows, {len(test_dates)} dates")
    print(f"Calibration pool: {len(pool_dates)} dates available\n")

    # Naive baseline on test set
    naive_pred = np.full(len(y_kt), np.mean(y_d))
    naive = eval_orig(y_kt, naive_pred)

    results = []
    for frac in CAL_FRACTIONS:
        n_cal = max(1, int(len(pool_dates) * frac))
        cal_dates = set(pool_dates[:n_cal])        # earliest n_cal dates
        k_cal = [r for r in karachi if r["date"] in cal_dates]
        X_kc, y_kc = to_Xy(k_cal)

        X_train = np.vstack([X_d, X_kc])
        y_train = np.concatenate([y_d, y_kc])

        m = make_rf()
        m.fit(X_train, y_train)
        pred = m.predict(X_kt)
        met = eval_orig(y_kt, pred)
        results.append({
            "frac": frac, "n_cal_dates": n_cal, "n_cal_rows": len(k_cal),
            **met,
        })
        print(f"  cal={frac*100:.0f}%  n_dates={n_cal:3d}  n_rows={len(k_cal):4d}  "
              f"R2={met['R2']:+.4f}  RMSE={met['RMSE']:.2f}  MAE={met['MAE']:.2f}")

    # Summary table
    print("\n" + "="*70)
    print(f"{'Cal%':>5}  {'Cal dates':>9}  {'Cal rows':>8}  "
          f"{'R2':>8}  {'RMSE':>7}  {'MAE':>7}")
    print("-"*70)
    for r in results:
        print(f"{r['frac']*100:>4.0f}%  {r['n_cal_dates']:>9}  {r['n_cal_rows']:>8}  "
              f"{r['R2']:>8.4f}  {r['RMSE']:>7.2f}  {r['MAE']:>7.2f}")
    print(f"{'Naive':>5}  {'—':>9}  {'—':>8}  "
          f"{naive['R2']:>8.4f}  {naive['RMSE']:>7.2f}  {naive['MAE']:>7.2f}")
    print("="*70)

    # Plot
    fracs  = [r["frac"] * 100 for r in results]
    r2s    = [r["R2"]   for r in results]
    maes   = [r["MAE"]  for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle("Karachi PM2.5 model quality vs. calibration fraction\n"
                 "(train: all Delhi + earliest X% Karachi dates;  "
                 f"test: fixed last {TEST_FRACTION*100:.0f}% Karachi dates)",
                 fontsize=10)

    for ax in (ax1, ax2):
        ax.set_facecolor("#f7f8fa")
        ax.grid(True, color="#e5e7eb", linewidth=0.7)
        ax.set_xlabel("Karachi calibration fraction (%)", fontsize=10)

    ax1.plot(fracs, r2s, "o-", color="#2a7ae0", linewidth=2, markersize=7)
    ax1.axhline(naive["R2"], color="#999", linewidth=1, linestyle="--", label="Naive baseline")
    ax1.axhline(0, color="#e05c2a", linewidth=0.8, linestyle=":")
    ax1.set_ylabel("R² (test Karachi)", fontsize=10)
    ax1.set_title("R²", fontsize=10)
    ax1.legend(fontsize=8)

    ax2.plot(fracs, maes, "s-", color="#7c5cd8", linewidth=2, markersize=7)
    ax2.axhline(naive["MAE"], color="#999", linewidth=1, linestyle="--", label="Naive baseline")
    ax2.set_ylabel("MAE µg/m³ (test Karachi)", fontsize=10)
    ax2.set_title("MAE", fontsize=10)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    pathlib.Path(CURVE_PNG).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CURVE_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved -> {CURVE_PNG}")

if __name__ == "__main__":
    main()
