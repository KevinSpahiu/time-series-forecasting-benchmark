from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from benchmark_design_v3 import (
    ROOT_CLEANED,
    OUT_BENCHMARK,
    DOMAIN_CONFIGS,
    build_and_save_benchmark,
    find_existing_folder,
    load_series,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ============================================================
# OUTPUT
# ============================================================

ROOT_OUT = r"C:\Users\kevin\Desktop\thesis_forecasting\outputs_v3\sarima_v3_energy_fixed"

# ============================================================
# SMOOTH-RUNTIME SETTINGS
# ============================================================

MAX_TRAIN_ROWS = {
    "hourly": 20000,
    "daily": 30000,
    "weekly": 30000,
    "monthly": 30000,
    "regular": 30000,
}

FIT_TAIL = {
    "hourly": 1500,   # used only by auto mode; fixed-hourly ignores this
    "daily": 5000,
    "weekly": 5000,
    "monthly": 5000,
    "regular": 5000,
}

REFIT_EVERY = {
    "hourly": 12,     # now used for fixed hourly refit schedule
    "daily": 12,
    "weekly": 8,
    "monthly": 4,
    "regular": 12,
}

MAX_P = 3
MAX_Q = 3
MAX_P_SEAS = 1
MAX_Q_SEAS = 1
MAX_ORDER = 8

# old hourly auto-arima bounds kept for reference but no longer used for energy hourly
HE_MAX_P = 1
HE_MAX_Q = 1
HE_MAX_P_SEAS = 1
HE_MAX_Q_SEAS = 0
HE_MAX_ORDER = 4

# ============================================================
# FIXED HOURLY SARIMA SPEC
# ============================================================

# Main fixed model for hourly energy
# SARIMA(1,0,1)(1,0,0)[24]
HOURLY_ORDER = (1, 0, 1)
HOURLY_SEASONAL_ORDER = (1, 0, 0, 24)

# Fallback if the main one fails
# SARIMA(1,0,0)(1,0,0)[24]
HOURLY_FALLBACK_ORDER = (1, 0, 0)
HOURLY_FALLBACK_SEASONAL_ORDER = (1, 0, 0, 24)

# if you want to cap hourly origins for feasibility, set an integer; else None
MAX_HOURLY_ORIGINS = None


# ============================================================
# HELPERS
# ============================================================

def fit_auto_arima_fast(y_train: np.ndarray, seasonal: bool, m: int, is_hourly_energy: bool) -> pm.ARIMA:
    # hourly energy will not use this anymore, but keep function for non-hourly benchmark continuity
    if seasonal:
        if is_hourly_energy:
            return pm.auto_arima(
                y_train,
                seasonal=True,
                m=int(m),
                stepwise=True,
                stationary=False,
                start_p=0, start_q=0,
                start_P=0, start_Q=0,
                max_p=HE_MAX_P, max_q=HE_MAX_Q,
                max_P=HE_MAX_P_SEAS, max_Q=HE_MAX_Q_SEAS,
                max_order=HE_MAX_ORDER,
                suppress_warnings=True,
                error_action="ignore",
                trace=False,
            )
        return pm.auto_arima(
            y_train,
            seasonal=True,
            m=int(m),
            stepwise=True,
            stationary=False,
            start_p=0, start_q=0,
            start_P=0, start_Q=0,
            max_p=MAX_P, max_q=MAX_Q,
            max_P=MAX_P_SEAS, max_Q=MAX_Q_SEAS,
            max_order=MAX_ORDER,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )

    return pm.auto_arima(
        y_train,
        seasonal=False,
        stepwise=True,
        stationary=False,
        start_p=0, start_q=0,
        max_p=MAX_P, max_q=MAX_Q,
        max_P=0, max_Q=0,
        max_order=MAX_ORDER,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )


def fit_fixed_hourly_sarima(y_train: np.ndarray) -> pm.ARIMA:
    """
    Fixed seasonal SARIMA for hourly energy:
    primary:  SARIMA(1,0,1)(1,0,0)[24]
    fallback: SARIMA(1,0,0)(1,0,0)[24]
    """
    try:
        model = pm.ARIMA(order=HOURLY_ORDER, seasonal_order=HOURLY_SEASONAL_ORDER)
        model.fit(y_train)
        return model
    except Exception:
        model = pm.ARIMA(order=HOURLY_FALLBACK_ORDER, seasonal_order=HOURLY_FALLBACK_SEASONAL_ORDER)
        model.fit(y_train)
        return model


def compute_metrics(preds: pd.DataFrame, denom: float) -> pd.DataFrame:
    rows = []
    for h, sub in preds.groupby("horizon"):
        err = sub["y_true"].to_numpy() - sub["y_pred"].to_numpy()
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mase = float(mae / denom) if np.isfinite(denom) and denom > 0 else np.nan
        rows.append(
            {
                "horizon": int(h),
                "mae": mae,
                "rmse": rmse,
                "mase": mase,
                "n_forecasts": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline() -> None:
    os.makedirs(ROOT_OUT, exist_ok=True)

    panel_path = Path(OUT_BENCHMARK) / "evaluation_panel_v3.csv"
    meta_path = Path(OUT_BENCHMARK) / "benchmark_series_summary_v3.csv"
    if not panel_path.exists() or not meta_path.exists():
        build_and_save_benchmark(ROOT_CLEANED, OUT_BENCHMARK)

    panel = pd.read_csv(panel_path, parse_dates=["origin_ds", "target_ds"])
    meta = pd.read_csv(meta_path)

    preds_all = []
    metrics_all = []
    coverage_rows = []

    root = Path(ROOT_CLEANED)

    for _, mrow in meta.iterrows():
        dataset = mrow["dataset"]
        series_id = mrow["series_id"]
        tag = mrow["series_tag"]

        if not isinstance(tag, str) or not tag:
            continue

        requested = panel[(panel["dataset"] == dataset) & (panel["series_id"] == series_id)].copy()
        requested_rows = len(requested)
        requested_origins = int(requested["origin_idx"].nunique()) if not requested.empty else 0

        # optional hourly-origin cap for large energy series
        if dataset == "energy" and tag == "hourly" and MAX_HOURLY_ORIGINS is not None and not requested.empty:
            keep_origins = sorted(requested["origin_idx"].unique())[-MAX_HOURLY_ORIGINS:]
            requested = requested[requested["origin_idx"].isin(keep_origins)].copy()
            requested_rows = len(requested)
            requested_origins = int(requested["origin_idx"].nunique())

        try:
            folder = find_existing_folder(root, DOMAIN_CONFIGS[dataset].folders)
            if folder is None:
                raise ValueError("Missing domain folder")

            fpath = folder / f"{series_id}.parquet"
            df = load_series(fpath)
            y = df["y"].to_numpy(dtype=np.float32)

            denom = float(mrow["mase_denom"])
            seasonal_m = int(mrow["seasonal_m"])
            seasonal = tag in {"hourly", "daily", "weekly", "monthly"}
            is_hourly_energy = dataset == "energy" and tag == "hourly"

            rows = []
            model = None
            prev_origin = None

            print(f"[RUN] SARIMA {dataset}/{series_id} ({tag}) requested_rows={requested_rows} requested_origins={requested_origins}", flush=True)

            for j, (origin_idx, gp) in enumerate(requested.groupby("origin_idx")):
                origin_idx = int(origin_idx)
                h_max = int(gp["horizon"].max())

                if j % 5 == 0:
                    print(
                        f"  [ORIGIN] {dataset}/{series_id} {j + 1}/{requested_origins} "
                        f"origin_idx={origin_idx} h_max={h_max}",
                        flush=True,
                    )

                fit_start = max(0, origin_idx - MAX_TRAIN_ROWS.get(tag, 30000))
                y_window = y[fit_start:origin_idx]

                if len(y_window) < max(80, 3 * h_max):
                    continue

                do_refit = model is None or (j % max(1, REFIT_EVERY.get(tag, 12)) == 0)

                if is_hourly_energy:
                    # Fixed-order seasonal SARIMA for hourly energy
                    if do_refit:
                        model = fit_fixed_hourly_sarima(y_window)
                    else:
                        if prev_origin is not None:
                            new_obs = y[prev_origin:origin_idx]
                            if len(new_obs):
                                model.update(new_obs)
                else:
                    # Original bounded auto_arima logic for non-hourly series
                    y_fit = y_window[-FIT_TAIL.get(tag, 5000):]
                    if len(y_fit) < max(60, 2 * h_max):
                        continue

                    if do_refit:
                        model = fit_auto_arima_fast(
                            y_train=y_fit,
                            seasonal=seasonal,
                            m=seasonal_m,
                            is_hourly_energy=is_hourly_energy,
                        )
                    else:
                        if prev_origin is not None:
                            new_obs = y[prev_origin:origin_idx]
                            if len(new_obs):
                                model.update(new_obs)

                fc = model.predict(n_periods=h_max)

                for _, prow in gp.iterrows():
                    h = int(prow["horizon"])
                    rows.append(
                        {
                            "dataset": dataset,
                            "series_id": series_id,
                            "series_tag": tag,
                            "model": "SARIMA",
                            "origin_idx": int(prow["origin_idx"]),
                            "origin_ds": prow["origin_ds"],
                            "horizon": h,
                            "target_idx": int(prow["target_idx"]),
                            "target_ds": prow["target_ds"],
                            "y_true": float(prow["y_true"]),
                            "y_pred": float(fc[h - 1]),
                        }
                    )

                prev_origin = origin_idx

            preds = pd.DataFrame(rows)
            if preds.empty:
                raise ValueError("No forecasts generated")

            mets = compute_metrics(preds, denom)
            mets["dataset"] = dataset
            mets["series_id"] = series_id
            mets["series_tag"] = tag
            mets["model"] = "SARIMA"

            preds_all.append(preds)
            metrics_all.append(mets)

            coverage_rows.append(
                {
                    "dataset": dataset,
                    "series_id": series_id,
                    "series_tag": tag,
                    "model": "SARIMA",
                    "requested_rows": requested_rows,
                    "generated_rows": len(preds),
                    "missing_rows": requested_rows - len(preds),
                    "n_origins_requested": requested_origins,
                    "n_origins_generated": int(preds["origin_idx"].nunique()),
                    "fail_reason": "",
                }
            )

            print(f"[OK] SARIMA {dataset}/{series_id} ({tag}) preds={len(preds)}", flush=True)

        except Exception as e:
            coverage_rows.append(
                {
                    "dataset": dataset,
                    "series_id": series_id,
                    "series_tag": tag,
                    "model": "SARIMA",
                    "requested_rows": requested_rows,
                    "generated_rows": 0,
                    "missing_rows": requested_rows,
                    "n_origins_requested": requested_origins,
                    "n_origins_generated": 0,
                    "fail_reason": str(e),
                }
            )
            print(f"[FAIL] SARIMA {dataset}/{series_id} -> {e}", flush=True)

    outp = Path(ROOT_OUT)
    pd.DataFrame(coverage_rows).to_csv(outp / "sarima_coverage_v3.csv", index=False)

    if preds_all:
        preds_df = pd.concat(preds_all, ignore_index=True)
        preds_df.to_csv(outp / "sarima_predictions_v3.csv", index=False)

    if metrics_all:
        metrics_df = pd.concat(metrics_all, ignore_index=True)
        metrics_df.to_csv(outp / "sarima_metrics_per_series_v3.csv", index=False)

        agg = (
            metrics_df.groupby(["dataset", "series_tag", "model", "horizon"])
            .agg(
                mean_mae=("mae", "mean"),
                median_mae=("mae", "median"),
                mean_rmse=("rmse", "mean"),
                median_rmse=("rmse", "median"),
                mean_mase=("mase", "mean"),
                median_mase=("mase", "median"),
                n_series=("series_id", "nunique"),
                n_forecasts_total=("n_forecasts", "sum"),
            )
            .reset_index()
        )
        agg.to_csv(outp / "sarima_metrics_aggregated_v3.csv", index=False)

    print(f"\nDone. Results saved to: {ROOT_OUT}")


if __name__ == "__main__":
    run_pipeline()