from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_design_v3 import (
    ROOT_CLEANED,
    OUT_BENCHMARK,
    DOMAIN_CONFIGS,
    build_and_save_benchmark,
    find_existing_folder,
    load_series,
)

# ============================================================
# OUTPUT
# ============================================================

ROOT_OUT = r"C:\Users\kevin\Desktop\thesis_forecasting\outputs_v3\timegpt_v3"

# ============================================================
# TIMeGPT SETTINGS
# ============================================================

# Use env var only
# set NIXTLA_API_KEY=...
TIMEGPT_ALLOW_HOURLY = True

# IMPORTANT: this restores the old working behavior
TIMEGPT_FORCE_REGULARIZE = True

MAX_TRAIN_ROWS = {
    "hourly": 5000,
    "daily": 3000,
    "weekly": 1500,
    "monthly": 600,
    "regular": 3000,
}

TIMEGPT_MODEL_SHORT = "timegpt-1"
TIMEGPT_MODEL_LONG = "timegpt-1-long-horizon"

LONG_HORIZON_THRESHOLDS = {
    "hourly": 48,
    "daily": 14,
    "weekly": 8,
    "monthly": 6,
    "regular": 14,
}

REDUCE_NIXTLA_LOG_SPAM = True
NIXTLA_LOG_LEVEL = logging.WARNING

# ============================================================
# IMPORTS
# ============================================================

def _nixtla_import():
    try:
        from nixtla import NixtlaClient
        return NixtlaClient
    except Exception:
        return None

def get_api_key() -> str:
    return (os.environ.get("nixak-473d35ab9f9470bf60e50df375316907f4aa2df0a80ef9c30e3efda3996085872d880cce1bb645c5") or "nixak-473d35ab9f9470bf60e50df375316907f4aa2df0a80ef9c30e3efda3996085872d880cce1bb645c5").strip()

# ============================================================
# HELPERS
# ============================================================

def pick_timegpt_model(tag: str, h_max: int) -> str:
    thr = LONG_HORIZON_THRESHOLDS.get(tag, 999999)
    return TIMEGPT_MODEL_LONG if h_max >= thr else TIMEGPT_MODEL_SHORT

def nixtla_freq_for_tag(tag: str) -> str:
    # IMPORTANT: Nixtla in your environment wants lowercase h
    if tag == "hourly":
        return "h"
    if tag == "weekly":
        return "W"
    if tag == "monthly":
        return "M"
    return "D"

def regularize_for_timegpt(train: pd.DataFrame, tag: str) -> pd.DataFrame:
    """
    Restore old working behavior:
    - enforce a regular grid using the frequency implied by the benchmark tag
    - forward fill missing values
    """
    if not TIMEGPT_FORCE_REGULARIZE:
        return train

    freq = nixtla_freq_for_tag(tag)
    s = train.sort_values("ds").drop_duplicates("ds").set_index("ds")
    s = s.asfreq(freq)
    s["y"] = s["y"].ffill()
    s = s.dropna(subset=["y"])
    return s.reset_index()

def timegpt_forecast_one_origin(client, train: pd.DataFrame, tag: str, h_max: int) -> np.ndarray:
    freq = nixtla_freq_for_tag(tag)
    model_name = pick_timegpt_model(tag=tag, h_max=h_max)

    dfin = train.copy()
    dfin["unique_id"] = "s"
    dfin = dfin[["unique_id", "ds", "y"]]

    fc = client.forecast(
        df=dfin,
        h=int(h_max),
        freq=freq,
        model=model_name,
    )

    yhat_col = None
    for c in fc.columns:
        lc = str(c).lower()
        if lc in ("timegpt", "yhat", "forecast"):
            yhat_col = c
            break

    if yhat_col is None:
        num_cols = [c for c in fc.columns if pd.api.types.is_numeric_dtype(fc[c])]
        if not num_cols:
            raise ValueError("TimeGPT forecast output has no numeric prediction column.")
        yhat_col = num_cols[-1]

    yhat = fc[yhat_col].to_numpy(dtype=float)
    if len(yhat) < h_max:
        raise ValueError(f"TimeGPT returned {len(yhat)} predictions for h={h_max}")
    return yhat[:h_max]

def compute_metrics(preds: pd.DataFrame, denom: float) -> pd.DataFrame:
    out = []
    for h, sub in preds.groupby("horizon"):
        err = sub["y_true"].to_numpy() - sub["y_pred"].to_numpy()
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mase = float(mae / denom) if np.isfinite(denom) and denom > 0 else np.nan
        out.append(
            {
                "horizon": int(h),
                "mae": mae,
                "rmse": rmse,
                "mase": mase,
                "n_forecasts": int(len(sub)),
            }
        )
    return pd.DataFrame(out)

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

    if REDUCE_NIXTLA_LOG_SPAM:
        logging.getLogger("nixtla.nixtla_client").setLevel(NIXTLA_LOG_LEVEL)

    NixtlaClient = _nixtla_import()
    if NixtlaClient is None:
        raise RuntimeError("nixtla not installed. Run: pip install nixtla")

    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("Missing NIXTLA_API_KEY environment variable.")

    client = NixtlaClient(api_key=api_key)

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

        if tag == "yearly":
            coverage_rows.append(
                {
                    "dataset": dataset,
                    "series_id": series_id,
                    "series_tag": tag,
                    "model": "TimeGPT",
                    "requested_rows": 0,
                    "generated_rows": 0,
                    "missing_rows": 0,
                    "n_origins_requested": 0,
                    "n_origins_generated": 0,
                    "freq_used": "",
                    "fail_reason": "yearly skipped",
                }
            )
            continue

        if tag == "hourly" and not TIMEGPT_ALLOW_HOURLY:
            coverage_rows.append(
                {
                    "dataset": dataset,
                    "series_id": series_id,
                    "series_tag": tag,
                    "model": "TimeGPT",
                    "requested_rows": 0,
                    "generated_rows": 0,
                    "missing_rows": 0,
                    "n_origins_requested": 0,
                    "n_origins_generated": 0,
                    "freq_used": "",
                    "fail_reason": "hourly disabled",
                }
            )
            continue

        folder = find_existing_folder(root, DOMAIN_CONFIGS[dataset].folders)
        if folder is None:
            continue

        fpath = folder / f"{series_id}.parquet"
        panel_s = panel[(panel["dataset"] == dataset) & (panel["series_id"] == series_id)].copy()

        requested_rows = len(panel_s)
        requested_origins = int(panel_s["origin_idx"].nunique()) if not panel_s.empty else 0
        freq_used = nixtla_freq_for_tag(tag)

        try:
            df = load_series(fpath)
            denom = float(mrow["mase_denom"])

            rows = []

            for origin_idx, gp in panel_s.groupby("origin_idx"):
                origin_idx = int(origin_idx)
                h_max = int(gp["horizon"].max())

                train = df.iloc[:origin_idx].copy()

                if len(train) > MAX_TRAIN_ROWS.get(tag, 3000):
                    train = train.iloc[-MAX_TRAIN_ROWS.get(tag, 3000):].reset_index(drop=True)

                train = regularize_for_timegpt(train, tag=tag)

                # same minimum-history logic as before
                if tag == "monthly" and len(train) < 12:
                    continue
                if tag != "monthly" and len(train) < 50:
                    continue

                yhat = timegpt_forecast_one_origin(
                    client=client,
                    train=train,
                    tag=tag,
                    h_max=h_max,
                )

                model_used = pick_timegpt_model(tag=tag, h_max=h_max)

                for _, prow in gp.iterrows():
                    h = int(prow["horizon"])
                    rows.append(
                        {
                            "dataset": dataset,
                            "series_id": series_id,
                            "series_tag": tag,
                            "model": model_used,
                            "origin_idx": int(prow["origin_idx"]),
                            "origin_ds": prow["origin_ds"],
                            "horizon": h,
                            "target_idx": int(prow["target_idx"]),
                            "target_ds": prow["target_ds"],
                            "y_true": float(prow["y_true"]),
                            "y_pred": float(yhat[h - 1]),
                        }
                    )

            preds = pd.DataFrame(rows)
            if preds.empty:
                raise ValueError("No TimeGPT forecasts generated.")

            mets = compute_metrics(preds, denom=denom)
            mets["dataset"] = dataset
            mets["series_id"] = series_id
            mets["series_tag"] = tag
            mets["model"] = "TimeGPT"

            preds_all.append(preds)
            metrics_all.append(mets)

            coverage_rows.append(
                {
                    "dataset": dataset,
                    "series_id": series_id,
                    "series_tag": tag,
                    "model": "TimeGPT",
                    "requested_rows": requested_rows,
                    "generated_rows": len(preds),
                    "missing_rows": requested_rows - len(preds),
                    "n_origins_requested": requested_origins,
                    "n_origins_generated": int(preds["origin_idx"].nunique()),
                    "freq_used": freq_used,
                    "fail_reason": "",
                }
            )
            print(f"[OK] TimeGPT {dataset}/{series_id} ({tag}) preds={len(preds)}")

        except Exception as e:
            coverage_rows.append(
                {
                    "dataset": dataset,
                    "series_id": series_id,
                    "series_tag": tag,
                    "model": "TimeGPT",
                    "requested_rows": requested_rows,
                    "generated_rows": 0,
                    "missing_rows": requested_rows,
                    "n_origins_requested": requested_origins,
                    "n_origins_generated": 0,
                    "freq_used": freq_used,
                    "fail_reason": str(e),
                }
            )
            print(f"[FAIL] TimeGPT {dataset}/{series_id} -> {e}")

    outp = Path(ROOT_OUT)
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(outp / "timegpt_coverage_v3.csv", index=False)

    if preds_all:
        preds_df = pd.concat(preds_all, ignore_index=True)
        preds_df.to_csv(outp / "timegpt_predictions_v3.csv", index=False)

    if metrics_all:
        metrics_df = pd.concat(metrics_all, ignore_index=True)
        metrics_df.to_csv(outp / "timegpt_metrics_per_series_v3.csv", index=False)

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
        agg.to_csv(outp / "timegpt_metrics_aggregated_v3.csv", index=False)

    print(f"\nDone. Results saved to: {ROOT_OUT}")

if __name__ == "__main__":
    run_pipeline()