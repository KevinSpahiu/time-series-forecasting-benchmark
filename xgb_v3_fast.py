from __future__ import annotations

import os
import warnings
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

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT_OUT = r"C:\Users\kevin\Desktop\thesis_forecasting\outputs_v3\xgb_v3_fast"

MAX_TRAIN_ROWS = {
    "hourly": 50000,
    "daily": 30000,
    "weekly": 30000,
    "monthly": 30000,
    "regular": 30000,
}
LAGS = {"hourly": 72, "daily": 30, "weekly": 24, "monthly": 24, "regular": 30}
REFIT_EVERY = {"hourly": 14, "daily": 7, "weekly": 4, "monthly": 3, "regular": 7}

XGB_N_ESTIMATORS = 250
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.9
XGB_COLSAMPLE = 0.9
XGB_RANDOM_STATE = 42


def _xgb_import():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor
    except Exception:
        return None


def make_supervised_1step(y: np.ndarray, lags: int):
    X, t = [], []
    for i in range(lags, len(y)):
        X.append(y[i - lags:i])
        t.append(y[i])
    return np.asarray(X, dtype=np.float32), np.asarray(t, dtype=np.float32)


def recursive_forecast_1step(model, history: np.ndarray, lags: int, h_max: int) -> np.ndarray:
    hist = history.astype(np.float32).tolist()
    preds = []
    for _ in range(h_max):
        x = np.asarray(hist[-lags:], dtype=np.float32).reshape(1, -1)
        yhat = float(model.predict(x)[0])
        preds.append(yhat)
        hist.append(yhat)
    return np.asarray(preds, dtype=np.float32)


def compute_metrics(preds: pd.DataFrame, denom: float) -> pd.DataFrame:
    rows = []
    for h, sub in preds.groupby("horizon"):
        err = sub["y_true"].to_numpy() - sub["y_pred"].to_numpy()
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mase = float(mae / denom) if np.isfinite(denom) and denom > 0 else np.nan
        rows.append({"horizon": int(h), "mae": mae, "rmse": rmse, "mase": mase, "n_forecasts": int(len(sub))})
    return pd.DataFrame(rows)


def run_pipeline() -> None:
    os.makedirs(ROOT_OUT, exist_ok=True)

    XGBRegressor = _xgb_import()
    if XGBRegressor is None:
        raise RuntimeError("xgboost not installed. Run: pip install xgboost")

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

        try:
            folder = find_existing_folder(root, DOMAIN_CONFIGS[dataset].folders)
            if folder is None:
                raise ValueError("Missing domain folder")
            fpath = folder / f"{series_id}.parquet"
            df = load_series(fpath)
            y = df["y"].to_numpy(dtype=float)
            denom = float(mrow["mase_denom"])
            lags = LAGS.get(tag, 30)
            refit_every = REFIT_EVERY.get(tag, 7)
            max_train_rows = MAX_TRAIN_ROWS.get(tag, 30000)

            rows = []
            model = None
            for j, (origin_idx, gp) in enumerate(requested.groupby("origin_idx")):
                origin_idx = int(origin_idx)
                h_max = int(gp["horizon"].max())

                start_idx = max(0, origin_idx - max_train_rows)
                y_train = y[start_idx:origin_idx].astype(np.float32)
                if len(y_train) < lags + 80:
                    continue

                if model is None or (j % max(1, refit_every) == 0):
                    X, t = make_supervised_1step(y_train, lags)
                    if len(X) < 10:
                        continue
                    model = XGBRegressor(
                        n_estimators=XGB_N_ESTIMATORS,
                        max_depth=XGB_MAX_DEPTH,
                        learning_rate=XGB_LEARNING_RATE,
                        subsample=XGB_SUBSAMPLE,
                        colsample_bytree=XGB_COLSAMPLE,
                        objective="reg:squarederror",
                        random_state=XGB_RANDOM_STATE,
                        n_jobs=1,
                    )
                    model.fit(X, t)

                history = y_train
                if len(history) < lags or model is None:
                    continue

                fc = recursive_forecast_1step(model, history, lags, h_max)
                for _, prow in gp.iterrows():
                    h = int(prow["horizon"])
                    rows.append(
                        {
                            "dataset": dataset,
                            "series_id": series_id,
                            "series_tag": tag,
                            "model": "XGBoost",
                            "origin_idx": int(prow["origin_idx"]),
                            "origin_ds": prow["origin_ds"],
                            "horizon": h,
                            "target_idx": int(prow["target_idx"]),
                            "target_ds": prow["target_ds"],
                            "y_true": float(prow["y_true"]),
                            "y_pred": float(fc[h - 1]),
                        }
                    )

            preds = pd.DataFrame(rows)
            if preds.empty:
                raise ValueError("No forecasts generated")

            mets = compute_metrics(preds, denom)
            mets["dataset"] = dataset
            mets["series_id"] = series_id
            mets["series_tag"] = tag
            mets["model"] = "XGBoost"

            preds_all.append(preds)
            metrics_all.append(mets)
            coverage_rows.append(
                {
                    "dataset": dataset,
                    "series_id": series_id,
                    "series_tag": tag,
                    "model": "XGBoost",
                    "requested_rows": requested_rows,
                    "generated_rows": len(preds),
                    "missing_rows": requested_rows - len(preds),
                    "n_origins_requested": requested_origins,
                    "n_origins_generated": int(preds["origin_idx"].nunique()),
                    "fail_reason": "",
                }
            )
            print(f"[OK] XGBoost {dataset}/{series_id} ({tag}) preds={len(preds)}")
        except Exception as e:
            coverage_rows.append(
                {
                    "dataset": dataset,
                    "series_id": series_id,
                    "series_tag": tag,
                    "model": "XGBoost",
                    "requested_rows": requested_rows,
                    "generated_rows": 0,
                    "missing_rows": requested_rows,
                    "n_origins_requested": requested_origins,
                    "n_origins_generated": 0,
                    "fail_reason": str(e),
                }
            )
            print(f"[FAIL] XGBoost {dataset}/{series_id} -> {e}")

    outp = Path(ROOT_OUT)
    pd.DataFrame(coverage_rows).to_csv(outp / "xgb_coverage_v3.csv", index=False)

    if preds_all:
        preds_df = pd.concat(preds_all, ignore_index=True)
        preds_df.to_csv(outp / "xgb_predictions_v3.csv", index=False)

    if metrics_all:
        metrics_df = pd.concat(metrics_all, ignore_index=True)
        metrics_df.to_csv(outp / "xgb_metrics_per_series_v3.csv", index=False)
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
        agg.to_csv(outp / "xgb_metrics_aggregated_v3.csv", index=False)

    print(f"\nDone. Results saved to: {ROOT_OUT}")


if __name__ == "__main__":
    run_pipeline()
