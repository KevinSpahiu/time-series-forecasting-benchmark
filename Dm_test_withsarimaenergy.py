from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUTPUTS_V3 = Path(r"C:\Users\kevin\Desktop\thesis_forecasting\outputs_v3")
RESULTS_DIR = OUTPUTS_V3 / "dm_tests_v3_energy_fixed"

MODEL_FILES = {
    "SARIMA": OUTPUTS_V3 / "sarima_v3_energy_fixed" / "sarima_predictions_v3.csv",
    "XGBoost": OUTPUTS_V3 / "xgb_v3" / "xgb_predictions_v3.csv",
    "LSTM": OUTPUTS_V3 / "lstm_v3" / "lstm_predictions_v3.csv",
    "TimeGPT": OUTPUTS_V3 / "timegpt_v3" / "timegpt_predictions_v3.csv",
}

KEY_COLS = [
    "dataset",
    "series_id",
    "series_tag",
    "origin_idx",
    "origin_ds",
    "horizon",
    "target_idx",
    "target_ds",
    "y_true",
]


def long_run_variance_bartlett(d: np.ndarray, lag: int) -> float:
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    T = len(d)
    if T < 3:
        return np.nan

    d0 = d - d.mean()
    gamma0 = np.dot(d0, d0) / T

    if lag <= 0:
        return gamma0

    lrv = gamma0
    max_k = min(lag, T - 1)
    for k in range(1, max_k + 1):
        w = 1.0 - (k / (lag + 1.0))
        cov = np.dot(d0[k:], d0[:-k]) / T
        lrv += 2.0 * w * cov
    return lrv


def dm_test(d: np.ndarray, h: int) -> dict:
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    T = len(d)
    if T < 10:
        return {"T": T, "DM_stat": np.nan, "p_value": np.nan, "mean_d": np.nan}

    h = int(h)
    lag = max(h - 1, 0)
    lrv = long_run_variance_bartlett(d, lag)

    if not np.isfinite(lrv) or lrv <= 0:
        return {
            "T": T,
            "DM_stat": np.nan,
            "p_value": np.nan,
            "mean_d": float(np.nanmean(d)),
        }

    mean_d = float(d.mean())
    dm_raw = mean_d / math.sqrt(lrv / T)

    factor_term = (T + 1 - 2 * h + (h * (h - 1) / T)) / T
    if factor_term <= 0:
        return {
            "T": T,
            "DM_stat": np.nan,
            "p_value": np.nan,
            "mean_d": mean_d,
        }

    factor = math.sqrt(factor_term)
    dm_hln = dm_raw * factor
    p = 2.0 * (1.0 - stats.t.cdf(abs(dm_hln), df=T - 1))

    return {"T": T, "DM_stat": float(dm_hln), "p_value": float(p), "mean_d": mean_d}


def load_preds(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file for {model_name}: {path}")

    df = pd.read_csv(path, parse_dates=["origin_ds", "target_ds"])
    df.columns = [c.strip().lower() for c in df.columns]

    required = {c.lower() for c in KEY_COLS} | {"y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{model_name} missing columns {sorted(missing)} in {path}")

    out = df[[c.lower() for c in KEY_COLS] + ["y_pred"]].copy()
    out["model"] = model_name

    out["dataset"] = out["dataset"].astype(str)
    out["series_id"] = out["series_id"].astype(str)
    out["series_tag"] = out["series_tag"].astype(str)

    out["horizon"] = pd.to_numeric(out["horizon"], errors="coerce").astype("Int64")
    out["origin_idx"] = pd.to_numeric(out["origin_idx"], errors="coerce").astype("Int64")
    out["target_idx"] = pd.to_numeric(out["target_idx"], errors="coerce").astype("Int64")
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")

    out = out.dropna(
        subset=[
            "dataset",
            "series_id",
            "series_tag",
            "origin_idx",
            "origin_ds",
            "horizon",
            "target_idx",
            "target_ds",
            "y_true",
            "y_pred",
        ]
    ).copy()

    return out


def compute_loss(y_true: pd.Series, y_pred: pd.Series, loss: str) -> pd.Series:
    e = y_true - y_pred
    if loss == "ae":
        return e.abs()
    if loss == "se":
        return e.pow(2)
    raise ValueError("loss must be 'ae' or 'se'")


def build_overlap_table(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    models = sorted(preds["model"].unique())

    for m1, m2 in itertools.combinations(models, 2):
        df1 = preds[preds["model"] == m1][KEY_COLS].copy()
        df2 = preds[preds["model"] == m2][KEY_COLS].copy()

        merged = df1.merge(df2, on=KEY_COLS, how="inner")
        if merged.empty:
            continue

        summary = (
            merged.groupby(["dataset", "series_id", "series_tag", "horizon"], dropna=False)
            .size()
            .reset_index(name="overlap_count")
        )
        summary["model_A"] = m1
        summary["model_B"] = m2
        rows.append(summary)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_dm(preds: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope not in {"dataset", "series"}:
        raise ValueError("scope must be 'dataset' or 'series'")

    group_keys = (
        ["dataset", "series_tag", "horizon"]
        if scope == "dataset"
        else ["dataset", "series_id", "series_tag", "horizon"]
    )

    wide = preds.pivot_table(
        index=KEY_COLS,
        columns="model",
        values="y_pred",
        aggfunc="first",
        observed=False,
    ).reset_index()

    model_cols = [c for c in wide.columns if c not in KEY_COLS]
    results = []

    for loss_code, loss_name in [("ae", "absolute_error"), ("se", "squared_error")]:
        for m1, m2 in itertools.combinations(model_cols, 2):
            sub = wide.dropna(subset=[m1, m2]).copy()
            if sub.empty:
                continue

            L1 = compute_loss(sub["y_true"], sub[m1], loss_code)
            L2 = compute_loss(sub["y_true"], sub[m2], loss_code)
            sub["d"] = (L1 - L2).astype(float)

            for gkey, g in sub.groupby(group_keys, dropna=False):
                if not isinstance(gkey, tuple):
                    gkey = (gkey,)

                h = int(pd.Series(g["horizon"]).iloc[0])
                test = dm_test(g["d"].to_numpy(), h)
                mean_d = test["mean_d"]
                winner = m1 if mean_d < 0 else (m2 if mean_d > 0 else "tie")

                row = dict(zip(group_keys, gkey))
                row.update(
                    {
                        "loss": loss_name,
                        "model_A": m1,
                        "model_B": m2,
                        "mean_loss_diff_A_minus_B": mean_d,
                        "T": test["T"],
                        "DM_stat": test["DM_stat"],
                        "p_value": test["p_value"],
                        "winner_by_mean": winner,
                    }
                )
                results.append(row)

    return pd.DataFrame(results)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    preds_all = []
    audit_rows = []

    print("Loading prediction files...", flush=True)

    for model, path in MODEL_FILES.items():
        print(f"  [LOAD] {model}: {path}", flush=True)
        preds = load_preds(path, model)
        preds_all.append(preds)
        audit_rows.append(
            {
                "model": model,
                "file": str(path),
                "n_rows": len(preds),
                "n_series": int(preds["series_id"].nunique()),
                "n_datasets": int(preds["dataset"].nunique()),
            }
        )

    pd.DataFrame(audit_rows).to_csv(RESULTS_DIR / "dm_input_files_used_v3.csv", index=False)

    preds = pd.concat(preds_all, ignore_index=True)
    preds.to_csv(RESULTS_DIR / "all_model_predictions_v3.csv", index=False)

    print("Building overlap table...", flush=True)
    overlap = build_overlap_table(preds)
    overlap.to_csv(RESULTS_DIR / "dm_overlap_counts_v3.csv", index=False)

    print("Building alignment coverage summary...", flush=True)
    cover = (
        preds.pivot_table(index=KEY_COLS, columns="model", values="y_pred", aggfunc="first", observed=False)
        .notna()
        .reset_index()
    )
    model_cols = [c for c in cover.columns if c not in KEY_COLS]
    cover["n_models_present"] = cover[model_cols].sum(axis=1)

    cover_summary = (
        cover.groupby(["dataset", "series_tag", "horizon"])["n_models_present"]
        .value_counts()
        .rename("count")
        .reset_index()
        .sort_values(["dataset", "series_tag", "horizon", "n_models_present"])
    )
    cover_summary.to_csv(RESULTS_DIR / "alignment_coverage_by_dataset_tag_horizon_v3.csv", index=False)

    print("Running dataset-level DM tests...", flush=True)
    dm_dataset = run_dm(preds, scope="dataset")
    dm_dataset.to_csv(RESULTS_DIR / "dm_results_by_dataset_v3.csv", index=False)

    print("Running series-level DM tests...", flush=True)
    dm_series = run_dm(preds, scope="series")
    dm_series.to_csv(RESULTS_DIR / "dm_results_by_series_v3.csv", index=False)

    if not dm_dataset.empty:
        dm_dataset[dm_dataset["p_value"] < 0.05].to_csv(
            RESULTS_DIR / "dm_results_by_dataset_significant_0.05_v3.csv",
            index=False,
        )

    if not dm_series.empty:
        dm_series[dm_series["p_value"] < 0.05].to_csv(
            RESULTS_DIR / "dm_results_by_series_significant_0.05_v3.csv",
            index=False,
        )

    print(f"Done. DM results saved to: {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()