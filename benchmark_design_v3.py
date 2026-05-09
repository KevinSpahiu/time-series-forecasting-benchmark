from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================
# BENCHMARK SETTINGS
# ============================================================

ROOT_CLEANED = r"C:\Users\kevin\Desktop\thesis_forecasting\data\datasets_cleaned"
OUT_BENCHMARK = r"C:\Users\kevin\Desktop\thesis_forecasting\outputs_v3\benchmark"

SKIP_YEARLY = True

# unified horizons across all models
HORIZONS_HOURLY = [1, 24, 168]
HORIZONS_DAILY = [1, 7, 30]
HORIZONS_WEEKLY = [1, 4, 12]
HORIZONS_MONTHLY = [1, 3, 6]

# unified step candidates
STEP_CANDIDATES = {
    "hourly": [24, 12, 1],
    "daily": [7, 3, 1],
    "weekly": [1],
    "monthly": [1],
    "regular": [7, 3, 1],
}

# unified origin caps
ORIGIN_CAPS = {
    "hourly": 365,
    "daily": 730,
    "weekly": 260,
    "monthly": 120,
    "regular": 730,
}

MIN_ORIGINS_TARGET = 5
MIN_ORIGINS_FALLBACK = 3

SEASONAL_M = {
    "hourly": 24,
    "daily": 7,
    "weekly": 52,
    "monthly": 12,
    "regular": 7,
    "yearly": 1,
}

# ============================================================
# DOMAIN CONFIG
# ============================================================

@dataclass(frozen=True)
class DomainConfig:
    folders: List[str]

DOMAIN_CONFIGS: Dict[str, DomainConfig] = {
    "stock": DomainConfig(folders=["stock", "stocks"]),
    "sales": DomainConfig(folders=["sales"]),
    "energy": DomainConfig(folders=["energy"]),
}

# ============================================================
# HELPERS
# ============================================================

def find_existing_folder(root: Path, options: List[str]) -> Optional[Path]:
    for name in options:
        p = root / name
        if p.exists() and p.is_dir():
            return p
    return None

def load_series(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not {"ds", "y"}.issubset(df.columns):
        raise ValueError(f"{path.name}: expected columns ['ds','y'], got {list(df.columns)}")

    df = df[["ds", "y"]].copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df = (
        df.dropna(subset=["ds", "y"])
        .sort_values("ds")
        .drop_duplicates("ds", keep="last")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(f"{path.name}: empty after parsing ds/y")
    return df

def infer_tag(df: pd.DataFrame) -> str:
    ds = df["ds"].sort_values()
    if len(ds) < 10:
        return "regular"

    deltas = ds.diff().dropna()
    if deltas.empty:
        return "regular"

    med = deltas.median()

    if med >= pd.Timedelta(days=300):
        return "yearly"
    if pd.Timedelta(days=25) <= med <= pd.Timedelta(days=35):
        return "monthly"
    if pd.Timedelta(days=5) <= med <= pd.Timedelta(days=10):
        return "weekly"
    if pd.Timedelta(hours=12) <= med <= pd.Timedelta(days=2):
        return "daily"
    if med <= pd.Timedelta(hours=2):
        return "hourly"
    return "regular"

def horizons_for_tag(tag: str) -> List[int]:
    if tag == "hourly":
        return HORIZONS_HOURLY
    if tag == "weekly":
        return HORIZONS_WEEKLY
    if tag == "monthly":
        return HORIZONS_MONTHLY
    return HORIZONS_DAILY

def seasonal_m_for_tag(tag: str) -> int:
    return SEASONAL_M.get(tag, 7)

def split_indices(n: int, h_max: int) -> Tuple[int, int]:
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    if n >= 120:
        train_end = max(train_end, 60)
        val_end = max(val_end, train_end + 20)
    else:
        train_end = max(train_end, max(10, int(0.50 * n)))
        val_end = max(val_end, train_end + 10)

    val_end = min(val_end, n - (h_max + 5))
    return train_end, val_end

def build_origins(test_start: int, n: int, h_max: int, tag: str) -> Tuple[List[int], int]:
    cap = ORIGIN_CAPS.get(tag, 730)
    candidates = STEP_CANDIDATES.get(tag, [7, 3, 1])

    def _build(step: int) -> List[int]:
        origins = list(range(test_start, n - h_max + 1, step))
        if len(origins) > cap:
            origins = origins[-cap:]
        return origins

    for step in candidates:
        origins = _build(step)
        if len(origins) >= MIN_ORIGINS_TARGET:
            return origins, step

    origins = _build(candidates[-1])
    return origins, candidates[-1]

def mase_denom(y_train: np.ndarray, m: int) -> float:
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) > m:
        d = np.abs(y_train[m:] - y_train[:-m])
    else:
        d = np.abs(np.diff(y_train))
    denom = float(np.mean(d)) if len(d) else np.nan
    return denom if np.isfinite(denom) and denom > 0 else np.nan

def build_evaluation_panel_for_series(
    df: pd.DataFrame,
    dataset: str,
    series_id: str,
) -> Tuple[pd.DataFrame, dict]:
    tag = infer_tag(df)

    if SKIP_YEARLY and tag == "yearly":
        raise ValueError("Skipped yearly series")

    horizons = horizons_for_tag(tag)
    h_max = max(horizons)

    n = len(df)
    train_end, val_end = split_indices(n, h_max)
    test_start = val_end

    if test_start >= n - h_max:
        raise ValueError(f"Too short for horizons {horizons}: n={n}")

    origins, step_used = build_origins(test_start, n, h_max, tag)
    if len(origins) < MIN_ORIGINS_FALLBACK:
        raise ValueError(f"Too few origins after fallback: {len(origins)}")

    ds = df["ds"].to_numpy()
    y = df["y"].to_numpy(dtype=float)

    rows = []
    for origin_idx in origins:
        origin_ds = ds[origin_idx]
        for h in horizons:
            target_idx = origin_idx + (h - 1)
            if target_idx >= n:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "series_id": series_id,
                    "series_tag": tag,
                    "origin_idx": int(origin_idx),
                    "origin_ds": origin_ds,
                    "horizon": int(h),
                    "target_idx": int(target_idx),
                    "target_ds": ds[target_idx],
                    "y_true": float(y[target_idx]),
                }
            )

    panel = pd.DataFrame(rows)
    if panel.empty:
        raise ValueError("No panel rows generated.")

    meta = {
        "dataset": dataset,
        "series_id": series_id,
        "series_tag": tag,
        "n_rows": int(n),
        "train_end": int(train_end),
        "val_end": int(val_end),
        "test_start": int(test_start),
        "n_origins": int(len(origins)),
        "step_used": int(step_used),
        "seasonal_m": int(seasonal_m_for_tag(tag)),
        "mase_denom": float(mase_denom(y[:train_end], seasonal_m_for_tag(tag))),
    }
    return panel, meta

def build_and_save_benchmark(root_cleaned: str = ROOT_CLEANED, out_dir: str = OUT_BENCHMARK) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(out_dir, exist_ok=True)
    root = Path(root_cleaned)

    panels = []
    metas = []

    for domain, cfg in DOMAIN_CONFIGS.items():
        folder = find_existing_folder(root, cfg.folders)
        if folder is None:
            print(f"[SKIP] missing folder for '{domain}': tried {cfg.folders}")
            continue

        files = sorted(folder.glob("*.parquet"))
        if not files:
            print(f"[SKIP] no parquet files in: {folder}")
            continue

        for f in files:
            try:
                df = load_series(f)
                panel, meta = build_evaluation_panel_for_series(df, dataset=domain, series_id=f.stem)
                panels.append(panel)
                metas.append(meta)
                print(f"[OK] panel {domain}/{f.stem} ({meta['series_tag']}) rows={len(panel)} origins={meta['n_origins']}")
            except Exception as e:
                print(f"[FAIL] panel {domain}/{f.stem} -> {e}")
                metas.append(
                    {
                        "dataset": domain,
                        "series_id": f.stem,
                        "series_tag": "",
                        "n_rows": 0,
                        "train_end": np.nan,
                        "val_end": np.nan,
                        "test_start": np.nan,
                        "n_origins": 0,
                        "step_used": np.nan,
                        "seasonal_m": np.nan,
                        "mase_denom": np.nan,
                        "error": str(e),
                    }
                )

    panel_df = pd.concat(panels, ignore_index=True) if panels else pd.DataFrame()
    meta_df = pd.DataFrame(metas)

    panel_path = Path(out_dir) / "evaluation_panel_v3.csv"
    meta_path = Path(out_dir) / "benchmark_series_summary_v3.csv"

    panel_df.to_csv(panel_path, index=False)
    meta_df.to_csv(meta_path, index=False)

    print(f"\nSaved benchmark panel to: {panel_path}")
    print(f"Saved series summary to: {meta_path}")
    return panel_df, meta_df

if __name__ == "__main__":
    build_and_save_benchmark()