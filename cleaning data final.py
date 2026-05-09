from __future__ import annotations

"""
UNIFIED CLEANING PIPELINE -> creates datasets_cleaned/ mirroring datasets/

Now supports your NEW energy schemas, including hourly load files like:
  Datetime,EKPC_MW
  2013-12-31 01:00:00,1861.0
  ...

Key upgrades:
- Energy inference supports:
  - Datetime + *_MW columns (EKPC_MW, PJME_MW, etc.)
  - ENTSO-E style datasets (time + total load actual / price actual etc.)
  - Appliances dataset (date + Appliances)
- ENERGY_OVERRIDES for deterministic mapping where needed.
- Auto-detect hourly vs daily vs business vs yearly frequency from timestamps and
  set appropriate freq + seasonal_m + missing policy.
- Robust datetime + numeric cleaning.
- Parquet output (fallback to CSV if parquet engine not installed).
- Manifest: datasets_cleaned/cleaning_manifest.csv
"""

import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# =========================
# Warnings
# =========================
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings(
    "ignore",
    message=r"Parsing dates in .* format when dayfirst=.* was specified.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Could not infer format, so each element will be parsed individually.*",
    category=UserWarning,
)

# =========================
# User knobs
# =========================
PREFERRED_FORMAT = "parquet"     # "parquet" or "csv"
PREFER_DAYFIRST = True          # EU default; smart parser will fallback if needed

# Minimum lengths
MIN_N_DEFAULT = 100             # typical daily/hourly series
MIN_N_YEARLY = 20               # yearly series

# Stock log-returns guardrail
MIN_POS_FOR_LOGRET = 100

# =========================
# Per-file overrides (stem -> (ds_col, y_col))
# Use these when "best target" is subjective or inference could pick the wrong thing.
# =========================
ENERGY_OVERRIDES: Dict[str, Tuple[str, str]] = {
    # Hourly load files
    "EKPC_hourly": ("Datetime", "EKPC_MW"),
    "FE_hourly": ("Datetime", "FE_MW"),
    "NI_hourly": ("Datetime", "NI_MW"),
    "PJM_Load_hourly": ("Datetime", "PJM_Load_MW"),
    "PJME_hourly": ("Datetime", "PJME_MW"),
    "PJMW_hourly": ("Datetime", "PJMW_MW"),

    # ENTSO-E style: pick ONE target
    "energy_dataset": ("time", "total load actual"),

    # Appliances dataset
    "KAG_energydata_complete": ("date", "Appliances"),
}

# =========================
# Config dataclasses
# =========================

@dataclass(frozen=True)
class MissingPolicy:
    mode: str                 # "ffill" or "drop"
    max_gap: Optional[int]


@dataclass(frozen=True)
class DomainConfig:
    folders: List[str]
    freq: str                 # default/fallback
    seasonal_m: int
    missing: MissingPolicy
    reindex_full_grid: bool
    use_log_returns: bool
    ds_col: Optional[str] = None
    y_col: Optional[str] = None


DOMAIN_CONFIGS: Dict[str, DomainConfig] = {
    "stock": DomainConfig(
        folders=["stocks", "stock"],
        freq="B",
        seasonal_m=5,
        missing=MissingPolicy("drop", None),
        reindex_full_grid=False,
        use_log_returns=True,
    ),
    "sales": DomainConfig(
        folders=["sales"],
        freq="D",
        seasonal_m=7,
        missing=MissingPolicy("ffill", 3),
        reindex_full_grid=True,
        use_log_returns=False,
    ),
    "energy": DomainConfig(
        folders=["energy"],
        freq="D",  # will be overridden by auto frequency detection
        seasonal_m=7,
        missing=MissingPolicy("ffill", 6),
        reindex_full_grid=True,
        use_log_returns=False,
    ),
}

# =========================
# Helpers
# =========================

def _norm(s: str) -> str:
    return s.strip().lower()


def parquet_supported() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False


def _clean_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    txt = s.astype(str).str.replace(",", "", regex=False)
    extracted = txt.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def read_csv_smart(path: str) -> pd.DataFrame:
    """
    Robust CSV reader:
    - normal read
    - if 1 col with semicolons -> sep=";"
    - fallback: sep=None, python engine
    """
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            col0 = str(df.columns[0])
            if ";" in col0:
                df = pd.read_csv(path, sep=";")
        return df
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")


def infer_ds_y(domain: str, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = list(df.columns)
    norm_map = {_norm(c): c for c in cols}

    ds_candidates = (
        "ds", "date", "datetime", "timestamp", "time",
        "orderdate", "order date", "saledate", "sale date",
        "year",
    )

    if domain == "stock":
        ds_col = next((norm_map[k] for k in ds_candidates if k in norm_map), None)
        y_candidates = ("close/last", "adj close", "adjclose", "close", "price", "last")
        y_col = next((norm_map[k] for k in y_candidates if k in norm_map), None)
        return ds_col, y_col

    if domain == "sales":
        ds_col = next((norm_map[k] for k in ds_candidates if k in norm_map), None)
        y_candidates = (
            "sales", "sale price", "sellingprice", "revenue",
            "weekly_sales", "weekly sales", "retail sales", "warehouse sales",
            "units sold", "inventory level", "demand forecast", "units ordered",
            "quantity", "qty",
        )
        y_col = next((norm_map[k] for k in y_candidates if k in norm_map), None)
        return ds_col, y_col

    if domain == "energy":
        ds_col = next((norm_map[k] for k in ds_candidates if k in norm_map), None)

        # Strong targets (if present)
        y_priority = (
            "ekpc_mw", "fe_mw", "ni_mw", "pjme_mw", "pjmw_mw", "pjm_load_mw",
            "total load actual", "total load forecast",
            "price actual", "price day ahead",
            "electricity_demand", "electricity demand",
            "load", "demand", "energy_price", "price",
            "appliances", "lights",
        )
        y_col = next((norm_map[k] for k in y_priority if k in norm_map), None)

        # Fallback: any column containing MW or ending in _MW
        if y_col is None:
            for c in cols:
                cn = _norm(c)
                if cn.endswith("_mw") or " mw" in cn or cn.endswith("mw"):
                    y_col = c
                    break

        return ds_col, y_col

    return None, None


def _detect_yearly_like(ds: pd.Series) -> bool:
    if ds.dtype.kind in "iu":
        return True
    s = ds.astype(str).str.strip()
    if len(s) == 0:
        return False
    return float(s.str.fullmatch(r"\d{4}").mean()) >= 0.8


def _coerce_year_only_dates(ds: pd.Series) -> pd.Series:
    if ds.dtype.kind in "iu":
        return ds.astype(str) + "-01-01"
    s = ds.astype(str).str.strip()
    only_year = s.str.fullmatch(r"\d{4}")
    return s.where(~only_year, s + "-01-01")


def _parse_datetime_smart(ds: pd.Series, prefer_dayfirst: bool) -> pd.Series:
    ds_raw = ds.copy()

    dt1 = pd.to_datetime(ds_raw, errors="coerce", dayfirst=prefer_dayfirst, utc=True)
    ok1 = float(dt1.notna().mean()) if len(dt1) else 0.0
    if ok1 >= 0.90:
        return dt1.dt.tz_convert(None)

    dt2 = pd.to_datetime(ds_raw, errors="coerce", dayfirst=not prefer_dayfirst, utc=True)
    ok2 = float(dt2.notna().mean()) if len(dt2) else 0.0

    best = dt1 if ok1 >= ok2 else dt2
    return best.dt.tz_convert(None)


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if _detect_yearly_like(df["ds"]):
        df["ds"] = _coerce_year_only_dates(df["ds"])

    df["ds"] = _parse_datetime_smart(df["ds"], prefer_dayfirst=PREFER_DAYFIRST)
    df["y"] = _clean_numeric_series(df["y"])

    df = df.dropna(subset=["ds"])
    df = df.sort_values("ds")
    df = df.drop_duplicates("ds", keep="last")
    return df.reset_index(drop=True)


def infer_time_config_from_ds(df: pd.DataFrame, base_cfg: DomainConfig) -> Tuple[DomainConfig, str]:
    """
    Auto-detect frequency from ds spacing and return an adjusted cfg_local plus a tag:
      - yearly: freq="YS", seasonal_m=1, drop missing, no reindex
      - hourly: freq="H", seasonal_m=24, ffill small gaps, reindex
      - daily: freq="D", seasonal_m=7, ffill small gaps, reindex
      - business: freq="B", seasonal_m=5, drop missing, no reindex (stocks)
    """
    ds = df["ds"].sort_values()
    if len(ds) < 3:
        return base_cfg, "unknown"

    # Yearly-like detection: if most unique years and low count, treat as yearly
    if _detect_yearly_like(ds.astype(str)):
        cfg_local = replace(
            base_cfg,
            freq="YS",
            seasonal_m=1,
            reindex_full_grid=False,
            missing=MissingPolicy("drop", None),
        )
        return cfg_local, "yearly"

    deltas = ds.diff().dropna()
    if deltas.empty:
        return base_cfg, "unknown"

    med = deltas.median()

    # Hourly-ish
    if med <= pd.Timedelta(hours=2):
        cfg_local = replace(
            base_cfg,
            freq="H",
            seasonal_m=24,
            reindex_full_grid=True,
            missing=MissingPolicy("ffill", 3),
        )
        return cfg_local, "hourly"

    # Daily-ish
    if med <= pd.Timedelta(days=2):
        cfg_local = replace(
            base_cfg,
            freq="D",
            seasonal_m=7,
            reindex_full_grid=True,
            missing=MissingPolicy("ffill", 6),
        )
        return cfg_local, "daily"

    # Weekly-ish
    if med <= pd.Timedelta(days=10):
        cfg_local = replace(
            base_cfg,
            freq="W",
            seasonal_m=52,
            reindex_full_grid=True,
            missing=MissingPolicy("ffill", 2),
        )
        return cfg_local, "weekly"

    # Fallback: keep base
    return base_cfg, "regular"


def apply_grid_and_missing(df: pd.DataFrame, cfg: DomainConfig) -> pd.DataFrame:
    df = df.set_index("ds")

    if cfg.reindex_full_grid:
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq=cfg.freq)
        df = df.reindex(full_idx)

    if cfg.missing.mode == "drop":
        df = df.dropna(subset=["y"])
    elif cfg.missing.mode == "ffill":
        df["y"] = df["y"].ffill(limit=cfg.missing.max_gap)
        df = df.dropna(subset=["y"])
    else:
        raise ValueError(f"Unknown missing.mode: {cfg.missing.mode}")

    return df.reset_index().rename(columns={"index": "ds"})


def to_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    y = df["y"].astype(float)

    if int((y > 0).sum()) < MIN_POS_FOR_LOGRET:
        return pd.DataFrame(columns=["ds", "y"])

    y = y.where(y > 0)
    df["y"] = np.log(y / y.shift(1))
    return df.dropna(subset=["y"]).reset_index(drop=True)


def find_existing_folder(root: Path, options: List[str]) -> Optional[Path]:
    for name in options:
        p = root / name
        if p.exists() and p.is_dir():
            return p
    return None


# =========================
# Processing
# =========================

def process_one_series(csv_path: Path, domain: str, cfg: DomainConfig) -> Tuple[pd.DataFrame, str]:
    raw = read_csv_smart(str(csv_path))

    ds_col = cfg.ds_col
    y_col = cfg.y_col

    # 1) Apply overrides first (deterministic)
    if domain == "energy":
        ov = ENERGY_OVERRIDES.get(csv_path.stem)
        if ov is not None:
            ds_col, y_col = ov

    # 2) Infer if needed
    if ds_col is None or y_col is None:
        ds2, y2 = infer_ds_y(domain, raw)
        ds_col = ds_col or ds2
        y_col = y_col or y2

    if ds_col is None or y_col is None:
        raise ValueError(f"Cannot infer ds/y. Columns: {list(raw.columns)}")

    df = raw[[ds_col, y_col]].copy()
    df.columns = ["ds", "y"]

    df = enforce_schema(df)

    # 3) Auto-config (hourly/daily/yearly) mainly for energy
    cfg_local = cfg
    tag = "regular"
    if domain == "energy":
        cfg_local, tag = infer_time_config_from_ds(df, cfg)

    df = apply_grid_and_missing(df, cfg_local)

    if cfg_local.use_log_returns:
        df = to_log_returns(df)

    df = enforce_schema(df)

    if df.empty:
        raise ValueError(
            f"Empty after cleaning/transforms (picked ds_col='{ds_col}', y_col='{y_col}')"
        )

    min_n = MIN_N_YEARLY if tag == "yearly" else MIN_N_DEFAULT
    if len(df) < min_n:
        raise ValueError(f"Too short after cleaning: n={len(df)} (<{min_n})")

    return df[["ds", "y"]], tag


def prepare_cleaned_datasets(raw_root: str) -> None:
    raw_root_path = Path(raw_root).resolve()
    if not raw_root_path.exists():
        raise FileNotFoundError(f"Raw datasets folder not found: {raw_root_path}")

    cleaned_root = raw_root_path.parent / f"{raw_root_path.name}_cleaned"
    cleaned_root.mkdir(parents=True, exist_ok=True)

    use_parquet = (PREFERRED_FORMAT == "parquet" and parquet_supported())
    ext = "parquet" if use_parquet else "csv"

    log_rows = []

    for domain, cfg in DOMAIN_CONFIGS.items():
        folder = find_existing_folder(raw_root_path, cfg.folders)
        if folder is None:
            print(f"[SKIP] missing folder for '{domain}': tried {cfg.folders}")
            continue

        out_domain_folder = cleaned_root / folder.name
        out_domain_folder.mkdir(parents=True, exist_ok=True)

        csvs = sorted(folder.glob("*.csv"))
        if not csvs:
            print(f"[SKIP] no CSV files in: {folder}")
            continue

        for csv_path in csvs:
            try:
                df, tag = process_one_series(csv_path, domain, cfg)

                out_path = out_domain_folder / f"{csv_path.stem}.{ext}"
                if use_parquet:
                    df.to_parquet(out_path, index=False)
                else:
                    df.to_csv(out_path, index=False)

                print(f"[OK] cleaned {domain}/{csv_path.stem} ({tag}) -> {out_path.name} n={len(df)}")

                log_rows.append({
                    "status": "ok",
                    "domain": domain,
                    "raw_file": str(csv_path),
                    "out_file": str(out_path),
                    "n_rows": int(len(df)),
                    "format": ext,
                    "series_type": tag,
                    "error": "",
                })

            except Exception as e:
                print(f"[FAIL] {domain}/{csv_path.stem} -> {e}")
                log_rows.append({
                    "status": "fail",
                    "domain": domain,
                    "raw_file": str(csv_path),
                    "out_file": "",
                    "n_rows": 0,
                    "format": ext,
                    "series_type": "",
                    "error": str(e),
                })

    manifest_path = cleaned_root / "cleaning_manifest.csv"
    pd.DataFrame(log_rows).to_csv(manifest_path, index=False)

    print(f"\nDone. Cleaned datasets saved to: {cleaned_root}")
    print(f"Manifest saved to: {manifest_path}")
    if not use_parquet:
        print("Note: Parquet engine not available; wrote CSV instead.")


if __name__ == "__main__":
    ROOT = r"C:\Users\kevin\Desktop\thesis_forecasting\data\datasets"
    prepare_cleaned_datasets(ROOT)
