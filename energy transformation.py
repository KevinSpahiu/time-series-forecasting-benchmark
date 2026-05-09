from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict

import pandas as pd

# =========================
# SETTINGS
# =========================
RAW_ENERGY_DIR = r"C:\Users\kevin\Desktop\thesis_forecasting\data\datasets\energy"
OUT_ENERGY_DIR = r"C:\Users\kevin\Desktop\thesis_forecasting\data\datasets_cleaned\energy"

PREFERRED_FORMAT = "parquet"  # "parquet" or "csv"
PREFER_DAYFIRST = True

MIN_N_DEFAULT = 100   # hourly/daily typical
MIN_N_YEARLY = 20     # yearly

# Optional deterministic overrides: stem -> (ds_col, y_col)
# Add more if you want exact targets.
OVERRIDES: Dict[str, Tuple[str, str]] = {
    "EKPC_hourly": ("Datetime", "EKPC_MW"),
    "FE_hourly": ("Datetime", "FE_MW"),
    "NI_hourly": ("Datetime", "NI_MW"),
    "PJM_Load_hourly": ("Datetime", "PJM_Load_MW"),
    "PJME_hourly": ("Datetime", "PJME_MW"),
    "PJMW_hourly": ("Datetime", "PJMW_MW"),
    "energy_dataset": ("time", "generation solar"),  # change to "total load actual" if you prefer
    "KAG_energydata_complete": ("date", "Appliances"),
}

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
# Helpers
# =========================

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


def _norm(s: str) -> str:
    return s.strip().lower()


def read_csv_smart(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            col0 = str(df.columns[0])
            if ";" in col0:
                df = pd.read_csv(path, sep=";")
        return df
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", on_bad_lines="skip")


def clean_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    txt = s.astype(str).str.replace(",", "", regex=False)
    extracted = txt.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def detect_yearly_like(ds: pd.Series) -> bool:
    if ds.dtype.kind in "iu":
        return True
    ss = ds.astype(str).str.strip()
    if len(ss) == 0:
        return False
    return float(ss.str.fullmatch(r"\d{4}").mean()) >= 0.8


def coerce_year_only_dates(ds: pd.Series) -> pd.Series:
    if ds.dtype.kind in "iu":
        return ds.astype(str) + "-01-01"
    ss = ds.astype(str).str.strip()
    only_year = ss.str.fullmatch(r"\d{4}")
    return ss.where(~only_year, ss + "-01-01")


def parse_datetime_smart(ds: pd.Series, prefer_dayfirst: bool) -> pd.Series:
    dt1 = pd.to_datetime(ds, errors="coerce", dayfirst=prefer_dayfirst, utc=True)
    ok1 = float(dt1.notna().mean()) if len(dt1) else 0.0
    if ok1 >= 0.90:
        return dt1.dt.tz_convert(None)

    dt2 = pd.to_datetime(ds, errors="coerce", dayfirst=not prefer_dayfirst, utc=True)
    ok2 = float(dt2.notna().mean()) if len(dt2) else 0.0
    best = dt1 if ok1 >= ok2 else dt2
    return best.dt.tz_convert(None)


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if detect_yearly_like(df["ds"]):
        df["ds"] = coerce_year_only_dates(df["ds"])

    df["ds"] = parse_datetime_smart(df["ds"], prefer_dayfirst=PREFER_DAYFIRST)
    df["y"] = clean_numeric_series(df["y"])

    df = df.dropna(subset=["ds"])
    df = df.sort_values("ds").drop_duplicates("ds", keep="last")
    return df.reset_index(drop=True)


def infer_ds_y(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = list(df.columns)
    norm_map = {_norm(c): c for c in cols}

    # ds candidates
    ds_candidates = ("ds", "date", "datetime", "timestamp", "time", "year")
    ds_col = next((norm_map[k] for k in ds_candidates if k in norm_map), None)

    # energy targets (priority)
    y_priority = (
        # solar / generation
        "generation solar", "solar", "solar_generation",
        # load
        "total load actual", "total load forecast",
        # price
        "price actual", "price day ahead",
        # generic
        "load", "demand",
        # appliances dataset
        "appliances", "lights",
    )
    y_col = next((norm_map[k] for k in y_priority if k in norm_map), None)

    # fallback: *_MW columns
    if y_col is None:
        for c in cols:
            cn = _norm(c)
            if cn.endswith("_mw") or cn.endswith("mw") or " mw" in cn:
                y_col = c
                break

    # fallback: any numeric column besides ds
    if y_col is None and ds_col is not None:
        for c in cols:
            if c == ds_col:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                y_col = c
                break

    return ds_col, y_col


def infer_time_freq(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Returns (freq, tag). Use pandas aliases:
      hourly -> 'h' (lowercase) to avoid your KeyError('H') issue
    """
    ds = df["ds"].sort_values()
    if len(ds) < 3:
        return "D", "regular"

    if detect_yearly_like(ds.astype(str)):
        return "YS", "yearly"

    deltas = ds.diff().dropna()
    if deltas.empty:
        return "D", "regular"

    med = deltas.median()

    if med <= pd.Timedelta(hours=2):
        return "h", "hourly"
    if med <= pd.Timedelta(days=2):
        return "D", "daily"
    if med <= pd.Timedelta(days=10):
        return "W", "weekly"

    return "D", "regular"


def apply_grid_and_missing(df: pd.DataFrame, freq: str, tag: str) -> pd.DataFrame:
    df = df.set_index("ds")

    # yearly: don't invent years; just drop missing y
    if tag == "yearly":
        df = df.dropna(subset=["y"])
        return df.reset_index()

    # hourly/daily: reindex & ffill small gaps
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    df = df.reindex(full_idx)

    # gap limits: tighter for hourly
    limit = 3 if tag == "hourly" else 6
    df["y"] = df["y"].ffill(limit=limit)
    df = df.dropna(subset=["y"])

    return df.reset_index().rename(columns={"index": "ds"})


# =========================
# Cleaning
# =========================

def clean_one_file(path: Path) -> Tuple[pd.DataFrame, str]:
    raw = read_csv_smart(str(path))

    ds_col = y_col = None
    if path.stem in OVERRIDES:
        ds_col, y_col = OVERRIDES[path.stem]

    if ds_col is None or y_col is None:
        ds2, y2 = infer_ds_y(raw)
        ds_col = ds_col or ds2
        y_col = y_col or y2

    if ds_col is None or y_col is None:
        raise ValueError(f"Cannot infer ds/y. Columns: {list(raw.columns)}")

    df = raw[[ds_col, y_col]].copy()
    df.columns = ["ds", "y"]

    df = enforce_schema(df)

    freq, tag = infer_time_freq(df)
    df = apply_grid_and_missing(df, freq=freq, tag=tag)
    df = enforce_schema(df)

    min_n = MIN_N_YEARLY if tag == "yearly" else MIN_N_DEFAULT
    if len(df) < min_n:
        raise ValueError(f"Too short after cleaning: n={len(df)} (<{min_n})")

    return df[["ds", "y"]], tag


def main() -> None:
    in_dir = Path(RAW_ENERGY_DIR)
    out_dir = Path(OUT_ENERGY_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_parquet = (PREFERRED_FORMAT == "parquet" and parquet_supported())
    ext = "parquet" if use_parquet else "csv"

    files = sorted(in_dir.glob("*.csv"))
    if not files:
        print(f"[SKIP] No CSV files in {in_dir}")
        return

    log_rows = []

    for f in files:
        try:
            df, tag = clean_one_file(f)

            out_path = out_dir / f"{f.stem}.{ext}"
            if use_parquet:
                df.to_parquet(out_path, index=False)
            else:
                df.to_csv(out_path, index=False)

            print(f"[OK] cleaned energy/{f.stem} ({tag}) -> {out_path.name} n={len(df)}")
            log_rows.append({
                "status": "ok",
                "file": str(f),
                "out_file": str(out_path),
                "n_rows": int(len(df)),
                "series_type": tag,
                "error": "",
            })

        except Exception as e:
            print(f"[FAIL] energy/{f.stem} -> {e}")
            log_rows.append({
                "status": "fail",
                "file": str(f),
                "out_file": "",
                "n_rows": 0,
                "series_type": "",
                "error": str(e),
            })

    manifest = out_dir / "cleaning_manifest_energy.csv"
    pd.DataFrame(log_rows).to_csv(manifest, index=False)
    print(f"\nDone. Cleaned energy saved to: {out_dir}")
    print(f"Manifest saved to: {manifest}")
    if not use_parquet:
        print("Note: Parquet engine not available; wrote CSV instead.")


if __name__ == "__main__":
    main()
