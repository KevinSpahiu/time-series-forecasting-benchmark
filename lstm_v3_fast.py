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

ROOT_OUT = r"C:\Users\kevin\Desktop\thesis_forecasting\outputs_v3\lstm_v3_fast"

MAX_TRAIN_ROWS = {
    "hourly": 50000,
    "daily": 30000,
    "weekly": 30000,
    "monthly": 30000,
    "regular": 30000,
}
LAGS = {"hourly": 72, "daily": 30, "weekly": 24, "monthly": 24, "regular": 30}
REFIT_EVERY = {"hourly": 28, "daily": 21, "weekly": 8, "monthly": 3, "regular": 21}

LSTM_UNITS = 16
DENSE_UNITS = 8
DROPOUT = 0.1

EPOCHS_AT_REFIT = 12
EPOCHS_BETWEEN_REFITS = 4
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

VAL_FRACTION_WITHIN_TRAIN = 0.15
EARLY_STOP_PATIENCE = 3
WARM_START_UPDATES = True
RESET_MODEL_ON_REFIT = True
TF_SET_MEMORY_GROWTH = True
TF_SEED = 42


def _tf_import():
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        return tf, keras, layers
    except Exception:
        return None, None, None


def _sk_import():
    try:
        from sklearn.preprocessing import StandardScaler
        return StandardScaler
    except Exception:
        return None


def tf_configure_runtime():
    tf, _, _ = _tf_import()
    if tf is None:
        return
    try:
        tf.random.set_seed(TF_SEED)
        if TF_SET_MEMORY_GROWTH:
            gpus = tf.config.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass


def make_sequences_1step(y_scaled: np.ndarray, lags: int):
    X, t = [], []
    for i in range(lags, len(y_scaled)):
        X.append(y_scaled[i - lags:i])
        t.append(y_scaled[i])
    X = np.asarray(X, dtype=np.float32).reshape(-1, lags, 1)
    t = np.asarray(t, dtype=np.float32).reshape(-1, 1)
    return X, t


def build_lstm_model(lags: int):
    tf, keras, layers = _tf_import()
    if tf is None:
        raise RuntimeError("tensorflow not installed. Run: pip install tensorflow")
    inp = keras.Input(shape=(lags, 1))
    x = layers.LSTM(LSTM_UNITS, dropout=DROPOUT, recurrent_dropout=0.0)(inp)
    x = layers.Dense(DENSE_UNITS, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = keras.Model(inp, out)
    opt = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=opt, loss="mse")
    return model


def fit_or_update_model(model, X_tr, y_tr, X_val, y_val, epochs: int):
    _, keras, _ = _tf_import()
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
        )
    ]
    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=callbacks,
        shuffle=True,
    )
    return model


def recursive_forecast_1step(model, scaler, history: np.ndarray, lags: int, h_max: int) -> np.ndarray:
    hist = history.astype(np.float32).tolist()
    preds = []
    for _ in range(h_max):
        x_win = np.asarray(hist[-lags:], dtype=np.float32).reshape(-1, 1)
        x_scaled = scaler.transform(x_win).astype(np.float32).reshape(1, lags, 1)
        yhat_scaled = float(model.predict(x_scaled, verbose=0)[0, 0])
        yhat = float(scaler.inverse_transform(np.asarray([[yhat_scaled]], dtype=np.float32))[0, 0])
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

    tf, _, _ = _tf_import()
    if tf is None:
        raise RuntimeError("tensorflow not installed. Run: pip install tensorflow")
    StandardScaler = _sk_import()
    if StandardScaler is None:
        raise RuntimeError("scikit-learn not installed. Run: pip install scikit-learn")

    tf_configure_runtime()

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
            refit_every = REFIT_EVERY.get(tag, 21)
            max_train_rows = MAX_TRAIN_ROWS.get(tag, 30000)

            rows = []
            model = None
            scaler = None

            for j, (origin_idx, gp) in enumerate(requested.groupby("origin_idx")):
                origin_idx = int(origin_idx)
                h_max = int(gp["horizon"].max())
                start_idx = max(0, origin_idx - max_train_rows)
                y_train = y[start_idx:origin_idx].astype(np.float32)
                if len(y_train) < lags + 80:
                    continue

                scaler = StandardScaler()
                y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).astype(np.float32).reshape(-1)
                X, t = make_sequences_1step(y_train_scaled, lags)
                if len(X) < 60:
                    continue

                n = len(X)
                val_n = max(20, int(VAL_FRACTION_WITHIN_TRAIN * n))
                val_n = min(val_n, n - 20)
                if val_n <= 0:
                    continue

                X_tr, t_tr = X[:-val_n], t[:-val_n]
                X_val, t_val = X[-val_n:], t[-val_n:]

                do_refit = model is None or (j % max(1, refit_every) == 0)
                if do_refit:
                    if RESET_MODEL_ON_REFIT or model is None:
                        model = build_lstm_model(lags)
                    model = fit_or_update_model(model, X_tr, t_tr, X_val, t_val, EPOCHS_AT_REFIT)
                else:
                    if WARM_START_UPDATES and EPOCHS_BETWEEN_REFITS > 0:
                        model = fit_or_update_model(model, X_tr, t_tr, X_val, t_val, EPOCHS_BETWEEN_REFITS)

                history = y_train
                if len(history) < lags or model is None or scaler is None:
                    continue

                fc = recursive_forecast_1step(model, scaler, history, lags, h_max)
                for _, prow in gp.iterrows():
                    h = int(prow["horizon"])
                    rows.append(
                        {
                            "dataset": dataset,
                            "series_id": series_id,
                            "series_tag": tag,
                            "model": "LSTM",
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
            mets["model"] = "LSTM"

            preds_all.append(preds)
            metrics_all.append(mets)
            coverage_rows.append(
                {
                    "dataset": dataset,
                    "series_id": series_id,
                    "series_tag": tag,
                    "model": "LSTM",
                    "requested_rows": requested_rows,
                    "generated_rows": len(preds),
                    "missing_rows": requested_rows - len(preds),
                    "n_origins_requested": requested_origins,
                    "n_origins_generated": int(preds["origin_idx"].nunique()),
                    "fail_reason": "",
                }
            )
            print(f"[OK] LSTM {dataset}/{series_id} ({tag}) preds={len(preds)}")
        except Exception as e:
            coverage_rows.append(
                {
                    "dataset": dataset,
                    "series_id": series_id,
                    "series_tag": tag,
                    "model": "LSTM",
                    "requested_rows": requested_rows,
                    "generated_rows": 0,
                    "missing_rows": requested_rows,
                    "n_origins_requested": requested_origins,
                    "n_origins_generated": 0,
                    "fail_reason": str(e),
                }
            )
            print(f"[FAIL] LSTM {dataset}/{series_id} -> {e}")

    outp = Path(ROOT_OUT)
    pd.DataFrame(coverage_rows).to_csv(outp / "lstm_coverage_v3.csv", index=False)

    if preds_all:
        preds_df = pd.concat(preds_all, ignore_index=True)
        preds_df.to_csv(outp / "lstm_predictions_v3.csv", index=False)

    if metrics_all:
        metrics_df = pd.concat(metrics_all, ignore_index=True)
        metrics_df.to_csv(outp / "lstm_metrics_per_series_v3.csv", index=False)
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
        agg.to_csv(outp / "lstm_metrics_aggregated_v3.csv", index=False)

    print(f"\nDone. Results saved to: {ROOT_OUT}")


if __name__ == "__main__":
    run_pipeline()
