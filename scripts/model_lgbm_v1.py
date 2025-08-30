"""
model_lgbm_v1.py — Minimal V1 özellikleriyle LightGBM modeli

Amaç:
- Train'de user_session bazında label (session_value) tekilleştir.
- V1 session özellikleriyle (counts, uniques, ratios) birleştir.
- GroupKFold(user_id, K=5) ile OOF/CV MSE hesapla; testte fold-average tahmin üret.
- Submission: submissions/lgbm_v1_minimal.csv

Girdi:
- data/raw/train.csv
- data/raw/sample_submission.csv
- data/processed/session_features_v1_train.{parquet|csv}
- data/processed/session_features_v1_test.{parquet|csv}

Çıktı:
- submissions/lgbm_v1_minimal.csv
- Konsolda: fold skorları ve OOF MSE özeti
"""
from __future__ import annotations
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SPLITS = 5
DATA_RAW = os.path.join("data", "raw")
DATA_PROC = os.path.join("data", "processed")
SUB_DIR = "submissions"


def _read_features(base: str) -> pd.DataFrame:
    parq = os.path.join(DATA_PROC, base + ".parquet")
    csv = os.path.join(DATA_PROC, base + ".csv")
    if os.path.exists(parq):
        return pd.read_parquet(parq)
    if os.path.exists(csv):
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Bulunamadı: {parq} veya {csv}")


def _read_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = os.path.join(DATA_RAW, "train.csv")
    sub = os.path.join(DATA_RAW, "sample_submission.csv")
    if not (os.path.exists(tr) and os.path.exists(sub)):
        raise FileNotFoundError(f"Eksik dosyalar: {tr} veya {sub}")
    return pd.read_csv(tr), pd.read_csv(sub)


def build_session_labels(train: pd.DataFrame) -> pd.DataFrame:
    req = {"user_id", "user_session", "session_value"}
    missing = req - set(train.columns)
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")
    lab = (
        train[["user_session", "user_id", "session_value"]]
        .dropna(subset=["user_session", "user_id", "session_value"]) 
        .drop_duplicates(subset=["user_session"], keep="first")
        .reset_index(drop=True)
    )
    if lab.empty:
        raise ValueError("Boş label seti")
    return lab


def group_kfold_indices(groups: np.ndarray, n_splits: int = N_SPLITS, seed: int = RANDOM_SEED):
    uniq = pd.unique(groups)
    rng = np.random.default_rng(seed)
    uniq = uniq[rng.permutation(len(uniq))]
    buckets = np.array_split(uniq, n_splits)
    for k in range(n_splits):
        val_groups = set(buckets[k])
        val_mask = np.array([g in val_groups for g in groups])
        train_idx = np.where(~val_mask)[0]
        val_idx = np.where(val_mask)[0]
        yield train_idx, val_idx


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def main() -> int:
    print("[model_lgbm_v1] reading data and features...")
    train, sample_sub = _read_raw()
    feat_tr = _read_features("session_features_v1_train")
    feat_te = _read_features("session_features_v1_test")
    print(f"[model_lgbm_v1] features loaded: train_feat={feat_tr.shape} test_feat={feat_te.shape}")

    print("[model_lgbm_v1] building session labels and merging features...")
    labels = build_session_labels(train)
    print(f"[model_lgbm_v1] labels built: {labels.shape}, unique sessions={labels['user_session'].nunique()}")
    # Hızlı anahtar kontrolü
    if not set(["user_session","user_id"]).issubset(feat_tr.columns):
        missing = {"user_session","user_id"} - set(feat_tr.columns)
        raise ValueError(f"Feature tablosunda eksik anahtar kolonlar: {missing}")
    df = labels.merge(feat_tr, on=["user_session", "user_id"], how="left")
    print(f"[model_lgbm_v1] merged shape: {df.shape}")

    # Feature set: user_session/user_id/target hariç tüm sayısal sütunlar
    drop_cols = {"user_session", "user_id", "session_value"}
    feature_cols = [c for c in df.columns if c not in drop_cols]
    print(f"[model_lgbm_v1] selected {len(feature_cols)} feature cols")

    X = df[feature_cols].fillna(0.0).to_numpy(dtype=float)
    y = df["session_value"].to_numpy(dtype=float)
    groups = df["user_id"].to_numpy()

    # LightGBM import
    try:
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
    except Exception as e:
        raise RuntimeError("LightGBM import edilemedi. Lütfen 'pip install lightgbm' kurun.") from e

    print(f"[model_lgbm_v1] features: {len(feature_cols)} cols, samples: {len(y)}")

    oof = np.zeros_like(y, dtype=float)
    test_pred = np.zeros(len(feat_te), dtype=float)
    # Test özellik matrisini bir kez hazırla
    X_te = feat_te[feature_cols].fillna(0.0).to_numpy(dtype=float)

    fold_scores: List[float] = []
    for i, (tr_idx, va_idx) in enumerate(group_kfold_indices(groups, N_SPLITS, RANDOM_SEED), start=1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        model = LGBMRegressor(
            random_state=RANDOM_SEED,
            n_estimators=1500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
        )

        print(f"[model_lgbm_v1] training fold {i}...")
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100),
        ]
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="l2",
            callbacks=callbacks,
        )
        # Not: early_stopping_rounds ile en iyi iterasyona göre duracaktır.

        pred_va = model.predict(X_va)
        oof[va_idx] = pred_va
        fold_mse = mse(y_va, pred_va)
        fold_scores.append(fold_mse)
        print(f"Fold {i}: val_MSE={fold_mse:.6f} (n_tr={len(tr_idx)} n_va={len(va_idx)})")

    # Test tahmini (best_iteration_ varsa kullanır)
    test_pred += model.predict(X_te) / N_SPLITS

    oof_mse = mse(y, oof)
    print(f"OOF MSE: {oof_mse:.6f}")
    print(f"CV MSE (mean over {N_SPLITS} folds): {np.mean(fold_scores):.6f}")

    # Submission yazımı — sample_submission sırası korunarak
    print("[model_lgbm_v1] writing submission...")
    sub = sample_sub.copy()
    # Feature test tablosundaki user_session'lar ile aynı olmalı
    # sample_submission kullanıcı oturum sırasını koruyoruz
    pred_map = dict(zip(feat_te["user_session"].values, test_pred))
    sub["session_value"] = sub["user_session"].map(pred_map).astype(float)

    os.makedirs(SUB_DIR, exist_ok=True)
    out_csv = os.path.join(SUB_DIR, "lgbm_v1_minimal.csv")
    sub.to_csv(out_csv, index=False)
    print({
        "oof_mse": round(oof_mse, 6),
        "fold_mses": [round(x, 6) for x in fold_scores],
        "submission": out_csv,
        "n_features": len(feature_cols),
    })

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"Hata: {e}", file=sys.stderr)
        raise
