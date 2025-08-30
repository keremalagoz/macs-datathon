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
from typing import List, Tuple, Callable

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
    backend = os.getenv("MODEL_BACKEND", "lgbm").strip().lower()
    target_transform = os.getenv("TARGET_TRANSFORM", "log1p").strip().lower()
    print(f"[model_lgbm_v1] reading data and features... (backend={backend})")
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
    raw_feature_cols = [c for c in df.columns if c not in drop_cols]
    # BUY feature'larını geri ekliyoruz (testte de mevcut; leakage değil)
    feature_cols = raw_feature_cols
    print(f"[model_lgbm_v1] selected {len(feature_cols)} feature cols (BUY features included)")

    print("[model_lgbm_v1] building matrices X/y/groups...")
    X = df[feature_cols].fillna(0.0).to_numpy(dtype=float)
    y = df["session_value"].to_numpy(dtype=float)
    groups = df["user_id"].to_numpy()
    print(f"[model_lgbm_v1] X shape: {X.shape}, y: {y.shape}, groups: {groups.shape}")

    # LightGBM import
    if backend == "lgbm":
        print("[model_lgbm_v1] importing LightGBM...")
        # Olası OpenMP/threads kilitlenmelerini azaltmak için thread sayısını kısıtla
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        try:
            import lightgbm as lgb
            from lightgbm import LGBMRegressor
        except Exception as e:
            raise RuntimeError("LightGBM import edilemedi. Lütfen 'pip install lightgbm' kurun.") from e
        print("[model_lgbm_v1] LightGBM imported")
    elif backend == "xgb":
        print("[model_lgbm_v1] importing XGBoost...")
        try:
            import xgboost as xgb
        except Exception as e:
            raise RuntimeError("XGBoost import edilemedi. Lütfen 'pip install xgboost' kurun.") from e
        print("[model_lgbm_v1] XGBoost imported")
    elif backend == "sk":
        print("[model_lgbm_v1] importing scikit-learn (HistGradientBoosting)...")
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
        except Exception as e:
            raise RuntimeError("scikit-learn import edilemedi. Lütfen 'pip install scikit-learn' kurun.") from e
        print("[model_lgbm_v1] scikit-learn imported")
    else:
        raise ValueError(f"Bilinmeyen backend: {backend}")

    print(f"[model_lgbm_v1] features: {len(feature_cols)} cols, samples: {len(y)}")

    # Hedef dönüşümü
    def _tf_y(arr: np.ndarray) -> np.ndarray:
        if target_transform == "log1p":
            return np.log1p(np.clip(arr, a_min=0.0, a_max=None))
        return arr

    def _inv_y(arr: np.ndarray) -> np.ndarray:
        if target_transform == "log1p":
            return np.expm1(arr)
        return arr

    oof = np.zeros_like(y, dtype=float)
    test_pred = np.zeros(len(feat_te), dtype=float)
    # Test özellik matrisini ve user_id'leri bir kez hazrla
    X_te_base = feat_te[feature_cols].fillna(0.0).to_numpy(dtype=float)
    user_id_te = feat_te["user_id"].to_numpy()

    fold_scores: List[float] = []
    for i, (tr_idx, va_idx) in enumerate(group_kfold_indices(groups, N_SPLITS, RANDOM_SEED), start=1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        # Kullanıcı bazlı backoff ortalaması (sadece bu fold'un train kısmından)
        tr_users = pd.Series(groups[tr_idx], name="user_id")
        va_users = pd.Series(groups[va_idx], name="user_id")
        df_mu = pd.DataFrame({"user_id": tr_users, "y": y_tr}).groupby("user_id")["y"].mean()
        global_mu = float(y_tr.mean())
        backoff_tr = tr_users.map(df_mu).fillna(global_mu).to_numpy(dtype=float)
        backoff_va = va_users.map(df_mu).fillna(global_mu).to_numpy(dtype=float)
        # Test için de aynı fold train'inden üret
        backoff_te = pd.Series(user_id_te).map(df_mu).fillna(global_mu).to_numpy(dtype=float)

        # X'lere ekle
        X_tr = np.hstack([X_tr, backoff_tr.reshape(-1, 1)])
        X_va = np.hstack([X_va, backoff_va.reshape(-1, 1)])
        X_te = np.hstack([X_te_base, backoff_te.reshape(-1, 1)])

        if backend == "lgbm":
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
                X_tr, _tf_y(y_tr),
                eval_set=[(X_va, _tf_y(y_va))],
                eval_metric="l2",
                callbacks=callbacks,
            )
        elif backend == "xgb":
            # XGBoost Regresör; benzer parametreler ve early stopping
            model = xgb.XGBRegressor(
                random_state=RANDOM_SEED,
                n_estimators=1500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                n_jobs=0,
            )
            print(f"[model_lgbm_v1] training fold {i} (XGB)...")
            model.fit(
                X_tr, _tf_y(y_tr),
                eval_set=[(X_va, _tf_y(y_va))],
                eval_metric="rmse",
                early_stopping_rounds=50,
                verbose=100,
            )
        else:  # sklearn HistGradientBoosting
            model = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=600,
                max_leaf_nodes=63,
                min_samples_leaf=20,
                l2_regularization=0.0,
                early_stopping=True,
                n_iter_no_change=30,
                validation_fraction=0.2,
                random_state=RANDOM_SEED,
            )
            print(f"[model_lgbm_v1] training fold {i} (SK-HGB)...")
            model.fit(X_tr, _tf_y(y_tr))
            print(f"[model_lgbm_v1] fold {i} training done (SK-HGB)")
        # Not: early_stopping_rounds ile en iyi iterasyona göre duracaktır.
        
        pred_va = _inv_y(model.predict(X_va))
        oof[va_idx] = pred_va
        fold_mse = mse(y_va, pred_va)
        fold_scores.append(fold_mse)
        print(f"Fold {i}: val_MSE={fold_mse:.6f} (n_tr={len(tr_idx)} n_va={len(va_idx)})")

        # Test tahmini — her fold modelinden toplayıp ortalıyoruz
        test_pred += _inv_y(model.predict(X_te)) / N_SPLITS

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
    # Güvenlik: eksik kalan varsa global ortalamayla doldur
    if sub["session_value"].isna().any():
        sub["session_value"].fillna(float(y.mean()), inplace=True)

    os.makedirs(SUB_DIR, exist_ok=True)
    out_name_map = {
        "lgbm": "lgbm_v1_minimal_buy_backoff_log.csv" if target_transform == "log1p" else "lgbm_v1_minimal_buy_backoff.csv",
        "xgb": "xgb_v1_minimal_buy_backoff_log.csv" if target_transform == "log1p" else "xgb_v1_minimal_buy_backoff.csv",
        "sk": "sk_v1_minimal_buy_backoff_log.csv" if target_transform == "log1p" else "sk_v1_minimal_buy_backoff.csv",
    }
    out_name = out_name_map.get(backend, "model_v1_minimal_nobuy_backoff.csv")
    out_csv = os.path.join(SUB_DIR, out_name)
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
