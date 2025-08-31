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


def _read_raw() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = os.path.join(DATA_RAW, "train.csv")
    te = os.path.join(DATA_RAW, "test.csv")
    sub = os.path.join(DATA_RAW, "sample_submission.csv")
    if not (os.path.exists(tr) and os.path.exists(sub) and os.path.exists(te)):
        raise FileNotFoundError(f"Eksik dosyalar: {tr} veya {te} veya {sub}")
    return pd.read_csv(tr), pd.read_csv(te), pd.read_csv(sub)


def _detect_time_col(df: pd.DataFrame) -> str | None:
    for c in ["event_time", "timestamp", "event_timestamp", "time", "ts"]:
        if c in df.columns:
            return c
    return None


def _last_items_per_session(df: pd.DataFrame) -> pd.DataFrame:
    # user_session bazında son product_id ve category_id'yi çıkar
    required = {"user_session", "product_id", "category_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")
    time_col = _detect_time_col(df)
    w = df[["user_session", "product_id", "category_id"] + ([time_col] if time_col else [])].copy()
    if time_col:
        w[time_col] = pd.to_datetime(w[time_col], errors="coerce")
        w.sort_values(["user_session", time_col], inplace=True)
    else:
        w["__row_id"] = np.arange(len(w))
        w.sort_values(["user_session", "__row_id"], inplace=True)
    last = w.groupby("user_session").agg(
        last_product_id=("product_id", "last"),
        last_category_id=("category_id", "last"),
    ).reset_index()
    return last


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
    winsorize_p = float(os.getenv("WINSORIZE_P", "0"))  # 0: kapalı, ör: 0.995 üstünü kırp
    full_train = os.getenv("FULL_TRAIN", "0").strip() in {"1", "true", "yes"}
    print(f"[model_lgbm_v1] reading data and features... (backend={backend})")
    train_events, test_events, sample_sub = _read_raw()
    # Özellik versiyonu: önce v2 dene, yoksa v1'e düş
    try:
        feat_tr = _read_features("session_features_v2_train")
        feat_te = _read_features("session_features_v2_test")
        feat_ver = "v2"
    except FileNotFoundError:
        feat_tr = _read_features("session_features_v1_train")
        feat_te = _read_features("session_features_v1_test")
        feat_ver = "v1"
    print(f"[model_lgbm_v1] features loaded ({feat_ver}): train_feat={feat_tr.shape} test_feat={feat_te.shape}")

    print("[model_lgbm_v1] building session labels and merging features...")
    labels = build_session_labels(train_events)
    print(f"[model_lgbm_v1] labels built: {labels.shape}, unique sessions={labels['user_session'].nunique()}")
    # Oturumların son ürün ve kategori bilgisi (train+test)
    last_tr = _last_items_per_session(train_events)
    last_te = _last_items_per_session(test_events)
    labels = labels.merge(last_tr, on="user_session", how="left")
    feat_te = feat_te.merge(last_te, on="user_session", how="left")
    # Hızlı anahtar kontrolü
    if not set(["user_session","user_id"]).issubset(feat_tr.columns):
        missing = {"user_session","user_id"} - set(feat_tr.columns)
        raise ValueError(f"Feature tablosunda eksik anahtar kolonlar: {missing}")
    df = labels.merge(feat_tr, on=["user_session", "user_id"], how="left")
    print(f"[model_lgbm_v1] merged shape: {df.shape}")

    # Feature set: user_session/user_id/target hariç tüm sayısal sütunlar
    drop_cols = {"user_session", "user_id", "session_value", "last_product_id", "last_category_id"}
    raw_feature_cols = [c for c in df.columns if c not in drop_cols]
    # BUY feature'larını geri ekliyoruz (testte de mevcut; leakage değil)
    feature_cols = raw_feature_cols
    print(f"[model_lgbm_v1] selected {len(feature_cols)} feature cols (BUY features included)")

    print("[model_lgbm_v1] building matrices X/y/groups...")
    X = df[feature_cols].fillna(0.0).to_numpy(dtype=float)
    y = df["session_value"].to_numpy(dtype=float)
    groups = df["user_id"].to_numpy()
    # Son ürün/kategori id dizileri (model içi fold target encoding için)
    last_prod = df["last_product_id"].to_numpy()
    last_cat = df["last_category_id"].to_numpy()
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

    def _winsorize(vec: np.ndarray, p: float) -> np.ndarray:
        if p and 0.5 < p < 1.0:
            hi = float(np.quantile(vec, p))
            return np.minimum(vec, hi)
        return vec

    # Ortak test matrisi
    X_te_base = feat_te[feature_cols].fillna(0.0).to_numpy(dtype=float)
    user_id_te = feat_te["user_id"].to_numpy()
    last_prod_te = feat_te.get("last_product_id", pd.Series(index=feat_te.index)).to_numpy()
    last_cat_te = feat_te.get("last_category_id", pd.Series(index=feat_te.index)).to_numpy()

    # FULL_TRAIN kipi: tek model, tüm train ile eğitim
    if full_train:
        print("[model_lgbm_v1] FULL_TRAIN=1 — tek modelle tüm train üzerinde eğitim")
        # User backoff (tüm train'den)
        df_mu_all = pd.DataFrame({"user_id": df["user_id"].to_numpy(), "y": y}).groupby("user_id")["y"].mean()
        global_mu_all = float(y.mean())
        backoff_tr_all = pd.Series(df["user_id"].to_numpy()).map(df_mu_all).fillna(global_mu_all).to_numpy(dtype=float)
        backoff_te_all = pd.Series(user_id_te).map(df_mu_all).fillna(global_mu_all).to_numpy(dtype=float)
        # Product/Category target means (tüm train'den) — last item'a göre
        prod_mu_all = pd.DataFrame({"pid": pd.Series(last_prod), "y": y}).groupby("pid")["y"].mean()
        cat_mu_all = pd.DataFrame({"cid": pd.Series(last_cat), "y": y}).groupby("cid")["y"].mean()
        prod_tr_all = pd.Series(last_prod).map(prod_mu_all).fillna(global_mu_all).to_numpy(dtype=float)
        cat_tr_all = pd.Series(last_cat).map(cat_mu_all).fillna(global_mu_all).to_numpy(dtype=float)
        prod_te_all = pd.Series(last_prod_te).map(prod_mu_all).fillna(global_mu_all).to_numpy(dtype=float)
        cat_te_all = pd.Series(last_cat_te).map(cat_mu_all).fillna(global_mu_all).to_numpy(dtype=float)

        X_full = np.hstack([
            X,
            backoff_tr_all.reshape(-1, 1),
            prod_tr_all.reshape(-1, 1),
            cat_tr_all.reshape(-1, 1),
        ])
        X_te = np.hstack([
            X_te_base,
            backoff_te_all.reshape(-1, 1),
            prod_te_all.reshape(-1, 1),
            cat_te_all.reshape(-1, 1),
        ])

        # Model seçimi
        if backend == "sk":
            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=700,
                max_leaf_nodes=63,
                min_samples_leaf=20,
                l2_regularization=0.0,
                early_stopping=True,
                n_iter_no_change=30,
                validation_fraction=0.2,
                random_state=RANDOM_SEED,
            )
        elif backend == "xgb":
            import xgboost as xgb
            model = xgb.XGBRegressor(
                random_state=RANDOM_SEED,
                n_estimators=3500,
                learning_rate=0.03,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=10,
                reg_alpha=0.1,
                reg_lambda=1.0,
                tree_method="hist",
                n_jobs=0,
                eval_metric="rmse",
            )
        else:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(
                random_state=RANDOM_SEED,
                n_estimators=2000,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.9,
                n_jobs=-1,
            )

        print("[model_lgbm_v1] fitting full-train model...")
        y_full = _winsorize(y, winsorize_p)
        model.fit(X_full, _tf_y(y_full))
        test_pred = _inv_y(model.predict(X_te))

        # Submission yazımı
        print("[model_lgbm_v1] writing submission (FULL_TRAIN)...")
        sub = sample_sub.copy()
        pred_map = dict(zip(feat_te["user_session"].values, test_pred))
        sub["session_value"] = sub["user_session"].map(pred_map).astype(float)
        if sub["session_value"].isna().any():
            sub["session_value"].fillna(float(y.mean()), inplace=True)
        os.makedirs(SUB_DIR, exist_ok=True)
        suffix = "_win" + str(winsorize_p) if winsorize_p and winsorize_p > 0 else ""
        out_csv = os.path.join(SUB_DIR, f"{backend}_v1_fulltrain_buy_backoff_log{suffix}.csv")
        sub.to_csv(out_csv, index=False)
        print({
            "submission": out_csv,
            "n_features": X_full.shape[1],
        })
        return 0

    # CV yolu
    oof = np.zeros_like(y, dtype=float)
    test_pred = np.zeros(len(feat_te), dtype=float)

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

    # Target encoding (fold-train'den): product/category mean
    tr_prod = pd.Series(last_prod[tr_idx], name="pid")
    tr_cat = pd.Series(last_cat[tr_idx], name="cid")
    prod_mu = pd.DataFrame({"pid": tr_prod, "y": y_tr}).groupby("pid")["y"].mean()
    cat_mu = pd.DataFrame({"cid": tr_cat, "y": y_tr}).groupby("cid")["y"].mean()
    # Map to splits, eksikleri global ortalama ile doldur
    prod_tr = pd.Series(last_prod[tr_idx]).map(prod_mu).fillna(global_mu).to_numpy(dtype=float)
    prod_va = pd.Series(last_prod[va_idx]).map(prod_mu).fillna(global_mu).to_numpy(dtype=float)
    prod_te = pd.Series(last_prod_te).map(prod_mu).fillna(global_mu).to_numpy(dtype=float)
    cat_tr = pd.Series(last_cat[tr_idx]).map(cat_mu).fillna(global_mu).to_numpy(dtype=float)
    cat_va = pd.Series(last_cat[va_idx]).map(cat_mu).fillna(global_mu).to_numpy(dtype=float)
    cat_te = pd.Series(last_cat_te).map(cat_mu).fillna(global_mu).to_numpy(dtype=float)

    # X'lere ekle: backoff + prod_mu + cat_mu
    X_tr = np.hstack([X_tr, backoff_tr.reshape(-1, 1), prod_tr.reshape(-1,1), cat_tr.reshape(-1,1)])
    X_va = np.hstack([X_va, backoff_va.reshape(-1, 1), prod_va.reshape(-1,1), cat_va.reshape(-1,1)])
    X_te = np.hstack([X_te_base, backoff_te.reshape(-1, 1), prod_te.reshape(-1,1), cat_te.reshape(-1,1)])

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
        eval_metric="rmse",
            )
            print(f"[model_lgbm_v1] training fold {i} (XGB)...")
            model.fit(X_tr, _tf_y(y_tr))
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
