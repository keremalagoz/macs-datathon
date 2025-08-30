"""
features_v1_minimal.py — Oturum-temelli minimal, sızıntısız özellik üretimi

Amaç:
- Train ve test event kayıtlarını oturum (user_session) seviyesine indirgemek.
- Basit, hesaplaması ucuz ve sızıntısız öznitelikler çıkarmak:
  * session_event_count
  * unique_product_count
  * unique_category_count
  * event_type_count_<type> (tüm event_type değerleri için sayım)
  * event_type_ratio_<type> (count/total)
  * user_id (eşleştirme ve CV gruplaması için)

Girdi:
- data/raw/train.csv (event bazlı)
- data/raw/test.csv  (event bazlı)

Çıktı:
- data/processed/session_features_v1_train.parquet (veya .csv fallback)
- data/processed/session_features_v1_test.parquet  (veya .csv fallback)

Notlar:
- Parquet yazımı için pyarrow gerekli; yoksa otomatik CSV’ye düşer.
- Hiçbir hedef (session_value) bilgisini kullanmaz; sadece oturum içi olaylardan türetir.
"""
from __future__ import annotations
import os
import sys
from typing import Tuple

import pandas as pd
import numpy as np

DATA_RAW = os.path.join("data", "raw")
DATA_PROC = os.path.join("data", "processed")


def _read_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr_p = os.path.join(DATA_RAW, "train.csv")
    te_p = os.path.join(DATA_RAW, "test.csv")
    if not (os.path.exists(tr_p) and os.path.exists(te_p)):
        raise FileNotFoundError(f"Eksik dosya: {tr_p} veya {te_p}")
    train = pd.read_csv(tr_p)
    test = pd.read_csv(te_p)
    return train, test


def _safe_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")
    return df[cols]


def _build_event_type_pivot(df: pd.DataFrame) -> pd.DataFrame:
    # user_session x event_type sayım matrisi
    pivot = (
        df.groupby(["user_session", "event_type"]).size().unstack(fill_value=0)
    )
    # Kolon adlarını güvenli prefix ile ver
    pivot.columns = [f"event_type_count_{str(c)}" for c in pivot.columns]
    pivot = pivot.reset_index()
    return pivot


def _session_base(df: pd.DataFrame) -> pd.DataFrame:
    # user_id, product_id, category_id, user_session zorunlu sütunlar
    base_cols = ["user_id", "product_id", "category_id", "user_session", "event_type"]
    df = _safe_cols(df, base_cols).copy()

    # Temel agregasyonlar
    grp = df.groupby("user_session")
    base = grp.agg(
        user_id=("user_id", "first"),
        session_event_count=("user_session", "size"),
        unique_product_count=("product_id", pd.Series.nunique),
        unique_category_count=("category_id", pd.Series.nunique),
    ).reset_index()

    # event_type sayım pivotu
    evt_pivot = _build_event_type_pivot(df)
    out = base.merge(evt_pivot, on="user_session", how="left")

    # Oranlar: event_type_ratio_* = count / total
    count_cols = [c for c in out.columns if c.startswith("event_type_count_")]
    for c in count_cols:
        r = c.replace("event_type_count_", "event_type_ratio_")
        out[r] = out[c] / out["session_event_count"].replace(0, np.nan)
    out.fillna(0.0, inplace=True)
    return out


def _split_train_test_features(all_feat: pd.DataFrame, train_sessions: set[str], test_sessions: set[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr_feat = all_feat[all_feat["user_session"].isin(train_sessions)].copy()
    te_feat = all_feat[all_feat["user_session"].isin(test_sessions)].copy()
    # Kolon düzeni: anahtar + user_id + özellikler
    cols = ["user_session", "user_id"] + [c for c in all_feat.columns if c not in ("user_session", "user_id")]
    tr_feat = tr_feat[cols]
    te_feat = te_feat[cols]
    return tr_feat, te_feat


def _try_write(df: pd.DataFrame, path_no_ext: str) -> str:
    os.makedirs(DATA_PROC, exist_ok=True)
    parquet_path = os.path.join(DATA_PROC, path_no_ext + ".parquet")
    csv_path = os.path.join(DATA_PROC, path_no_ext + ".csv")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as e:
        # pyarrow yoksa CSV’ye düş
        df.to_csv(csv_path, index=False)
        return csv_path


def main() -> int:
    print("[features_v1_minimal] reading raw data...")
    train, test = _read_raw()

    # Train/test birleştirip ortak event_type uzayı çıkar
    print("[features_v1_minimal] building combined feature space...")
    train["__split"] = "train"
    test["__split"] = "test"
    all_df = pd.concat([train, test], axis=0, ignore_index=True)

    # Oturum bazlı temel özellikler
    print("[features_v1_minimal] aggregating session-level features...")
    all_feat = _session_base(all_df)

    train_sessions = set(train["user_session"].unique())
    test_sessions = set(test["user_session"].unique())

    tr_feat, te_feat = _split_train_test_features(all_feat, train_sessions, test_sessions)

    print(
        f"[features_v1_minimal] shapes -> all:{all_feat.shape} train_feat:{tr_feat.shape} test_feat:{te_feat.shape}"
    )

    # Yaz
    tr_out = _try_write(tr_feat, "session_features_v1_train")
    te_out = _try_write(te_feat, "session_features_v1_test")

    print("[features_v1_minimal] wrote:")
    print({"train_features": tr_out, "test_features": te_out})

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"Hata: {e}", file=sys.stderr)
        raise
