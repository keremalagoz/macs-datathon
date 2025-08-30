"""
features_v2_session.py — Oturum-temelli zengin özellikler (sızıntısız)

Eklenenler (v1'e ek olarak):
- is_buyer (event_type_count_BUY>0)
- unique_product_ratio, unique_category_ratio
- revisit_ratio = 1 - unique_product_count/session_event_count
- product_repeat_max, product_repeat_max_ratio
- category_repeat_max, category_repeat_max_ratio
- event_type_entropy (Shannon)
- Oranlar: buy/view, cart/view, buy/cart (güvenli payda)
- last_event_type (sayısal kod) ve first_event_type (varsa zaman sırasına göre)
- session_duration_seconds, events_per_minute (zaman kolonu varsa)

Not: Hiçbir yerde hedef (session_value) kullanılmaz.
"""
from __future__ import annotations
import os
import sys
from typing import Tuple, Optional

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


def _detect_time_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["event_time", "timestamp", "event_timestamp", "time", "ts"]:
        if c in df.columns:
            return c
    return None


def _event_type_codes(all_df: pd.DataFrame) -> dict:
    # Tüm veride görülen event_type değerlerine sabit kodlama
    vals = pd.Categorical(all_df["event_type"].astype(str))
    mapping = {cat: i for i, cat in enumerate(vals.categories)}
    return mapping


def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _aggregate_features(df: pd.DataFrame, time_col: Optional[str], evt_code_map: dict) -> pd.DataFrame:
    base_cols = ["user_id", "product_id", "category_id", "user_session", "event_type"]
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")

    work = df[base_cols + ([time_col] if time_col else [])].copy()
    if time_col:
        # Zamanı datetime'a çevir ve oturum içi sırala
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work.sort_values(["user_session", time_col], inplace=True)
    else:
        # En azından grup içi stabil sıra için index'e göre sırala
        work["__row_id"] = np.arange(len(work))
        work.sort_values(["user_session", "__row_id"], inplace=True)

    grp = work.groupby("user_session")

    # Temel sayımlar
    base = grp.agg(
        user_id=("user_id", "first"),
        session_event_count=("user_session", "size"),
        unique_product_count=("product_id", pd.Series.nunique),
        unique_category_count=("category_id", pd.Series.nunique),
    ).reset_index()

    # event_type sayım pivotu
    evt_counts = (
        work.groupby(["user_session", "event_type"]).size().unstack(fill_value=0)
    )
    evt_counts.columns = [f"event_type_count_{str(c)}" for c in evt_counts.columns]
    evt_counts = evt_counts.reset_index()

    out = base.merge(evt_counts, on="user_session", how="left")

    # Oranlar
    for c in [col for col in out.columns if col.startswith("event_type_count_")]:
        out[c.replace("event_type_count_", "event_type_ratio_")] = out[c] / out[
            "session_event_count"
        ].replace(0, np.nan)

    # is_buyer
    buy_col = "event_type_count_BUY"
    out["is_buyer"] = (out[buy_col] > 0).astype(int) if buy_col in out.columns else 0

    # Unique oranları ve revisit
    out["unique_product_ratio"] = out["unique_product_count"] / out["session_event_count"].replace(0, np.nan)
    out["unique_category_ratio"] = out["unique_category_count"] / out["session_event_count"].replace(0, np.nan)
    out["revisit_ratio"] = 1.0 - out["unique_product_ratio"].clip(lower=0, upper=1)

    # Ürün ve kategori tekrarlarının maksimumu
    prod_max = grp["product_id"].agg(lambda s: int(s.value_counts().max()))
    cat_max = grp["category_id"].agg(lambda s: int(s.value_counts().max()))
    out = out.merge(prod_max.rename("product_repeat_max").reset_index(), on="user_session", how="left")
    out = out.merge(cat_max.rename("category_repeat_max").reset_index(), on="user_session", how="left")
    out["product_repeat_max_ratio"] = out["product_repeat_max"] / out["session_event_count"].replace(0, np.nan)
    out["category_repeat_max_ratio"] = out["category_repeat_max"] / out["session_event_count"].replace(0, np.nan)

    # Entropy (event_type dağılımı)
    count_cols = [c for c in out.columns if c.startswith("event_type_count_")]
    probs = out[count_cols].div(out["session_event_count"].replace(0, np.nan), axis=0).fillna(0.0).to_numpy()
    ent = np.apply_along_axis(_entropy, 1, probs)
    out["event_type_entropy"] = ent

    # Basit oranlar
    def _safe_ratio(n, d):
        return n / np.where(d <= 0, 1, d)
    out["buy_to_view_ratio"] = _safe_ratio(out.get("event_type_count_BUY", 0), out.get("event_type_count_VIEW", 0))
    out["cart_to_view_ratio"] = _safe_ratio(out.get("event_type_count_CART", 0), out.get("event_type_count_VIEW", 0))
    out["buy_to_cart_ratio"] = _safe_ratio(out.get("event_type_count_BUY", 0), out.get("event_type_count_CART", 0))

    # Son/ilk event_type (kod)
    def _first_last_codes(g: pd.DataFrame) -> pd.Series:
        first_evt = str(g["event_type"].iloc[0])
        last_evt = str(g["event_type"].iloc[-1])
        return pd.Series({
            "first_event_type_code": evt_code_map.get(first_evt, -1),
            "last_event_type_code": evt_code_map.get(last_evt, -1),
        })
    fl = work.groupby("user_session").apply(_first_last_codes).reset_index()
    out = out.merge(fl, on="user_session", how="left")

    # Zaman özellikleri varsa
    if time_col:
        ses = work.groupby("user_session")[time_col].agg(["min", "max", "size"]).reset_index()
        ses["session_duration_seconds"] = (ses["max"] - ses["min"]).dt.total_seconds().fillna(0.0)
        ses["events_per_minute"] = ses["size"] / np.where(ses["session_duration_seconds"] <= 0, 60.0, ses["session_duration_seconds"]/60.0)
        out = out.merge(ses[["user_session", "session_duration_seconds", "events_per_minute"]], on="user_session", how="left")

    out.replace([np.inf, -np.inf], 0.0, inplace=True)
    out.fillna(0.0, inplace=True)
    return out


def _split_train_test_features(all_feat: pd.DataFrame, train_sessions: set[str], test_sessions: set[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr_feat = all_feat[all_feat["user_session"].isin(train_sessions)].copy()
    te_feat = all_feat[all_feat["user_session"].isin(test_sessions)].copy()
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
    except Exception:
        df.to_csv(csv_path, index=False)
        return csv_path


def main() -> int:
    print("[features_v2_session] reading raw data...")
    train, test = _read_raw()
    train["__split"] = "train"
    test["__split"] = "test"
    all_df = pd.concat([train, test], axis=0, ignore_index=True)

    if "event_type" not in all_df.columns:
        raise ValueError("'event_type' kolonu bulunamadı")

    time_col = _detect_time_col(all_df)
    evt_code_map = _event_type_codes(all_df)
    print(f"[features_v2_session] detected time_col={time_col}")

    print("[features_v2_session] aggregating v2 session features...")
    all_feat = _aggregate_features(all_df, time_col, evt_code_map)

    train_sessions = set(train["user_session"].unique())
    test_sessions = set(test["user_session"].unique())
    tr_feat, te_feat = _split_train_test_features(all_feat, train_sessions, test_sessions)

    print(
        f"[features_v2_session] shapes -> all:{all_feat.shape} train_feat:{tr_feat.shape} test_feat:{te_feat.shape}"
    )

    tr_out = _try_write(tr_feat, "session_features_v2_train")
    te_out = _try_write(te_feat, "session_features_v2_test")
    print("[features_v2_session] wrote:")
    print({"train_features": tr_out, "test_features": te_out})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"Hata: {e}", file=sys.stderr)
        raise
