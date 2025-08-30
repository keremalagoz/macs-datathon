"""
Baseline v0 — Global Mean Predictor (sızıntısız referans çizgi)

Amaç:
- Train'de her user_session için tekil session_value etiketini al.
- GroupKFold(user_id, K=5) ile CV MSE hesapla (her fold'da train ortalamasıyla tahmin).
- Test için sample_submission sırasını koruyarak tüm satırlara global train ortalamasını yaz.
- Çıktı: submissions/baseline_v0_mean.csv

Girdiler:
- data/raw/train.csv  (kolonlar: event_type, product_id, category_id, user_id, user_session, session_value)
- data/raw/test.csv   (kolonlar: event_type, product_id, category_id, user_id, user_session)
- data/raw/sample_submission.csv (kolonlar: user_session, session_value)

Notlar:
- Sadece global ortalama kullanılır; user_id bazlı/diğer istatistikler yoktur (sızıntıdan kaçınma ve basitlik için).
- GroupKFold sklearn'süz, deterministik şekilde uygulanmıştır.
"""
from __future__ import annotations
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

DATA_DIR = os.path.join("data", "raw")
SUB_DIR = os.path.join("submissions")

RANDOM_SEED = 42
N_SPLITS = 5


def read_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

    if not (os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(sub_path)):
        raise FileNotFoundError(
            f"Beklenen dosyalar bulunamadı. Lütfen kontrol edin:\n{train_path}\n{test_path}\n{sub_path}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sub_path)

    return train, test, sample_submission


def build_session_labels(train: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"user_id", "user_session", "session_value"}
    missing = required_cols - set(train.columns)
    if missing:
        raise ValueError(f"train.csv eksik kolonlar: {missing}")

    # Her session için tekil etiket ve ilgili user_id'yi al
    # Eğer bir session birden fazla satıra sahipse, ilk görülen değerler kullanılır
    session_df = (
        train[["user_session", "user_id", "session_value"]]
        .dropna(subset=["user_session", "user_id", "session_value"])  # minimum temizlik
        .drop_duplicates(subset=["user_session"], keep="first")
        .reset_index(drop=True)
    )

    if session_df.empty:
        raise ValueError("train içinde user_session bazlı etiket türetilemedi (boş sonuç)")

    return session_df


def group_kfold_indices(groups: np.ndarray, n_splits: int = N_SPLITS, seed: int = RANDOM_SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Basit ve deterministik GroupKFold: benzersiz grup kimliklerini karıştırıp K dilime ayırır.
    Her fold: val grupları = 1 dilim; train grupları = diğer dilimler.
    """
    uniq_groups = pd.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq_groups))
    uniq_groups_shuffled = uniq_groups[perm]

    # Dilimler
    folds = np.array_split(uniq_groups_shuffled, n_splits)

    indices = []
    for k in range(n_splits):
        val_groups = set(folds[k])
        val_mask = np.array([g in val_groups for g in groups])
        train_mask = ~val_mask
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        indices.append((train_idx, val_idx))
    return indices


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def run_cv(session_df: pd.DataFrame) -> Tuple[float, List[float]]:
    y = session_df["session_value"].to_numpy(dtype=float)
    groups = session_df["user_id"].to_numpy()

    fold_indices = group_kfold_indices(groups, n_splits=N_SPLITS, seed=RANDOM_SEED)

    fold_mses: List[float] = []
    for i, (tr_idx, va_idx) in enumerate(fold_indices, start=1):
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        # Her fold'da train ortalaması ile tahmin
        mu = float(np.mean(y_tr))
        yhat_va = np.full_like(y_va, fill_value=mu, dtype=float)
        fold_mse = mse(y_va, yhat_va)
        fold_mses.append(fold_mse)
        print(f"Fold {i}: train_mean={mu:.6f} val_MSE={fold_mse:.6f} (n_tr={len(tr_idx)} n_va={len(va_idx)})")

    cv_mean = float(np.mean(fold_mses))
    print(f"CV MSE (mean over {N_SPLITS} folds): {cv_mean:.6f}")
    return cv_mean, fold_mses


def make_submission(global_mean: float, sample_submission: pd.DataFrame, out_path: str) -> None:
    sub = sample_submission.copy()
    # sample_submission'un kolon sırasını koruyarak sadece target sütununu doldur
    target_col = "session_value"
    if target_col not in sub.columns:
        # Bazı yarışmalarda target kolonu farklı olabilir; bu durumda ekle
        sub[target_col] = global_mean
    else:
        sub[target_col] = float(global_mean)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"Submission yazıldı: {out_path} (constant={global_mean:.6f}, rows={len(sub)})")


def main() -> int:
    print("[baseline_v0_mean] Veri okunuyor...")
    train, test, sample_submission = read_data()

    print("[baseline_v0_mean] Session etiketleri hazırlanıyor...")
    session_df = build_session_labels(train)

    print("[baseline_v0_mean] CV hesaplanıyor (GroupKFold by user_id)...")
    cv_mean, fold_mses = run_cv(session_df)

    print("[baseline_v0_mean] Global ortalama ile submission hazırlanıyor...")
    global_mean = float(session_df["session_value"].mean())
    out_csv = os.path.join(SUB_DIR, "baseline_v0_mean.csv")
    make_submission(global_mean, sample_submission, out_csv)

    print("[baseline_v0_mean] Tamamlandı. Özet:")
    print({
        "cv_mean_mse": round(cv_mean, 6),
        "fold_mses": [round(x, 6) for x in fold_mses],
        "global_mean": round(global_mean, 6),
        "submission": out_csv,
    })

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"Hata: {e}", file=sys.stderr)
        raise
